"""
NER Fusion Engine — combines 4 NER models + entity resolution + sentiment.

Loads all models once at startup, processes articles in ~0.9s:
- spaCy trf: people, orgs, money, percentages, dates
- GLiNER: crypto, indices, financial institutions, economic indicators
- Flair OntoNotes: high-confidence ORG/PERSON/MONEY
- ReFinED: Wikidata entity linking (resolves ambiguity)
- FinBERT: sentence-level sentiment
- Override dict: hardcoded financial entity mappings
- Asset registry: keyword-based ticker extraction
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ner_fusion")


@contextmanager
def _log_time(label: str):
    t0 = time.perf_counter()
    yield
    logger.info(f"[NER] {label} took {(time.perf_counter() - t0) * 1000:.1f}ms")


def _gpu_guard(gpu_lock):
    return gpu_lock if gpu_lock is not None else contextlib.nullcontext()
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Cap on extracted clean text fed to the NER models (after boilerplate removal).
# Bounds latency; trafilatura output from full HTML can be long.
_MAX_CLEAN_CHARS = 4000


@dataclass
class ResolvedEntity:
    """A fully resolved entity with canonical identity."""
    text: str
    canonical_name: str
    entity_type: str
    ticker: Optional[str] = None
    asset_class: Optional[str] = None
    wikidata_qid: Optional[str] = None
    role: Optional[str] = None
    confidence: float = 1.0
    source: str = "unknown"
    start: int = 0
    end: int = 0
    sentiment_toward: Optional[str] = None
    is_primary_subject: bool = False
    # All surface strings this entity was matched by (names/aliases/cashtags as
    # they appeared in the article). Used for per-asset aspect sentiment so we
    # find an asset's sentences by what the prose actually says ("Tesla"), not
    # only its ticker symbol ("TSLA"). None for non-keyword-resolved entities.
    surface_forms: Optional[List[str]] = None


@dataclass
class NERResult:
    """Complete NER extraction result for an article."""
    resolved_assets: List[ResolvedEntity] = field(default_factory=list)
    resolved_entities: List[ResolvedEntity] = field(default_factory=list)
    money_values: List[dict] = field(default_factory=list)
    percentages: List[dict] = field(default_factory=list)
    dates: List[dict] = field(default_factory=list)
    sentence_sentiments: List[dict] = field(default_factory=list)
    detected_language: str = "en"


def _load_json(filename: str) -> dict:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


class NERFusionEngine:
    """Loads all NER models at init, processes articles via extract_and_resolve()."""

    def __init__(self, enable_refined: bool = True, enable_flair: bool = True,
                 enable_gliner: bool = True, enable_finbert: bool = True,
                 enable_multilingual: bool = True):
        # Patch curated-transformers FIRST so prefer_gpu()/flair.device work.
        NERFusionEngine._patch_curated_transformers_for_gpu()

        # Pin cuDNN/cuBLAS/RNG before any model loads so miner and validator get
        # reproducible neural output (shrinks cross-run GPU jitter; no quality cost).
        from .determinism import configure_determinism
        configure_determinism()
        self._overrides = _load_json("financial_overrides.json").get("entities", {})
        self._qid_map = _load_json("wikidata_ticker_map.json").get("entities", {})

        from .asset_extractor import AssetExtractor
        from .entity_filter import EntityFilter
        self._asset_extractor = AssetExtractor()
        self._entity_filter = EntityFilter()

        # spaCy — always loaded; prefer_gpu() routes the transformer to CUDA
        # (curated-transformers input tensor device mismatch is fixed by the
        # _patch_curated_transformers_for_gpu() call above).
        import spacy
        try:
            import torch as _torch
            self._cuda = bool(getattr(_torch, "cuda", None) and _torch.cuda.is_available())
        except Exception:
            self._cuda = False
        logger.info(f"[NER] CUDA available for NER models: {self._cuda}")
        if getattr(self, "_cuda", False):
            _gpu_ok = spacy.prefer_gpu()
            logger.info(f"[NER] spaCy GPU enabled: {_gpu_ok}")
        logger.info("[NER] Loading spaCy en_core_web_trf...")
        self._nlp = spacy.load("en_core_web_trf")
        logger.info("[NER] spaCy loaded")

        # GLiNER
        self._gliner = None
        if enable_gliner:
            try:
                from gliner import GLiNER
                logger.info("[NER] Loading GLiNER...")
                self._gliner = GLiNER.from_pretrained("urchade/gliner_base")
                if getattr(self, "_cuda", False):
                    try:
                        self._gliner = self._gliner.to("cuda")
                    except Exception as _e:
                        logger.warning(f"[NER] GLiNER GPU move failed: {_e}")
                logger.info("[NER] GLiNER loaded")
            except Exception as e:
                logger.warning(f"[NER] GLiNER failed to load: {e}")

        # Flair
        self._flair_tagger = None
        if enable_flair:
            try:
                import flair as _flair
                import torch as _torch
                from flair.models import SequenceTagger
                if getattr(self, "_cuda", False):
                    _flair.device = _torch.device("cuda")
                    logger.info("[NER] Flair device set to CUDA")
                logger.info("[NER] Loading Flair OntoNotes...")
                self._flair_tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
                logger.info("[NER] Flair loaded")
            except Exception as e:
                logger.warning(f"[NER] Flair failed to load: {e}")

        # Multilingual NER path (Italian/Russian/etc.) — routed to when the
        # detected language is not English, so English-only spaCy/Flair never
        # see foreign text (fixes §3.3 "domenica"->location).
        self._wikineural = None
        self._gliner_multi = None
        if enable_multilingual:
            try:
                from transformers import pipeline
                logger.info("[NER] Loading WikiNEuRal multilingual NER...")
                self._wikineural = pipeline(
                    "ner", model="Babelscape/wikineural-multilingual-ner",
                    aggregation_strategy="simple",
                    device=0 if getattr(self, "_cuda", False) else -1)
                logger.info("[NER] WikiNEuRal loaded")
            except Exception as e:
                logger.warning(f"[NER] WikiNEuRal failed to load: {e}")
            try:
                from gliner import GLiNER
                logger.info("[NER] Loading GLiNER multilingual...")
                self._gliner_multi = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
                if getattr(self, "_cuda", False):
                    try:
                        self._gliner_multi = self._gliner_multi.to("cuda")
                    except Exception as _e:
                        logger.warning(f"[NER] GLiNER-multi GPU move failed: {_e}")
                logger.info("[NER] GLiNER multilingual loaded")
            except Exception as e:
                logger.warning(f"[NER] GLiNER-multi failed to load: {e}")

        # ReFinED
        self._refined = None
        if enable_refined:
            try:
                from transformers import AutoTokenizer
                _orig = AutoTokenizer.from_pretrained.__func__
                def _patched(cls, *args, **kwargs):
                    kwargs.pop("add_special_tokens", None)
                    return _orig(cls, *args, **kwargs)
                AutoTokenizer.from_pretrained = classmethod(_patched)

                from refined.inference.processor import Refined
                logger.info("[NER] Loading ReFinED aida_model...")
                self._refined = Refined.from_pretrained(model_name="aida_model", entity_set="wikidata",
                                                        device="cuda" if getattr(self, "_cuda", False) else "cpu")
                logger.info("[NER] ReFinED loaded")
                # ReFinED is slow per call; cache unique (text, label) lookups.
                self._refined_cache: Dict[tuple, Optional[tuple]] = {}
                self._refined_cache_maxsize = 8192
            except Exception as e:
                logger.warning(f"[NER] ReFinED failed to load: {e}")

        # FinBERT
        self._finbert = None
        if enable_finbert:
            try:
                from transformers import pipeline
                logger.info("[NER] Loading FinBERT...")
                self._finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                                         device=0 if getattr(self, "_cuda", False) else -1)
                logger.info("[NER] FinBERT loaded")
            except Exception as e:
                logger.warning(f"[NER] FinBERT failed to load: {e}")

        # SentenceTransformer — for article embeddings
        self._embedder = None
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[NER] Loading SentenceTransformer all-MiniLM-L6-v2...")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("[NER] SentenceTransformer loaded")
        except Exception as e:
            logger.warning(f"[NER] SentenceTransformer failed to load: {e}")

        # Main-content extractor (boilerplate removal). Reuses the embedder for
        # the topic-relevance pass.
        from .content_extractor import ContentExtractor
        self._content_extractor = ContentExtractor(embedder=self._embedder)

        logger.info(f"[NER] Fusion engine ready: spaCy=✓ GLiNER={'✓' if self._gliner else '✗'} "
                    f"Flair={'✓' if self._flair_tagger else '✗'} ReFinED={'✓' if self._refined else '✗'} "
                    f"FinBERT={'✓' if self._finbert else '✗'} "
                    f"WikiNEuRal={'✓' if self._wikineural else '✗'} "
                    f"GLiNERmulti={'✓' if self._gliner_multi else '✗'} "
                    f"Embedder={'✓' if self._embedder else '✗'} "
                    f"overrides={len(self._overrides)} qid_map={len(self._qid_map)}")

    def encode_text(self, text: str, gpu_lock=None) -> Optional[List[float]]:
        """Encode text into a 384-dim L2-normalized embedding vector."""
        if self._embedder is None or not text or not text.strip():
            return None
        with _gpu_guard(gpu_lock), _log_time("encode_text"):
            return self._embedder.encode(text.strip(), normalize_embeddings=True).tolist()

    def encode_texts(self, texts: List[str], normalize_embeddings: bool = True,
                     gpu_lock=None) -> Optional[List[List[float]]]:
        """Encode a batch of texts into L2-normalized embedding vectors."""
        if self._embedder is None or not texts:
            return []
        with _gpu_guard(gpu_lock), _log_time("encode_texts"):
            return self._embedder.encode(list(texts), normalize_embeddings=normalize_embeddings).tolist()

    @staticmethod
    def _patch_curated_transformers_for_gpu() -> None:
        """Move curated-transformers BertEmbeddings inputs to the model's device.

        curated-transformers 0.1.1 creates input_ids on CPU even when model
        weights are on CUDA.  Patching only BertEmbeddings is sufficient:
        RobertaEmbeddings.forward delegates to self.inner (a BertEmbeddings),
        so both spaCy en_core_web_trf (RoBERTa) and Flair (XLM-RoBERTa) are
        covered by this single patch.
        NOTE: RobertaEmbeddings has no word_embeddings attribute directly
        (it lives in self.inner) — patching RobertaEmbeddings would AttributeError.
        """
        import torch as _torch

        def _to_device(t, device):
            if isinstance(t, _torch.Tensor):
                return t.to(device, non_blocking=True)
            return t

        try:
            from curated_transformers.models.bert.embeddings import BertEmbeddings
            _orig_bert = BertEmbeddings.forward

            def _bert_forward(self, input_ids, token_type_ids=None, position_ids=None):
                dev = self.word_embeddings.weight.device
                return _orig_bert(
                    self,
                    _to_device(input_ids, dev),
                    _to_device(token_type_ids, dev) if token_type_ids is not None else None,
                    _to_device(position_ids, dev) if position_ids is not None else None,
                )

            BertEmbeddings.forward = _bert_forward

            from curated_transformers.models.attention import ScaledDotProductAttention
            _orig_sdpa = ScaledDotProductAttention.forward

            def _sdpa_forward(self, k, q, v, attn_mask):
                dev = q.device
                if hasattr(attn_mask, "bool_mask") and attn_mask.bool_mask.device != dev:
                    attn_mask.bool_mask = attn_mask.bool_mask.to(dev, non_blocking=True)
                    attn_mask._logit_mask = None  # reset so logit_mask regenerates on GPU
                return _orig_sdpa(self, k, q, v, attn_mask)

            ScaledDotProductAttention.forward = _sdpa_forward
            logger.info("[NER] Patched curated-transformers BertEmbeddings + ScaledDotProductAttention for GPU")
        except Exception as e:
            logger.warning(f"[NER] Could not patch curated-transformers for GPU: {e}")

    def extract_and_resolve(self, title: str, body: str,
                            raw_html: Optional[str] = None,
                            gpu_lock=None) -> NERResult:
        """Full NER pipeline: clean -> detect language -> language-routed
        candidate generation -> confidence/blocklist/offset filtering ->
        resolution. Deterministic end to end.

        When ``raw_html`` is provided, the content extractor runs trafilatura on
        the real DOM (best-in-class boilerplate removal); otherwise it falls
        back to the plain-text block classifier on ``body``.
        """
        from .text_cleaner import clean_text
        from .language_detector import detect_language

        # 1. Detect language on the raw text (robust to boilerplate); drives
        #    both routing and the language-aware content extractor.
        with _log_time("detect_language_and_clean"):
            lang = detect_language(f"{title}\n{body}").code
            result = NERResult(detected_language=lang)

            # 0. Remove site chrome (share bars, CTAs, photo credits, nav) before
            #    anything sees the text — kills §3.1/§3.3 boilerplate leakage.
            #    HTML (raw_html) -> trafilatura; plain text -> jusText-style + relevance.
            clean_title = clean_text(title)
            source = raw_html if (raw_html and raw_html.strip()) else body
            clean_body = self._content_extractor.extract(source, lang, title=title)[:_MAX_CLEAN_CHARS]
            full_text = f"{clean_title}\n{clean_body}"

        # 2. Keyword asset extraction (now language-aware + on cleaned text).
        with _log_time("asset_extraction"):
            asset_matches = self._asset_extractor.extract_assets(clean_title, clean_body, language=lang)
            for m in asset_matches:
                result.resolved_assets.append(ResolvedEntity(
                    text=m.evidence_spans[0] if m.evidence_spans else m.ticker,
                    canonical_name=m.asset_name,
                    entity_type="asset",
                    ticker=m.ticker,
                    asset_class=m.asset_class,
                    confidence=m.disambiguation_confidence,
                    source="keyword",
                    is_primary_subject=m.is_primary_subject,
                    surface_forms=list(dict.fromkeys((m.evidence_spans or []) + [m.asset_name])),
                ))

        # 3. spaCy always runs for MONEY/PERCENT/DATE spans (numeric, language
        #    -tolerant); its entity spans only feed the English path.
        with _gpu_guard(gpu_lock), _log_time("spacy_ner"):
            doc = self._nlp(full_text)
            for ent in doc.ents:
                if ent.label_ == "MONEY":
                    result.money_values.append({"text": ent.text, "start": ent.start_char, "end": ent.end_char})
                elif ent.label_ == "PERCENT":
                    result.percentages.append({"text": ent.text, "start": ent.start_char, "end": ent.end_char})
                elif ent.label_ == "DATE":
                    result.dates.append({"text": ent.text, "start": ent.start_char, "end": ent.end_char})

        # 4. Language-routed entity candidate generation + grounding.
        with _gpu_guard(gpu_lock), _log_time("generate_candidates"):
            candidates = self._generate_candidates(full_text, lang, doc, gpu_lock=gpu_lock)
            grounded = self._entity_filter.filter(candidates, lang)

        # 5. Resolve each grounded candidate to canonical form.
        with _log_time("resolve_entities"):
            seen_tickers = {e.ticker for e in result.resolved_assets if e.ticker}
            for resolved in self._resolve_grounded(grounded, gpu_lock=gpu_lock):
                if resolved:
                    if resolved.ticker and resolved.ticker not in seen_tickers:
                        seen_tickers.add(resolved.ticker)
                        result.resolved_assets.append(resolved)
                    elif resolved.entity_type != "asset":
                        result.resolved_entities.append(resolved)

        # 6. FinBERT sentence sentiments (on cleaned text).
        if self._finbert:
            with _gpu_guard(gpu_lock), _log_time("finbert_sentiment"):
                # Split on sentence punctuation OR newlines, but NOT on decimals
                # ("$94.8" must stay intact) — target-aware ABSA downstream needs whole
                # sentences with the asset named, not fragments truncated at a decimal.
                sentences = [s.strip() for s in re.split(r"(?<!\d)[.!?]+(?!\d)\s+|\n+", full_text)
                             if len(s.strip()) > 20]
                try:
                    batch = sentences[:20]
                    fb_results = self._finbert(batch, truncation=True, max_length=512)
                    for sent, fb_result in zip(batch, fb_results):
                        label = fb_result["label"].lower()
                        direction = "bullish" if label == "positive" else "bearish" if label == "negative" else "neutral"
                        result.sentence_sentiments.append({
                            "text": sent[:200],
                            "sentiment": direction,
                            "score": fb_result["score"],
                        })
                except Exception:
                    pass

        return result

    # spaCy entity labels we keep for the English path.
    _SPACY_LABELS = ("PERSON", "ORG", "GPE", "NORP", "FAC", "EVENT", "PRODUCT")
    _GLINER_LABELS = [
        "person", "company", "organization", "financial institution",
        "cryptocurrency", "stock ticker", "index", "commodity",
        "government body", "regulatory body", "economic indicator",
    ]
    # WikiNEuRal -> spaCy-style label normalization (MISC dropped as noise).
    _WIKINEURAL_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "GPE"}

    def _generate_candidates(self, full_text: str, language: str, doc,
                              gpu_lock=None) -> List["Candidate"]:
        """Route to English or multilingual NER backends and return candidates.

        English: spaCy + GLiNER + Flair. Non-English: WikiNEuRal multilingual +
        GLiNER-multi. Every candidate carries character offsets, the producing
        model, and a real confidence score for downstream agreement scoring.
        """
        from .entity_filter import Candidate, apply_salience
        cands: List[Candidate] = []

        if language == "en":
            for ent in doc.ents:
                if ent.label_ in self._SPACY_LABELS:
                    cands.append(Candidate(ent.text, ent.start_char, ent.end_char,
                                           ent.label_, "spacy", 0.75))
            if self._gliner:
                try:
                    with _gpu_guard(gpu_lock), _log_time("gliner_predict"):
                        ge_list = self._gliner.predict_entities(full_text, self._GLINER_LABELS, threshold=0.4)
                    for ge in ge_list:
                        cands.append(Candidate(ge["text"], ge["start"], ge["end"],
                                               ge["label"], "gliner", float(ge["score"])))
                except Exception as e:
                    logger.debug(f"[NER] GLiNER predict failed: {e}")
            if self._flair_tagger:
                try:
                    from flair.data import Sentence
                    fs = Sentence(full_text)
                    with _gpu_guard(gpu_lock), _log_time("flair_predict"):
                        self._flair_tagger.predict(fs)
                    for ent in fs.get_spans("ner"):
                        if ent.tag in ("PERSON", "ORG"):
                            cands.append(Candidate(ent.text, ent.start_position, ent.end_position,
                                                   ent.tag, "flair", float(ent.score)))
                except Exception as e:
                    logger.debug(f"[NER] Flair predict failed: {e}")
            # Dependency-parse salience: drop candidates stranded in verbless
            # fragments (share bars / nav lists) using the spaCy parse.
            try:
                salient = [(s.start_char, s.end_char) for s in doc.sents
                           if any(t.pos_ in ("VERB", "AUX") for t in s)]
                cands = apply_salience(cands, salient)
            except Exception as e:
                logger.debug(f"[NER] salience pass failed: {e}")
        else:
            if self._wikineural:
                try:
                    with _gpu_guard(gpu_lock), _log_time("wikineural_predict"):
                        ne_list = self._wikineural(full_text)
                    for ne in ne_list:
                        label = self._WIKINEURAL_MAP.get(ne.get("entity_group"))
                        if label:
                            cands.append(Candidate(ne["word"], int(ne["start"]), int(ne["end"]),
                                                   label, "wikineural", float(ne["score"])))
                except Exception as e:
                    logger.debug(f"[NER] WikiNEuRal predict failed: {e}")
            if self._gliner_multi:
                try:
                    with _gpu_guard(gpu_lock), _log_time("gliner_multi_predict"):
                        ge_list = self._gliner_multi.predict_entities(full_text, self._GLINER_LABELS, threshold=0.4)
                    for ge in ge_list:
                        cands.append(Candidate(ge["text"], ge["start"], ge["end"],
                                               ge["label"], "gliner_multi", float(ge["score"])))
                except Exception as e:
                    logger.debug(f"[NER] GLiNER-multi predict failed: {e}")

        return cands

    def _resolve_entity(self, text: str, label: str, source: str,
                        conf: float, start: int, end: int,
                        gpu_lock=None) -> Optional[ResolvedEntity]:
        """Resolve a raw NER entity to canonical form."""
        key = text.lower().strip()

        # Step 1: Override dict
        if key in self._overrides:
            info = self._overrides[key]
            return ResolvedEntity(
                text=text, canonical_name=info["canonical"],
                entity_type=info["type"],
                ticker=info.get("ticker"),
                asset_class=info.get("asset_class"),
                wikidata_qid=info.get("wikidata"),
                role=info.get("role"),
                confidence=1.0, source="override",
                start=start, end=end,
            )

        # Step 2: ReFinED entity linking
        if self._refined:
            try:
                cache_key = (text.lower().strip(), label)
                with _gpu_guard(gpu_lock), _log_time("refined_resolve"):
                    cached = self._refined_cache.get(cache_key)
                    if cached is None and cache_key not in self._refined_cache:
                        spans = self._refined.process_text(text)
                        cached = None
                        for span in spans:
                            if span.predicted_entity and span.predicted_entity.wikidata_entity_id:
                                qid = span.predicted_entity.wikidata_entity_id
                                if qid in self._qid_map:
                                    info = self._qid_map[qid]
                                    cached = (
                                        info.get("name", span.predicted_entity.wikipedia_entity_title or text),
                                        info.get("type", "unknown"),
                                        info.get("ticker"),
                                        info.get("asset_class"),
                                        qid,
                                        info.get("role"),
                                        0.9,
                                        "refined",
                                    )
                                    break
                                if not cached:
                                    wiki_title = span.predicted_entity.wikipedia_entity_title
                                    if wiki_title:
                                        etype = self._infer_type_from_label(label)
                                        cached = (wiki_title, etype, None, None, qid, None, 0.7, "refined")
                                        break
                        if len(self._refined_cache) >= self._refined_cache_maxsize:
                            # Evict an arbitrary entry to cap memory.
                            self._refined_cache.pop(next(iter(self._refined_cache)))
                        self._refined_cache[cache_key] = cached
                    if cached is not None:
                        return ResolvedEntity(
                            text=text,
                            canonical_name=cached[0],
                            entity_type=cached[1],
                            ticker=cached[2],
                            asset_class=cached[3],
                            wikidata_qid=cached[4],
                            role=cached[5],
                            confidence=cached[6],
                            source=cached[7],
                            start=start, end=end,
                        )
            except Exception:
                pass

        # Step 3: Map NER label to our type system
        etype = self._infer_type_from_label(label)
        return ResolvedEntity(
            text=text, canonical_name=text, entity_type=etype,
            confidence=conf, source=source, start=start, end=end,
        )

    def _resolve_grounded(self, grounded: List["Candidate"], gpu_lock=None) -> List[Optional[ResolvedEntity]]:
        """Resolve grounded candidates. Overrides first, then cache, then a single batched ReFinED call."""
        if not grounded:
            return []
        resolved: List[Optional[ResolvedEntity]] = [None] * len(grounded)
        cache = getattr(self, "_refined_cache", None)
        cache_max = getattr(self, "_refined_cache_maxsize", 8192)
        refined = self._refined

        batch_texts: List[str] = []
        batch_idx: List[int] = []
        batch_keys: List[tuple] = []

        for i, c in enumerate(grounded):
            override_key = c.text.lower().strip()
            if override_key in self._overrides:
                info = self._overrides[override_key]
                resolved[i] = ResolvedEntity(
                    text=c.text, canonical_name=info["canonical"],
                    entity_type=info["type"],
                    ticker=info.get("ticker"), asset_class=info.get("asset_class"),
                    wikidata_qid=info.get("wikidata"), role=info.get("role"),
                    confidence=1.0, source="override",
                    start=c.start, end=c.end,
                )
                continue

            if refined is None or cache is None:
                etype = self._infer_type_from_label(c.label)
                resolved[i] = ResolvedEntity(
                    text=c.text, canonical_name=c.text, entity_type=etype,
                    confidence=c.score, source=c.model, start=c.start, end=c.end,
                )
                continue

            cache_key = (c.text, c.label)
            cached = cache.get(cache_key)
            if cached is not None or cache_key in cache:
                if cached is not None:
                    resolved[i] = ResolvedEntity(
                        text=c.text, canonical_name=cached[0], entity_type=cached[1],
                        ticker=cached[2], asset_class=cached[3], wikidata_qid=cached[4],
                        role=cached[5], confidence=cached[6], source=cached[7],
                        start=c.start, end=c.end,
                    )
                else:
                    etype = self._infer_type_from_label(c.label)
                    resolved[i] = ResolvedEntity(
                        text=c.text, canonical_name=c.text, entity_type=etype,
                        confidence=c.score, source=c.model, start=c.start, end=c.end,
                    )
                continue

            batch_texts.append(c.text)
            batch_idx.append(i)
            batch_keys.append(cache_key)

        if batch_texts and refined is not None and cache is not None:
            with _gpu_guard(gpu_lock), _log_time("refined_batch"):
                try:
                    docs = refined.process_text_batch(batch_texts, max_batch_size=8)
                except Exception as e:
                    logger.debug(f"[NER] ReFinED batch failed: {e}")
                    docs = []

            for text, doc, idx, key in zip(batch_texts, docs, batch_idx, batch_keys):
                cache_value: Optional[tuple] = None
                try:
                    for span in doc.spans:
                        pred = getattr(span, "predicted_entity", None)
                        if pred and getattr(pred, "wikidata_entity_id", None):
                            qid = pred.wikidata_entity_id
                            if qid in self._qid_map:
                                info = self._qid_map[qid]
                                cache_value = (
                                    info.get("name", pred.wikipedia_entity_title or text),
                                    info.get("type", "unknown"),
                                    info.get("ticker"),
                                    info.get("asset_class"),
                                    qid,
                                    info.get("role"),
                                    0.9,
                                    "refined",
                                )
                            else:
                                wiki_title = pred.wikipedia_entity_title
                                if wiki_title:
                                    etype = self._infer_type_from_label(grounded[idx].label)
                                    cache_value = (wiki_title, etype, None, None, qid, None, 0.7, "refined")
                            break
                except Exception:
                    pass

                if len(cache) >= cache_max:
                    cache.pop(next(iter(cache)))
                cache[key] = cache_value

                if cache_value is not None:
                    resolved[idx] = ResolvedEntity(
                        text=text, canonical_name=cache_value[0], entity_type=cache_value[1],
                        ticker=cache_value[2], asset_class=cache_value[3], wikidata_qid=cache_value[4],
                        role=cache_value[5], confidence=cache_value[6], source=cache_value[7],
                        start=grounded[idx].start, end=grounded[idx].end,
                    )
                else:
                    etype = self._infer_type_from_label(grounded[idx].label)
                    resolved[idx] = ResolvedEntity(
                        text=text, canonical_name=text, entity_type=etype,
                        confidence=grounded[idx].score, source=grounded[idx].model,
                        start=grounded[idx].start, end=grounded[idx].end,
                    )

            # If batch failed partially, fall back the remainder.
            for j in range(len(docs), len(batch_texts)):
                idx = batch_idx[j]
                key = batch_keys[j]
                c = grounded[idx]
                cache[key] = None
                etype = self._infer_type_from_label(c.label)
                resolved[idx] = ResolvedEntity(
                    text=c.text, canonical_name=c.text, entity_type=etype,
                    confidence=c.score, source=c.model, start=c.start, end=c.end,
                )

        return resolved

    def _infer_type_from_label(self, label: str) -> str:
        """Map NER labels to our entity type system."""
        label_map = {
            "PERSON": "person", "person": "person",
            "ORG": "organization", "company": "organization",
            "organization": "organization", "financial institution": "organization",
            "GPE": "location", "NORP": "government",
            "government body": "government", "regulatory body": "regulatory_body",
            "cryptocurrency": "asset", "stock ticker": "asset",
            "index": "asset", "commodity": "asset",
            "economic indicator": "metric",
            "MONEY": "metric", "PERCENT": "metric",
            "PRODUCT": "product", "EVENT": "event",
        }
        return label_map.get(label, "organization")
