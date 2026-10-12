# Baked-in launch env (set BEFORE bittensor/torch import so operators don't have to set them):
#   BT_NO_PARSE_CLI_ARGS   — bittensor 10.4 ignores CLI args (--netuid/--wallet) without this
#   CUBLAS_WORKSPACE_CONFIG — deterministic cuBLAS for cross-host consensus parity (must precede CUDA init)
import os
os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import time
import typing
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
import copy
import asyncio
import bittensor as bt

# Max concurrent background processing threads per miner process.
# Keeps CPU/thread count bounded while still overlapping I/O.
_MAX_MINER_WORKERS = int(os.environ.get("MINER_MAX_WORKERS", "4"))

# Bittensor Miner Template:
import alpharidge_ai

# import base miner class which takes care of most of the boilerplate
from alpharidge_ai.base.miner import BaseMinerNeuron
from alpharidge_ai.analyzer.aspect_sentiment import install_meta_init_guard
from alpharidge_ai.analyzer import setup_analyzer
from alpharidge_ai.analyzer import setup_telegram_analyzer
from alpharidge_ai.analyzer import setup_news_analyzer
from alpharidge_ai.analyzer import setup_article_intelligence_analyzer
from alpharidge_ai.utils.api_models import TweetAnalysisBase, TelegramMessageAnalysis, NewsArticleAnalysisBase
from alpharidge_ai.models.article_intelligence import SCHEMA_VERSION
from alpharidge_ai.triage import (
    FLAG_DISCARD, FLAG_VALUABLE, LABEL_BORDERLINE, LABEL_IRRELEVANT,
    LABEL_RELEVANT, TRIAGE_SCHEMA_VERSION, analysis_indicates_value,
    build_triage_record)

# Must precede every model load — see install_meta_init_guard. A miner that loses the
# race returns nothing for the affected articles and earns no points for them.
install_meta_init_guard()


def _analysis_looks_degraded(intel, content: str) -> bool:
    """True when an analysis is too thin to survive the validator's sampled
    re-comparison — a partial NER/embedding result that still parses.

    Missing embeddings hard-fail Tier-2.5; a substantive article with zero
    resolved entities means the NER stage silently produced nothing. Either
    outcome is worth one retry before submitting."""
    if intel.title_embedding is None or intel.body_embedding is None:
        return True
    if len((content or "").split()) >= 60 and not intel.entities:
        return True
    return False


def _analysis_is_doomed(intel, content: str) -> bool:
    """True only for a guaranteed fail: missing embeddings hard-fail Tier-2.5
    no matter what the reference holds, so the minimal blob is the least-bad
    answer. Zero entities AND zero assets is deliberately NOT doomed — on
    genuinely thin content the reference extracts nothing either, so the
    schema-valid thin analysis can still pass the sampled composite and earn
    credit, while a blob is a discrepancy whenever it is sampled."""
    return intel.title_embedding is None or intel.body_embedding is None


class Miner(BaseMinerNeuron):
    """
    V3 Miner: Processes TweetBatch requests from validators.
    
    The miner receives batches of tweets from validators, analyzes each tweet
    for subnet relevance and sentiment, and returns the enriched batch.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # Initialize analyzer for tweet classification
        bt.logging.info("[Miner] Initializing analyzer...")
        self.analyzer = setup_analyzer()
        self.telegram_analyzer = setup_telegram_analyzer()
        self.news_analyzer = setup_news_analyzer()
        bt.logging.info("[Miner] News analyzer initialized")

        # The V2 analyzer and the triage stage are required; fail startup
        # rather than fall back silently.
        try:
            self.article_intel_analyzer = setup_article_intelligence_analyzer()
            bt.logging.info("[Miner] ArticleIntelligence analyzer initialized")
        except Exception as e:
            raise RuntimeError(
                "[Miner] ArticleIntelligence (V2) analyzer failed to initialize; "
                "the subnet requires triage and triage requires V2. "
                f"Fix the analyzer setup and restart: {e}") from e

        try:
            from alpharidge_ai.analyzer.asset_extractor import AssetExtractor
            from alpharidge_ai.analyzer.triage_stage import TriageStage
            self.triage_stage = TriageStage(AssetExtractor())
            bt.logging.info("[Miner] Article triage stage initialized")
        except Exception as e:
            raise RuntimeError(
                "[Miner] Triage stage failed to initialize (asset gazetteer or "
                "language detector). The subnet requires triage; fix the named "
                f"component and restart: {e}") from e
        bt.logging.info("[Miner] Analyzer initialized")
        # NOTE: we intentionally do NOT reuse a single bt.Dendrite across threads/event-loops.
        # Miner responses are sent back to validators from a background thread with its own event loop.

        # Push-back ordering: concurrent dendrite sends to the same validator race
        # its monotonic-nonce check; a higher-nonce request landing first gets the
        # earlier one rejected ("Nonce is too old") and the batch is dropped.
        self._send_locks: dict = {}
        self._send_locks_guard = threading.Lock()

        # Bounded thread pool for background processing — prevents thread explosion.
        self._executor = ThreadPoolExecutor(
            max_workers=_MAX_MINER_WORKERS,
            thread_name_prefix="miner-bg",
        )
        self._article_executor = ThreadPoolExecutor(
            max_workers=int(os.environ.get("MINER_ARTICLE_WORKERS", "8")),
            thread_name_prefix="miner-article",
        )

        # IMPORTANT: Register a concrete TweetBatch handler on the axon.
        # Bittensor routes requests by synapse class name; attaching only `forward(self, bt.Synapse)`
        # registers the generic `Synapse` endpoint and does *not* register `TweetBatch`.
        self.axon.attach(
            forward_fn=self.forward_tweets,
            blacklist_fn=self.blacklist_tweet_batch,
            priority_fn=self.priority_tweet_batch,
        )
        
        # Register TelegramBatch handler
        self.axon.attach(
            forward_fn=self.forward_telegram_messages,
            blacklist_fn=self.blacklist_telegram_batch,
            priority_fn=self.priority_telegram_batch,
        )

        # Register ArticleBatch handler
        self.axon.attach(
            forward_fn=self.forward_articles,
            blacklist_fn=self.blacklist_article_batch,
            priority_fn=self.priority_article_batch,
        )

        hotkey = self.wallet.hotkey.ss58_address
        bt.logging.info(f"[Miner] V3 miner started with hotkey: {hotkey}")

    def _validator_send_lock(self, validator_hotkey: str) -> threading.Lock:
        with self._send_locks_guard:
            return self._send_locks.setdefault(validator_hotkey, threading.Lock())

    async def blacklist_tweet_batch(
        self, synapse: alpharidge_ai.protocol.TweetBatch
    ) -> typing.Tuple[bool, str]:
        """Typed wrapper so bittensor's axon signature checks pass for TweetBatch."""
        return await self.blacklist(synapse)

    async def priority_tweet_batch(self, synapse: alpharidge_ai.protocol.TweetBatch) -> float:
        """Typed wrapper so bittensor's axon signature checks pass for TweetBatch."""
        return await self.priority(synapse)

    async def blacklist_telegram_batch(
        self, synapse: alpharidge_ai.protocol.TelegramBatch
    ) -> typing.Tuple[bool, str]:
        """Typed wrapper so bittensor's axon signature checks pass for TelegramBatch."""
        return await self.blacklist(synapse)

    async def priority_telegram_batch(self, synapse: alpharidge_ai.protocol.TelegramBatch) -> float:
        """Typed wrapper so bittensor's axon signature checks pass for TelegramBatch."""
        return await self.priority(synapse)

    async def blacklist_article_batch(
        self, synapse: alpharidge_ai.protocol.ArticleBatch
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_article_batch(self, synapse: alpharidge_ai.protocol.ArticleBatch) -> float:
        return await self.priority(synapse)

    async def forward_is_alive(self, synapse: alpharidge_ai.protocol.IsAlive) -> alpharidge_ai.protocol.IsAlive:
        """
        Processes incoming IsAlive synapses from validators.
        """
        synapse.is_alive = True
        return synapse
    
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        """
        Processes incoming synapses. Routes TweetBatch requests to forward_tweets.
        
        Args:
            synapse (bt.Synapse): The incoming synapse request.
            
        Returns:
            bt.Synapse: The processed synapse response.
        """
        if isinstance(synapse, alpharidge_ai.protocol.TweetBatch):
            return await self.forward_tweets(synapse)
        
        bt.logging.warning(f"Received synapse type: {type(synapse).__name__}, but no handler implemented")
        return synapse

    async def forward_tweets(self, synapse: alpharidge_ai.protocol.TweetBatch) -> alpharidge_ai.protocol.TweetBatch:
        """
        Processes TweetBatch requests from validators.
        
        Spawns a background thread to analyze tweets and send results back to the validator.
        Returns immediately to avoid blocking the axon.
        
        Args:
            synapse: TweetBatch containing list of tweets to analyze
            
        Returns:
            TweetBatch (returns immediately, processing happens in background)
        """
        validator_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        bt.logging.info(f"[Miner] Received TweetBatch with {len(synapse.tweet_batch)} tweet(s) from validator {validator_hotkey}")
        
        if not validator_hotkey:
            bt.logging.warning("[Miner] No validator hotkey found in synapse, cannot send response back")
            return synapse
        
        # Make a deep copy of the synapse for background processing
        synapse_copy = copy.deepcopy(synapse)
        
        # Submit to bounded thread pool instead of spawning an unbounded thread
        self._executor.submit(self._process_and_send_tweets, synapse_copy, validator_hotkey)
        
        bt.logging.info(f"[Miner] Started background processing for TweetBatch, returning immediately")
        return synapse

    async def forward_telegram_messages(self, synapse: alpharidge_ai.protocol.TelegramBatch) -> alpharidge_ai.protocol.TelegramBatch:
        """
        Processes TelegramBatch requests from validators.
        
        Spawns a background thread to analyze telegram messages and send results back to the validator.
        Returns immediately to avoid blocking the axon.
        
        Args:
            synapse: TelegramBatch containing list of messages to analyze
            
        Returns:
            TelegramBatch (returns immediately, processing happens in background)
        """
        validator_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        bt.logging.info(f"[Miner] Received TelegramBatch with {len(synapse.message_batch)} message(s) from validator {validator_hotkey}")
        
        if not validator_hotkey:
            bt.logging.warning("[Miner] No validator hotkey found in synapse, cannot send response back")
            return synapse
        
        # Make a deep copy of the synapse for background processing
        synapse_copy = copy.deepcopy(synapse)
        
        # Submit to bounded thread pool instead of spawning an unbounded thread
        self._executor.submit(self._process_and_send_telegram_messages, synapse_copy, validator_hotkey)
        
        bt.logging.info(f"[Miner] Started background processing for TelegramBatch, returning immediately")
        return synapse

    async def forward_articles(self, synapse: alpharidge_ai.protocol.ArticleBatch) -> alpharidge_ai.protocol.ArticleBatch:
        validator_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        bt.logging.info(f"[Miner] Received ArticleBatch with {len(synapse.article_batch)} article(s) from validator {validator_hotkey}")

        if not validator_hotkey:
            bt.logging.warning("[Miner] No validator hotkey found in synapse, cannot send response back")
            return synapse

        synapse_copy = copy.deepcopy(synapse)

        self._executor.submit(self._process_and_send_articles, synapse_copy, validator_hotkey)

        bt.logging.info(f"[Miner] Started background processing for ArticleBatch, returning immediately")
        return synapse

    def _process_and_send_tweets(self, synapse: alpharidge_ai.protocol.TweetBatch, validator_hotkey: str):
        """
        Background thread function to process tweets and send results back to validator.
        
        Args:
            synapse: TweetBatch to process
            validator_hotkey: Hotkey of the validator to send results back to
        """
        try:
            t0 = time.perf_counter()
            bt.logging.info(f"[Miner] Background: Processing {len(synapse.tweet_batch)} tweets")
            
            # Process each tweet
            for tweet in synapse.tweet_batch:
                if not tweet.text:
                    bt.logging.warning(f"[Miner] Skipping tweet {tweet.id} - no text content")
                    continue
                
                # Classify the tweet
                classification = self.analyzer.classify_post(tweet.text)
                
                if classification is None:
                    bt.logging.warning(f"[Miner] Failed to classify tweet {tweet.id}")
                    continue
                
                tweet.analysis = TweetAnalysisBase(
                    sentiment=classification.sentiment.value,
                    asset_id=classification.asset_id,
                    asset_symbol=classification.asset_symbol,
                    content_type=classification.content_type.value,
                    technical_quality=classification.technical_quality.value,
                    market_analysis=classification.market_analysis.value,
                    impact_potential=classification.impact_potential.value,
                )
            
            process_ms = (time.perf_counter() - t0) * 1000
            bt.logging.info(f"[Miner] Background: Finished processing {len(synapse.tweet_batch)} tweets "
                            f"in {process_ms:.1f} ms")

            from alpharidge_ai.utils.miner_signing import sign_items
            miner_signatures, nonces = sign_items(self.wallet.hotkey, synapse.tweet_batch, id_attr="id")
            synapse.miner_signatures = miner_signatures
            synapse.nonces = nonces

            # Find validator UID and axon info from metagraph
            try:
                validator_uid = self.metagraph.hotkeys.index(validator_hotkey)
            except ValueError:
                bt.logging.error(f"[Miner] Validator hotkey {validator_hotkey} not found in metagraph")
                return

            validator_axon = self.metagraph.axons[validator_uid]
            bt.logging.info(f"[Miner] Background: Found validator UID {validator_uid}, sending response via dendrite")

            # Send the processed batch back to the validator
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            dendrite = None
            try:
                dendrite = bt.Dendrite(wallet=self.wallet)
                with self._validator_send_lock(validator_hotkey):
                    responses = loop.run_until_complete(
                        dendrite.forward(
                            axons=[validator_axon],
                            synapse=synapse,
                            timeout=30.0,
                            deserialize=True,
                        )
                    )
                try:
                    status_code = responses[0].dendrite.status_code if responses and responses[0].dendrite else None
                    status_msg = responses[0].dendrite.status_message if responses and responses[0].dendrite else None
                except Exception:
                    status_code, status_msg = None, None
                total_ms = (time.perf_counter() - t0) * 1000
                if status_code != 200:
                    bt.logging.error(f"[Miner] Background: Validator response failed (status={status_code}): {status_msg} "
                                     f"total={total_ms:.1f} ms")
                else:
                    bt.logging.info(
                        f"[Miner] Background: Successfully sent processed TweetBatch back to validator {validator_hotkey} "
                        f"in {total_ms:.1f} ms"
                    )
            except Exception as e:
                bt.logging.error(f"[Miner] Background: Failed to send response to validator: {e}")
            finally:
                try:
                    if dendrite is not None:
                        if hasattr(dendrite, "aclose_session"):
                            loop.run_until_complete(dendrite.aclose_session())
                        elif hasattr(dendrite, "close_session"):
                            dendrite.close_session()
                except Exception:
                    pass
                loop.close()
                
        except Exception as e:
            bt.logging.error(f"[Miner] Background: Error processing tweets: {e}")

    def _process_and_send_telegram_messages(self, synapse: alpharidge_ai.protocol.TelegramBatch, validator_hotkey: str):
        """
        Background thread function to process telegram messages and send results back to validator.
        
        Args:
            synapse: TelegramBatch to process
            validator_hotkey: Hotkey of the validator to send results back to
        """
        try:
            t0 = time.perf_counter()
            bt.logging.info(f"[Miner] Background: Processing {len(synapse.message_batch)} telegram messages")
            
            # Process each message
            for msg in synapse.message_batch:
                if not msg.content:
                    bt.logging.warning(f"[Miner] Skipping telegram message {msg.id} - no content")
                    continue
                
                # Build message dict for analyzer
                messages_for_analysis = [{
                    'message_id': msg.id,
                    'username': msg.sender_username or msg.sender_name,
                    'content': msg.content,
                }]
                
                # Add context messages if available
                if msg.context_messages:
                    for ctx in msg.context_messages:
                        messages_for_analysis.insert(0, {
                            'message_id': ctx.id,
                            'username': ctx.sender_username or ctx.sender_name,
                            'content': ctx.content,
                        })
                
                inherited_asset_id = msg.inherited_asset_id
                
                classification = self.telegram_analyzer.classify_message_group(
                    messages_for_analysis, 
                    asset_id=inherited_asset_id
                )
                
                if classification is None:
                    bt.logging.warning(f"[Miner] Failed to classify telegram message {msg.id}")
                    continue
                
                from datetime import datetime
                msg.analysis = TelegramMessageAnalysis(
                    id=0,
                    message_id=msg.id,
                    sentiment=classification.sentiment.value,
                    asset_id=classification.asset_id,
                    asset_symbol=classification.asset_symbol,
                    content_type=classification.content_type.value,
                    technical_quality=classification.technical_quality.value,
                    market_analysis=classification.market_analysis.value,
                    impact_potential=classification.impact_potential.value,
                    relevance_confidence=classification.relevance_confidence,
                    analyzed_at=datetime.now().isoformat(),
                )
            
            process_ms = (time.perf_counter() - t0) * 1000
            bt.logging.info(f"[Miner] Background: Finished processing {len(synapse.message_batch)} telegram messages "
                            f"in {process_ms:.1f} ms")

            from alpharidge_ai.utils.miner_signing import sign_items
            miner_signatures, nonces = sign_items(self.wallet.hotkey, synapse.message_batch, id_attr="id")
            synapse.miner_signatures = miner_signatures
            synapse.nonces = nonces

            # Find validator UID and axon info from metagraph
            try:
                validator_uid = self.metagraph.hotkeys.index(validator_hotkey)
            except ValueError:
                bt.logging.error(f"[Miner] Validator hotkey {validator_hotkey} not found in metagraph")
                return

            validator_axon = self.metagraph.axons[validator_uid]
            bt.logging.info(f"[Miner] Background: Found validator UID {validator_uid}, sending telegram response via dendrite")

            # Send the processed batch back to the validator
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            dendrite = None
            try:
                dendrite = bt.Dendrite(wallet=self.wallet)
                with self._validator_send_lock(validator_hotkey):
                    responses = loop.run_until_complete(
                        dendrite.forward(
                            axons=[validator_axon],
                            synapse=synapse,
                            timeout=30.0,
                            deserialize=True,
                        )
                    )
                try:
                    status_code = responses[0].dendrite.status_code if responses and responses[0].dendrite else None
                    status_msg = responses[0].dendrite.status_message if responses and responses[0].dendrite else None
                except Exception:
                    status_code, status_msg = None, None
                total_ms = (time.perf_counter() - t0) * 1000
                if status_code != 200:
                    bt.logging.error(f"[Miner] Background: Validator response failed (status={status_code}): {status_msg} "
                                     f"total={total_ms:.1f} ms")
                else:
                    bt.logging.info(
                        f"[Miner] Background: Successfully sent processed TelegramBatch back to validator {validator_hotkey} "
                        f"in {total_ms:.1f} ms"
                    )
            except Exception as e:
                bt.logging.error(f"[Miner] Background: Failed to send telegram response to validator: {e}")
            finally:
                try:
                    if dendrite is not None:
                        if hasattr(dendrite, "aclose_session"):
                            loop.run_until_complete(dendrite.aclose_session())
                        elif hasattr(dendrite, "close_session"):
                            dendrite.close_session()
                except Exception:
                    pass
                loop.close()
                
        except Exception as e:
            bt.logging.error(f"[Miner] Background: Error processing telegram messages: {e}")

    def _process_and_send_articles(self, synapse: alpharidge_ai.protocol.ArticleBatch, validator_hotkey: str):
        try:
            t0 = time.perf_counter()
            bt.logging.info(f"[Miner] Background: Processing {len(synapse.article_batch)} articles")

            # Up to the validator's borderline cap per batch, zero-signal
            # articles take the analyze-then-flag lane: a valuable flag pays
            # the relevance premium without relevance-audit exposure, a
            # discard flag still earns the fee and improves the keep-rate
            # bound factor. Borderline claims are never audited for false
            # negatives. Excess zero-signal articles go out as irrelevant
            # claims — never deep-validated, and a hard false-negative is
            # impossible without a gazetteer hit on either side.
            borderline_slots = threading.BoundedSemaphore(3)

            def _process_single_article(article):
                # NFC-normalise so analysis_data["title"] agrees with the
                # validator's reference.title after synapse round-tripping.
                _title = unicodedata.normalize('NFC', article.title or '').strip()
                if not _title:
                    # Never skip: an absent response fails the size check. A
                    # titleless article still gets a claim and a proof.
                    rec, proof, _ = self.triage_stage.evaluate("", article.content or "")
                    article.analysis = NewsArticleAnalysisBase(
                        sentiment="neutral",
                        analysisData={
                            "schema_version": SCHEMA_VERSION,
                            "triage_schema_version": TRIAGE_SCHEMA_VERSION,
                            "title": _title,
                            "triage": rec,
                            "proof_of_read": proof,
                        },
                    )
                    return

                # Label every article. Relevant and borderline get the full
                # analysis; borderline is then flagged from the analysis result.
                triage_rec, proof, _ = self.triage_stage.evaluate(
                    _title, article.content or "")
                if (triage_rec["label"] == LABEL_IRRELEVANT
                        and borderline_slots.acquire(blocking=False)):
                    triage_rec = build_triage_record(LABEL_BORDERLINE, confidence=0.5)
                if triage_rec["label"] == LABEL_IRRELEVANT:
                    article.analysis = NewsArticleAnalysisBase(
                        sentiment="neutral",
                        analysisData={
                            "schema_version": SCHEMA_VERSION,
                            "triage_schema_version": TRIAGE_SCHEMA_VERSION,
                            "title": _title,
                            "triage": triage_rec,
                            "proof_of_read": proof,
                        },
                    )
                    return

                # V2: Full ArticleIntelligence analysis
                if self.article_intel_analyzer is not None:
                    # One retry on a failed or degraded analysis (transient
                    # NER/LLM errors surface as None or a thin result). The
                    # last non-None result is kept. Only a missing-embedding
                    # result is doomed (guaranteed Tier-2.5 fail) and goes out
                    # as the minimal blob; an empty-extraction result is still
                    # submitted — thin content often matches a thin reference,
                    # and a blob is a discrepancy whenever it is sampled.
                    intel = None
                    for _attempt in range(2):
                        candidate = self.article_intel_analyzer.analyze(
                            article_id=article.id,
                            url=article.url,
                            title=article.title,
                            source=article.source,
                            published=article.published,
                            summary=article.summary,
                            content=article.content,
                            raw_html=getattr(article, "raw_html", None),
                            miner_hotkey=self.wallet.hotkey.ss58_address if self.wallet else None,
                        )
                        if candidate is None:
                            if _attempt == 0:
                                bt.logging.warning(
                                    f"[Miner] Article {article.id}: analysis returned None, retrying once")
                            continue
                        intel = candidate
                        if not _analysis_looks_degraded(intel, article.content):
                            if _attempt:
                                bt.logging.info(
                                    f"[Miner] Article {article.id}: analysis succeeded on retry")
                            break
                        if _attempt == 0:
                            bt.logging.warning(
                                f"[Miner] Article {article.id}: degraded analysis "
                                f"(entities={len(intel.entities)} assets={len(intel.assets)} "
                                f"title_embedding={'set' if intel.title_embedding else 'missing'}), "
                                f"retrying once")
                    if intel is not None and _analysis_looks_degraded(intel, article.content):
                        if _analysis_is_doomed(intel, article.content):
                            bt.logging.warning(
                                f"[Miner] Article {article.id}: analysis still degraded "
                                f"(entities={len(intel.entities)} assets={len(intel.assets)} "
                                f"title_embedding={'set' if intel.title_embedding else 'missing'}), "
                                f"submitting minimal blob")
                            intel = None
                        else:
                            bt.logging.warning(
                                f"[Miner] Article {article.id}: retry still degraded, submitting anyway")
                    if intel is None:
                        # No usable analysis: if the claim was relevant, park it
                        # in a free borderline slot — an unflagged borderline is
                        # returned to the pool untouched, while a blob on a
                        # relevant claim can still be sampled and costs the
                        # whole batch's credit.
                        if (triage_rec["label"] == LABEL_RELEVANT
                                and borderline_slots.acquire(blocking=False)):
                            triage_rec = build_triage_record(
                                LABEL_BORDERLINE, confidence=0.5)
                            bt.logging.info(
                                f"[Miner] Article {article.id}: no usable analysis, "
                                f"downgraded relevant->borderline")
                        # Send the claim + proof so the batch stays complete.
                        # The title keeps the validator's title sweep from
                        # rejecting the batch.
                        article.analysis = NewsArticleAnalysisBase(
                            sentiment="neutral",
                            analysisData={
                                "schema_version": SCHEMA_VERSION,
                                "triage_schema_version": TRIAGE_SCHEMA_VERSION,
                                "title": _title,
                                "triage": triage_rec,
                                "proof_of_read": proof,
                            },
                        )
                        return
                    if intel is not None:
                        analysis_blob = intel.model_dump()
                        if triage_rec["label"] == "borderline":
                            triage_rec["flag"] = (
                                FLAG_VALUABLE if analysis_indicates_value(analysis_blob)
                                else FLAG_DISCARD)
                        analysis_blob["triage_schema_version"] = TRIAGE_SCHEMA_VERSION
                        analysis_blob["triage"] = triage_rec
                        analysis_blob["proof_of_read"] = proof
                        article.analysis = NewsArticleAnalysisBase(
                            sentiment=intel.overall_sentiment.value,
                            sectorId=intel.topic_signature.primary_sector_id,
                            sectorSymbol=intel.topic_signature.primary_sector_symbol,
                            contentType=intel.content_type.value,
                            technicalQuality=intel.technical_quality if isinstance(intel.technical_quality, str) else intel.technical_quality.value if hasattr(intel.technical_quality, 'value') else str(intel.technical_quality),
                            marketAnalysis=intel.market_analysis_type.value,
                            impactPotential=intel.impact_potential.value,
                            relevanceConfidence="high" if intel.assets else "low",
                            overallSentimentScore=intel.overall_sentiment_score,
                            sentimentDirection=intel.sentiment_direction.value,
                            urgency=intel.urgency.value,
                            temporalFocus=intel.temporal_focus.value,
                            factualConfidence=intel.factual_confidence.value,
                            positioningSignal=intel.positioning_signal.value,
                            targetAudience=intel.target_audience.value,
                            credibilityFlag=intel.credibility_flag.value,
                            primaryGeo=intel.primary_geo.value,
                            marketSession=intel.market_session.value,
                            detectedLanguage=intel.detected_language,
                            stalenessFlag=intel.staleness_flag.value,
                            forwardEventType=intel.forward_event_type.value,
                            assets=[a.model_dump() for a in intel.assets],
                            entities=[e.model_dump() for e in intel.entities],
                            economicData=[d.model_dump() for d in intel.economic_data],
                            numericClaims=[c.model_dump() for c in intel.numeric_claims],
                            quotes=[q.model_dump() for q in intel.quotes],
                            contagionLinks=[l.model_dump() for l in intel.contagion_links],
                            chartSummary=intel.chart_summary.model_dump(),
                            eventFingerprint=intel.event_fingerprint.model_dump(),
                            narrativeKeywords=intel.narrative_keywords,
                            topicSignature=intel.topic_signature.model_dump(),
                            textStats=intel.text_stats.model_dump(),
                            inferredImpacts=[i.model_dump() for i in intel.inferred_impacts] if intel.inferred_impacts else None,
                            analysisData=analysis_blob,
                        )
                        bt.logging.info(f"[Miner] V2 analysis complete for article {article.id}: "
                                        f"{len(intel.assets)} assets, {len(intel.entities)} entities, "
                                        f"{len(intel.contagion_links)} contagion links")
                        return

                # V1 fallback: original NewsRelevanceAnalyzer
                classification = self.news_analyzer.classify_article(article.title, article.summary, article.content)

                if classification is None:
                    bt.logging.warning(f"[Miner] Failed to classify article {article.id}")
                    return

                article.analysis = NewsArticleAnalysisBase(
                    sentiment=classification.sentiment.value,
                    sector_id=classification.sector_id,
                    sector_symbol=classification.sector_symbol,
                    content_type=classification.content_type.value,
                    technical_quality=classification.technical_quality.value,
                    market_analysis=classification.market_analysis.value,
                    impact_potential=classification.impact_potential.value,
                    relevance_confidence=classification.relevance_confidence,
                )

            list(self._article_executor.map(_process_single_article, synapse.article_batch))
            process_ms = (time.perf_counter() - t0) * 1000
            bt.logging.info(f"[Miner] Background: Finished processing {len(synapse.article_batch)} articles "
                            f"in {process_ms:.1f} ms")

            from alpharidge_ai.utils.miner_signing import sign_items
            miner_signatures, nonces = sign_items(self.wallet.hotkey, synapse.article_batch, id_attr="id")
            synapse.miner_signatures = miner_signatures
            synapse.nonces = nonces

            try:
                validator_uid = self.metagraph.hotkeys.index(validator_hotkey)
            except ValueError:
                bt.logging.error(f"[Miner] Validator hotkey {validator_hotkey} not found in metagraph")
                return

            validator_axon = self.metagraph.axons[validator_uid]
            bt.logging.info(f"[Miner] Background: Found validator UID {validator_uid}, sending article response via dendrite")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            dendrite = None
            try:
                dendrite = bt.Dendrite(wallet=self.wallet)
                with self._validator_send_lock(validator_hotkey):
                    responses = loop.run_until_complete(
                        dendrite.forward(
                            axons=[validator_axon],
                            synapse=synapse,
                            timeout=30.0,
                            deserialize=True,
                        )
                    )
                try:
                    status_code = responses[0].dendrite.status_code if responses and responses[0].dendrite else None
                    status_msg = responses[0].dendrite.status_message if responses and responses[0].dendrite else None
                except Exception:
                    status_code, status_msg = None, None
                total_ms = (time.perf_counter() - t0) * 1000
                if status_code != 200:
                    bt.logging.error(f"[Miner] Background: Validator response failed (status={status_code}): {status_msg} "
                                     f"total={total_ms:.1f} ms")
                else:
                    bt.logging.info(
                        f"[Miner] Background: Successfully sent processed ArticleBatch back to validator {validator_hotkey} "
                        f"in {total_ms:.1f} ms"
                    )
            except Exception as e:
                bt.logging.error(f"[Miner] Background: Failed to send article response to validator: {e}")
            finally:
                try:
                    if dendrite is not None:
                        if hasattr(dendrite, "aclose_session"):
                            loop.run_until_complete(dendrite.aclose_session())
                        elif hasattr(dendrite, "close_session"):
                            dendrite.close_session()
                except Exception:
                    pass
                loop.close()

        except Exception as e:
            bt.logging.error(f"[Miner] Background: Error processing articles: {e}")

    async def forward_score(self, synapse: alpharidge_ai.protocol.Score) -> alpharidge_ai.protocol.Score:
        """
        Processes incoming Score synapses from validators.
        
        Receives the score that the validator has given this hotkey for a 100-block interval.
        """
        block_window_start = synapse.block_window_start
        block_window_end = synapse.block_window_end
        score = synapse.score
        rewards = synapse.rewards
        penalties = synapse.penalties
        validator_hotkey = synapse.validator_hotkey
        bt.logging.info(
            f"[Score] Epoch blocks {block_window_start}-{block_window_end}: {score:.0f} points"
            f" ({rewards} rewards, {penalties} penalties) from validator {validator_hotkey}"
        )
        return synapse

    async def forward_validation_result(self, synapse: alpharidge_ai.protocol.ValidationResult) -> alpharidge_ai.protocol.ValidationResult:
        """
        Processes incoming ValidationResult synapses from validators.
        
        Receives validation results for a specific post, including whether it passed or failed and why.
        """
        validation_id = synapse.validation_id
        post_id = synapse.post_id
        success = synapse.success
        validator_hotkey = synapse.validator_hotkey
        failure_reason = synapse.failure_reason
        
        if success:
            bt.logging.info(
                f"[ValidationResult] ✓ Post {post_id} PASSED validation from validator {validator_hotkey} "
                f"(validation_id: {validation_id})"
            )
        else:
            failure_code = failure_reason.get("code", "unknown") if failure_reason else "unknown"
            failure_message = failure_reason.get("message", "Unknown error") if failure_reason else "Unknown error"
            bt.logging.warning(
                f"[ValidationResult] ✗ Post {post_id} FAILED validation from validator {validator_hotkey} "
                f"(validation_id: {validation_id}): {failure_code} - {failure_message}"
            )
        
        return synapse

    async def blacklist(
        self, synapse: bt.Synapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contracted via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (bt.Synapse): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        # Check if hotkey is registered BEFORE trying to get its index
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        # Only get uid if hotkey is in metagraph (to avoid IndexError)
        try:
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        except ValueError:
            # Hotkey not found in metagraph (shouldn't happen if check above passed, but be safe)
            bt.logging.warning(f"Hotkey {synapse.dendrite.hotkey} not found in metagraph")
            return True, "Hotkey not in metagraph"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: bt.Synapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (bt.Synapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up when miner exits."""
        self._executor.shutdown(wait=False)
        super().__exit__(exc_type, exc_value, traceback)
        return False


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(60)

