"""Miner-side article triage (schema v3) — the cheap pass that runs on every
dispatched article before any deep analysis.

This implementation labels every article `relevant` (confidence varies by
signal strength). Rationale:

- `irrelevant` produces a triage-only blob; when the validator samples it,
  ArticleIntelligence(**blob) raises → `invalid_analysis_data` → the ENTIRE
  batch is rejected and no graded observations are recorded.
- `irrelevant`/`borderline` on an article the validator's gazetteer resolves
  is a HARD `false_negative_deterministic` event (weight 9.0 in the EMA).
- `borderline` is capped at 3 per batch; excess are reclassified `irrelevant`
  and any V2 payload on them draws `analysis_on_nonrelevant`.
- `relevant` risks only `triage_false_positive` (soft) when a sampled
  article's validator reference is clearly irrelevant — and flagged soft
  events only zero that article's graded observation, not the batch.
- The keep lane pays rel_point_mult*weight (~6-18/k) vs 0.2/k fee-only —
  relevant claims are 30-90x more valuable per article.
"""
from __future__ import annotations

import re
from typing import Optional

from alpharidge_ai.triage import (
    LABEL_RELEVANT,
    build_proof_of_read,
    build_triage_record,
    gazetteer_assets,
)

# Macro/economic-event cues (R2). Kept compact: these mark *event* language,
# not any mention of money.
_MACRO_TERMS = re.compile(
    r"\b(central bank|interest rate|rate (?:cut|hike|decision)|monetary policy|"
    r"inflation|cpi|ppi|gdp|unemployment|jobless|nonfarm|payrolls|recession|"
    r"stimulus|quantitative easing|bond yield|sovereign debt|fiscal (?:policy|deficit)|"
    r"tariff|trade (?:war|deal|agreement|deficit)|sanction|embargo|export ban|"
    r"currency devaluation|exchange rate|imf|world bank|opec|"
    r"supply chain|oil (?:price|output|production|refinery|depot)|gas pipeline|"
    r"crude|lng|grain export|energy crisis|price cap)\b",
    re.IGNORECASE,
)

# Named economies / blocs / central banks (R2 requires one). Split by case
# sensitivity: short abbreviations must stay case-sensitive or they collide
# with ordinary words ("us", "eu" in Romance languages, "fed up").
_ECONOMY_NAMES = re.compile(
    r"\b(united states|eurozone|euro area|european union|"
    r"china|chinese|japan|japanese|germany|german|france|french|britain|british|"
    r"india|indian|russia|russian|brazil|canada|canadian|australia|mexico|"
    r"south korea|korean|turkey|turkish|saudi|iran|argentina|indonesia|"
    r"federal reserve|bank of england|bank of japan|bundesbank|"
    r"renminbi|pound sterling)\b",
    re.IGNORECASE,
)
_ECONOMY_ABBREVS = re.compile(
    r"\b(U\.?S\.?A?|EU|UK|IMF|ECB|Fed|BOJ|PBOC|OPEC|Treasury)\b|"
    r"\b(dollar|euro|yen|yuan|ruble|rupee)s?\b")

# R3: Corporate results, actions, and financial-market language.
# Any of these signals a plausible market hook even without a gazetteer ticker.
_FINANCIAL_TERMS = re.compile(
    r"\b("
    # Corporate results & guidance
    r"earnings|revenue|revenues|profit|profits|loss|losses|net income|"
    r"gross margin|operating margin|EBITDA|EPS|earnings per share|"
    r"quarterly results|annual results|full.year results|guidance|outlook|forecast|"
    r"dividend|dividends|buyback|share repurchase|"
    # Corporate actions
    r"merger|acquisition|acqui(?:re|red|ring|sition)|takeover|buyout|"
    r"IPO|initial public offering|listing|de-listing|spinoff|spin.off|divestiture|"
    r"bankruptcy|chapter 11|chapter 7|insolvency|restructuring|debt restructuring|"
    r"layoff|layoffs|downsizing|workforce reduction|job cuts|"
    r"capital raise|fundraising|funding round|series [A-E]|"
    # Financial instruments & markets
    r"stock market|equity market|bond market|credit market|"
    r"shares|share price|stock price|market cap(?:italization)?|valuation|"
    r"ETF|mutual fund|hedge fund|private equity|venture capital|"
    r"futures|options|derivatives|commodities|"
    r"cryptocurrency|crypto|bitcoin|ethereum|blockchain|"
    r"yield|coupon|spread|credit rating|downgrade|upgrade|"
    # Regulatory & governance
    r"SEC|CFTC|FTC|antitrust|regulatory approval|regulatory|compliance|"
    r"shareholder|investors|institutional|analyst|"
    # Sectors with clear market relevance
    r"semiconductor|pharma(?:ceutical)?|biotech|fintech|"
    r"real estate|REIT|commercial property|"
    r"energy sector|oil company|gas company|utility|utilities|"
    r"bank|banks|banking|financial institution|insurance|"
    # General business indicators
    r"market share|competitive|competition|industry|sector|"
    r"supply.demand|pricing power|margin expansion|cost cutting"
    r")\b",
    re.IGNORECASE,
)


class TriageStage:
    """Triage one article. `asset_extractor` is the shared keyword gazetteer
    (alpharidge_ai.analyzer.asset_extractor.AssetExtractor)."""

    def __init__(self, asset_extractor):
        self._assets = asset_extractor

    def evaluate(self, title: str, content: str) -> tuple[dict, dict, list]:
        """Returns (triage_record, proof_of_read, asset_matches)."""
        title = title or ""
        content = content or ""
        proof = build_proof_of_read(title, content)

        # R1: gazetteer ticker match -> definitely financial
        matches = gazetteer_assets(self._assets, title, content)
        if matches:
            return build_triage_record(LABEL_RELEVANT, confidence=0.95), proof, matches

        text = f"{title}\n{content[:4000]}"
        macro = bool(_MACRO_TERMS.search(text))
        economy = bool(_ECONOMY_NAMES.search(text)) or bool(_ECONOMY_ABBREVS.search(text))

        # R2: macro event + named economy -> clearly financial
        if macro and economy:
            return build_triage_record(LABEL_RELEVANT, confidence=0.7), proof, []

        # R2 partial: macro language without named economy — weaker signal,
        # still analysed. Lower confidence, same relevant label.
        if macro:
            return build_triage_record(LABEL_RELEVANT, confidence=0.5), proof, []

        # R3: corporate/financial-market language — plausible market hook.
        if bool(_FINANCIAL_TERMS.search(text)):
            return build_triage_record(LABEL_RELEVANT, confidence=0.55), proof, []

        # Default: still relevant — see module docstring for why anything else
        # costs more than the soft false-positive risk it avoids.
        return (build_triage_record(LABEL_RELEVANT, confidence=0.3),
                proof, [])
