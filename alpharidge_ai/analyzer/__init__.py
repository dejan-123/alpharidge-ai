"""
Analyzer module for crypto asset relevance and sentiment analysis.
"""

import json
from pathlib import Path
from typing import List
from alpharidge_ai import config  # Loads .miner_env and .vali_env
from .relevance import AssetRelevanceAnalyzer
from .telegram_relevance import TelegramRelevanceAnalyzer
from .news_relevance import NewsRelevanceAnalyzer
from .article_intelligence_analyzer import ArticleIntelligenceAnalyzer


def setup_analyzer(assets_file: str = None) -> AssetRelevanceAnalyzer:
    """
    Setup analyzer with assets from JSON file.
    
    Args:
        assets_file: Path to assets.json file. If None, uses default location.
        
    Returns:
        Configured AssetRelevanceAnalyzer instance
    """
    if assets_file is None:
        analyzer_dir = Path(__file__).parent
        assets_file = analyzer_dir / "data" / "assets.json"
    
    with open(assets_file, 'r') as f:
        assets_data = json.load(f)
    
    analyzer = AssetRelevanceAnalyzer(assets=assets_data)
    
    return analyzer


def setup_telegram_analyzer(assets_file: str = None) -> TelegramRelevanceAnalyzer:
    """
    Setup telegram analyzer with assets from JSON file.

    Args:
        assets_file: Path to assets.json file. If None, uses default location.

    Returns:
        Configured TelegramRelevanceAnalyzer instance
    """
    if assets_file is None:
        analyzer_dir = Path(__file__).parent
        assets_file = analyzer_dir / "data" / "assets.json"

    with open(assets_file, 'r') as f:
        assets_data = json.load(f)

    analyzer = TelegramRelevanceAnalyzer(assets=assets_data)

    return analyzer


def setup_news_analyzer(sectors_file: str = None) -> NewsRelevanceAnalyzer:
    """
    Setup news analyzer with sectors from JSON file.

    Args:
        sectors_file: Path to sectors.json file. If None, uses default location.

    Returns:
        Configured NewsRelevanceAnalyzer instance
    """
    if sectors_file is None:
        analyzer_dir = Path(__file__).parent
        sectors_file = analyzer_dir / "data" / "sectors.json"

    with open(sectors_file, 'r') as f:
        sectors_data = json.load(f)

    return NewsRelevanceAnalyzer(sectors=sectors_data)


def setup_article_intelligence_analyzer() -> ArticleIntelligenceAnalyzer:
    """Setup the full ArticleIntelligence analyzer.

    Uses model/api_key/llm_base from config (loaded from .miner_env / .vali_env).
    Loads all data files (assets, contagion templates, narratives, dependency graph, source profiles).

    When NLP_SERVER_URL is set in the environment the analyzer is wired to a
    RemoteNEREngine that calls the centralised GPU server for Stage 1 inference
    (NER + embeddings).  Each miner still calls the LLM API with its own API_KEY
    for Stage 2 / Stage 3.  If NLP_SERVER_URL is empty the original local
    NERFusionEngine is used (single-miner / validator mode).

    Returns:
        Configured ArticleIntelligenceAnalyzer instance
    """
    nlp_url = getattr(config, "NLP_SERVER_URL", "")
    if nlp_url:
        from alpharidge_ai.analyzer.remote_ner import RemoteNEREngine
        timeout = float(getattr(config, "NLP_SERVER_TIMEOUT", 120.0))
        remote_engine = RemoteNEREngine(server_url=nlp_url, timeout=timeout)
        return ArticleIntelligenceAnalyzer(ner_engine=remote_engine)
    return ArticleIntelligenceAnalyzer()
