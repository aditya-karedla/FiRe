"""
nodes package - All workflow nodes
"""

from nodes.company_resolution import (
    fuzzy_match_companies,
    resolve_company_selection
)
from nodes.company_suggestions import (
    suggest_company_names,
    search_suggestions_in_tickers
)
from nodes.company_validation import (
    validate_company_match,
    prepare_hitl_message
)
from nodes.report_generation import generate_report
from nodes.sec_data import (
    extract_financial_statements,
    fetch_sec_data,
    load_company_tickers
)
from nodes.sentiment_analysis import analyze_social_sentiment
from nodes.synthesis import synthesize_research
from nodes.web_research import (
    fetch_company_profile,
    fetch_news_timeline,
    identify_competitors,
    extract_investor_materials
)

__all__ = [
    # Company resolution
    "fuzzy_match_companies",
    "resolve_company_selection",
    "suggest_company_names",
    "search_suggestions_in_tickers",
    "validate_company_match",
    "prepare_hitl_message",
    
    # SEC data
    "load_company_tickers",
    "fetch_sec_data",
    "extract_financial_statements",
    
    # Web research
    "fetch_company_profile",
    "fetch_news_timeline",
    "identify_competitors",
    "extract_investor_materials",
    
    # Sentiment
    "analyze_social_sentiment",
    
    # Synthesis
    "synthesize_research",
    
    # Report
    "generate_report"
]
