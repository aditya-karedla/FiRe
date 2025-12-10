"""
SEC EDGAR data fetching nodes.
Handles company resolution, financial data extraction, and filing retrieval.
"""

import difflib
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import aiohttp
import requests

from agents.state import (
    CompanyMatch,
    FinancialMetric,
    FinancialStatements,
    ResearchState,
)
from config.settings import settings
from utils.cache import JsonFileCache
from utils.retry import retry, rate_limit

logger = logging.getLogger(__name__)


# Initialize cache for company tickers
tickers_cache = JsonFileCache(
    settings.TICKERS_CACHE_FILE,
    ttl=settings.TICKERS_CACHE_TTL
)


@retry(max_attempts=3, exceptions=(requests.RequestException,))
@rate_limit(calls_per_second=settings.SEC_RATE_LIMIT)
def load_company_tickers(state: ResearchState) -> ResearchState:
    """
    Load SEC company tickers database.
    Caches results for 24 hours to be nice to SEC servers.
    """
    logger.info("Loading SEC company tickers...")
    state.current_node = "load_tickers"
    
    # Try cache first
    cached_tickers = tickers_cache.load()
    if cached_tickers:
        logger.info(f"Loaded {len(cached_tickers)} tickers from cache")
        state.company_tickers = cached_tickers
        return state
    
    # Fetch from SEC
    response = requests.get(
        settings.SEC_COMPANY_TICKERS_URL,
        headers=settings.get_headers(),
        timeout=30
    )
    response.raise_for_status()
    
    # SEC returns dict with numeric keys, convert to list
    data = response.json()
    tickers_list = list(data.values()) if isinstance(data, dict) else data
    
    # Cache for next time
    tickers_cache.save(tickers_list)
    
    logger.info(f"Loaded {len(tickers_list)} tickers from SEC")
    state.company_tickers = tickers_list
    
    return state


def fuzzy_match_companies(state: ResearchState) -> ResearchState:
    """
    Find top company matches using fuzzy string matching.
    Returns top 5 candidates for human review.
    """
    logger.info(f"Finding matches for: {state.company_name}")
    state.current_node = "fuzzy_match"
    
    if not state.company_name:
        raise ValueError("company_name is required")
    
    if not state.company_tickers:
        raise ValueError("Company tickers not loaded")
    
    # Extract company titles for matching
    company_titles = [t.get("title", "") for t in state.company_tickers]
    
    # Fuzzy match using difflib
    matches = difflib.get_close_matches(
        state.company_name,
        company_titles,
        n=5,
        cutoff=0.55
    )
    
    # If no fuzzy matches, try substring search
    if not matches:
        query_lower = state.company_name.lower()
        matches = [
            title for title in company_titles
            if query_lower in title.lower()
        ][:5]
    
    # If still no matches, return empty list (LLM suggestions will be triggered)
    if not matches:
        logger.warning(
            f"No matches found for '{state.company_name}'. "
            f"Will try LLM suggestions next."
        )
        state.match_options = []
        return state
    
    # Build CompanyMatch objects
    match_options = []
    for title in matches:
        for ticker_data in state.company_tickers:
            if ticker_data.get("title") == title:
                match_options.append(CompanyMatch(
                    title=ticker_data["title"],
                    ticker=ticker_data["ticker"],
                    cik_str=str(ticker_data["cik_str"])  # Convert to string
                ))
                break
    
    state.match_options = match_options
    logger.info(f"Found {len(match_options)} potential matches")
    
    return state


def resolve_company_selection(state: ResearchState) -> ResearchState:
    """
    Process human's company selection and resolve to final company.
    """
    logger.info("Resolving company selection...")
    state.current_node = "resolve"
    
    if not state.human_response:
        raise ValueError("human_response is required (user must select a company)")
    
    if not state.match_options:
        raise ValueError("No match options available")
    
    # Parse selection (expecting "1", "2", etc.)
    try:
        selection_index = int(state.human_response.strip()) - 1
    except (ValueError, AttributeError):
        raise ValueError(
            f"Invalid selection: '{state.human_response}'. "
            f"Please enter a number between 1 and {len(state.match_options)}"
        )
    
    if selection_index < 0 or selection_index >= len(state.match_options):
        raise IndexError(
            f"Selection {selection_index + 1} is out of range. "
            f"Please choose between 1 and {len(state.match_options)}"
        )
    
    # Set resolved company
    selected = state.match_options[selection_index]
    state.found = selected
    state.cik10 = selected.cik10
    
    logger.info(f"Resolved to: {selected.title} ({selected.ticker})")
    
    return state


@retry(max_attempts=3, exceptions=(aiohttp.ClientError,))
@rate_limit(calls_per_second=settings.SEC_RATE_LIMIT)
async def fetch_sec_data(state: ResearchState) -> ResearchState:
    """
    Fetch companyfacts and submissions from SEC EDGAR API.
    These are the raw JSON files that contain all financial data.
    """
    logger.info(f"Fetching SEC data for CIK: {state.cik10}")
    state.current_node = "sec_fetch"
    
    if not state.cik10:
        raise ValueError("CIK not set - company must be resolved first")
    
    # Build URLs
    companyfacts_url = settings.SEC_COMPANY_FACTS_URL.format(cik=state.cik10)
    submissions_url = settings.SEC_SUBMISSIONS_URL.format(cik=state.cik10)
    
    # Create SSL context that doesn't verify certificates (for development only)
    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Fetch companyfacts
        async with session.get(
            companyfacts_url,
            headers=settings.get_headers(),
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            response.raise_for_status()
            state.companyfacts = await response.json()
            logger.info("✓ Fetched companyfacts")
        
        # Fetch submissions
        async with session.get(
            submissions_url,
            headers=settings.get_headers(),
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            response.raise_for_status()
            state.submissions = await response.json()
            logger.info("✓ Fetched submissions")
    
    return state


def extract_financial_statements(state: ResearchState) -> ResearchState:
    """
    Extract key financial metrics from companyfacts JSON.
    Focuses on most recent reported values for income statement,
    balance sheet, and cash flow.
    """
    logger.info("Extracting financial statements...")
    state.current_node = "extract_financials"
    
    if not state.companyfacts:
        logger.warning("No companyfacts data available")
        return state
    
    facts = state.companyfacts.get("facts", {})
    us_gaap = facts.get("us-gaap", {})
    
    if not us_gaap:
        logger.warning("No US-GAAP data found in companyfacts")
        return state
    
    # Calculate 1-year cutoff date
    cutoff_date = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    logger.info(f"Filtering financial data to last 1 year (from {cutoff_date})")
    
    # Helper function to get latest value for a metric within 1-year timeframe
    def get_latest_metric(element_names: List[str]) -> Optional[FinancialMetric]:
        """Try multiple element names, return latest value within 1-year timeframe"""
        for name in element_names:
            element = us_gaap.get(name)
            if not element:
                continue
            
            units = element.get("units", {})
            # Prefer USD, but accept any unit
            unit_data = units.get("USD") or next(iter(units.values()), [])
            
            if not unit_data:
                continue
            
            # Filter data within 1-year timeframe
            recent_data = [
                item for item in unit_data
                if (item.get("end") or item.get("filed", "")) >= cutoff_date
            ]
            
            if not recent_data:
                continue
            
            # Sort by date (end or filed) and get most recent
            sorted_data = sorted(
                recent_data,
                key=lambda x: x.get("end") or x.get("filed", ""),
                reverse=True
            )
            
            if sorted_data:
                latest = sorted_data[0]
                return FinancialMetric(
                    element=name,
                    value=latest.get("val"),
                    date=latest.get("end") or latest.get("filed"),
                    unit=latest.get("unit", "USD"),
                    form_type=latest.get("form")
                )
        
        return None
    
    # Extract income statement
    income_statement = {
        "revenues": get_latest_metric([
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerExcludingAssessedTax"
        ]),
        "operating_income": get_latest_metric([
            "OperatingIncomeLoss"
        ]),
        "net_income": get_latest_metric([
            "NetIncomeLoss",
            "ProfitLoss"
        ]),
    }
    
    # Extract balance sheet
    balance_sheet = {
        "assets": get_latest_metric([
            "Assets"
        ]),
        "liabilities": get_latest_metric([
            "Liabilities"
        ]),
        "equity": get_latest_metric([
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
        ]),
    }
    
    # Extract cash flow
    cashflow = {
        "operating_cashflow": get_latest_metric([
            "NetCashProvidedByUsedInOperatingActivities"
        ]),
        "investing_cashflow": get_latest_metric([
            "NetCashProvidedByUsedInInvestingActivities"
        ]),
        "financing_cashflow": get_latest_metric([
            "NetCashProvidedByUsedInFinancingActivities"
        ]),
    }
    
    # Create FinancialStatements object
    statements = FinancialStatements(
        income_statement=income_statement,
        balance_sheet=balance_sheet,
        cashflow=cashflow
    )
    
    state.financials_1yr = statements
    
    # Log what we found
    metrics_found = sum(
        1 for stmt in [income_statement, balance_sheet, cashflow]
        for metric in stmt.values()
        if metric is not None
    )
    logger.info(f"✓ Extracted {metrics_found} financial metrics")
    
    return state
