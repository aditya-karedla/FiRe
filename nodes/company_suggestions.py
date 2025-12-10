"""
LLM-powered company name suggestion when no matches found.
"""

import logging

from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.state import CompanyMatch, ResearchState
from config.settings import settings
from utils.retry import retry

logger = logging.getLogger(__name__)


@retry(max_attempts=2)
async def suggest_company_names(state: ResearchState) -> ResearchState:
    """
    Use LLM to suggest alternative company names when no matches found.
    
    The LLM will:
    1. Generate possible official company names
    2. Suggest ticker symbols
    3. Consider abbreviations and variations
    """
    logger.info(f"No matches found for '{state.company_name}', asking LLM for suggestions...")
    state.current_node = "suggest_names"
    
    user_input = state.company_name
    
    # Initialize output parser
    parser = CommaSeparatedListOutputParser()
    
    # Build prompt for LLM
    prompt_text = f"""You are a financial data specialist helping to identify companies in the SEC database.

User Input: "{user_input}"

The user's input did not match any company in the SEC database. Your task is to suggest possible official company names that might match.

Consider:
1. Official legal names (e.g., "Apple" → "Apple Inc.", "Apple Computer Inc.")
2. Common abbreviations (e.g., "FB" → "Meta Platforms Inc", "Facebook Inc")
3. Historical names (companies that were renamed)
4. Spelling variations
5. Full names vs short names

Provide 5-8 alternative names to search for, ordered by likelihood.

Examples:
User: "Google" → Alphabet Inc, Google LLC, Alphabet Inc Class A, Alphabet Inc Class C
User: "FB" → Meta Platforms Inc, Facebook Inc, Meta Platforms Inc Class A
User: "Tesla" → Tesla Inc, Tesla Motors Inc, Tesla Motors

Now suggest alternatives for: "{user_input}"

{parser.get_format_instructions()}
"""

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=settings.SECONDARY_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.GEMINI_TEMPERATURE_SUGGESTIONS,
        max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS_SUGGESTIONS
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a financial data specialist with expertise in company names and SEC filings."),
        ("human", "{prompt}")
    ])
    
    chain = prompt_template | llm | parser
    
    try:
        # Get LLM suggestions (parser returns list directly)
        suggested_names = await chain.ainvoke({"prompt": prompt_text})
        
        # Clean up any empty strings
        suggested_names = [name.strip() for name in suggested_names if name.strip()]
        
        logger.info(f"LLM Suggestions: {', '.join(suggested_names)}")
        logger.info(f"Parsed {len(suggested_names)} alternative names")
        
        # Store suggestions in state
        if not hasattr(state, 'llm_suggestions'):
            state.llm_suggestions = []
        
        state.llm_suggestions = suggested_names
        
        return state
        
    except Exception as e:
        logger.error(f"LLM suggestion failed: {e}")
        state.error_message = f"Could not generate suggestions: {str(e)}"
        state.llm_suggestions = []
        return state


def search_suggestions_in_tickers(state: ResearchState) -> ResearchState:
    """
    Search the LLM-suggested names in the tickers database.
    """
    logger.info("Searching LLM suggestions in tickers database...")
    state.current_node = "search_suggestions"
    
    if not hasattr(state, 'llm_suggestions') or not state.llm_suggestions:
        logger.warning("No LLM suggestions available")
        return state
    
    if not state.company_tickers:
        logger.error("Company tickers not loaded")
        return state
    
    # Extract company titles from tickers
    company_titles = [t.get("title", "") for t in state.company_tickers]
    company_titles_lower = [t.lower() for t in company_titles]
    
    found_matches = []
    
    # Try each suggested name
    for suggestion in state.llm_suggestions:
        suggestion_lower = suggestion.lower()
        
        # Exact match
        if suggestion_lower in company_titles_lower:
            idx = company_titles_lower.index(suggestion_lower)
            ticker_data = state.company_tickers[idx]
            match = CompanyMatch(
                title=ticker_data["title"],
                ticker=ticker_data["ticker"],
                cik_str=str(ticker_data["cik_str"])
            )
            if match not in found_matches:
                found_matches.append(match)
                logger.info(f"✓ Found exact match: {match.title}")
        
        # Partial match (contains)
        else:
            for i, title in enumerate(company_titles):
                title_lower = title.lower()
                # Check if suggestion is a significant part of the title
                if (suggestion_lower in title_lower or 
                    title_lower in suggestion_lower):
                    ticker_data = state.company_tickers[i]
                    match = CompanyMatch(
                        title=ticker_data["title"],
                        ticker=ticker_data["ticker"],
                        cik_str=str(ticker_data["cik_str"])
                    )
                    if match not in found_matches:
                        found_matches.append(match)
                        logger.info(f"✓ Found partial match: {match.title}")
                        break  # One match per suggestion
    
    if found_matches:
        logger.info(f"Found {len(found_matches)} matches from LLM suggestions")
        state.match_options = found_matches[:5]  # Limit to top 5
    else:
        logger.warning("No matches found even with LLM suggestions")
    
    return state
