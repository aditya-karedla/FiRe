"""
Research synthesis using Gemini Pro with context management.
Generates comprehensive investment-grade reports.
"""

import logging
import re
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.state import ResearchState
from config.prompts import prompts
from config.settings import settings
from utils.retry import retry

logger = logging.getLogger(__name__)


def normalize_markdown_text(text: str) -> str:
    """
    Normalize markdown text to fix spacing issues.
    
    Fixes common issues where LLM-generated text has:
    - Missing spaces between formatted text (e.g., *word1**word2*)
    - Missing spaces after numbers before text
    - Missing spaces around markdown formatting
    """
    if not text:
        return text
    
    # Fix missing spaces between asterisk-wrapped words
    # Pattern: *word1**word2* -> *word1* *word2*
    text = re.sub(r'\*(\w+)\*\*(\w+)\*', r'*\1* *\2*', text)
    
    # Fix missing spaces: number followed by asterisk and word
    # Pattern: 416*billion -> 416 *billion
    text = re.sub(r'(\d+)\*(\w+)\*', r'\1 *\2*', text)
    
    # Fix pattern: *word*and*word* -> *word* and *word*
    text = re.sub(r'\*(\w+)\*and\*(\w+)\*', r'*\1* and *\2*', text)
    
    # Fix pattern: *word*of*word* -> *word* of *word*
    text = re.sub(r'\*(\w+)\*of\*(\w+)\*', r'*\1* of *\2*', text)
    
    # Fix pattern: *word1word2* where it should be *word1 word2*
    # This is tricky, but we can catch common cases where lowercase follows uppercase
    text = re.sub(r'\*([A-Z][a-z]+)([A-Z][a-z]+)\*', r'*\1 \2*', text)
    
    # Fix missing spaces before opening parentheses after words
    text = re.sub(r'([a-zA-Z])\(', r'\1 (', text)
    
    # Fix multiple spaces
    text = re.sub(r'  +', ' ', text)
    
    return text


def prepare_synthesis_context(state: ResearchState) -> Dict[str, any]:
    """
    Prepare context sections from state for synthesis.
    Returns dict with structured data for the prompt function.
    """
    context: Dict[str, Any] = {}
    
    # Company info from resolved company
    company_name = state.found.title if state.found else state.company_name
    ticker = state.found.ticker if state.found else "N/A"
    
    context["company_name"] = company_name
    context["ticker"] = ticker
    
    # Company profile - return comprehensive profile data
    if state.company_profile:
        profile = state.company_profile
        context["profile_description"] = profile.description or "No description available"
        context["industry"] = profile.industry or "N/A"
        context["sector"] = profile.sector or "N/A"
        context["founded"] = profile.founded or "N/A"
        context["headquarters"] = profile.headquarters or "N/A"
        context["employees"] = profile.employees
        context["key_products"] = profile.key_products or []
        context["geographic_presence"] = profile.geographic_presence or []
        context["management_team"] = profile.management_team or []
    else:
        context["profile_description"] = ""
        context["industry"] = "N/A"
        context["sector"] = "N/A"
        context["founded"] = "N/A"
        context["headquarters"] = "N/A"
        context["employees"] = None
        context["key_products"] = []
        context["geographic_presence"] = []
        context["management_team"] = []
    
    # Financial statements - return structured dict for prompt formatter
    if state.financials_1yr:
        financials = state.financials_1yr
        # Convert FinancialMetric objects to dicts
        fin_dict = {
            "income_statement": {},
            "balance_sheet": {},
            "cashflow": {}
        }
        
        for key, metric in financials.income_statement.items():
            if metric:
                fin_dict["income_statement"][key] = {
                    "value": metric.value,
                    "date": metric.date or metric.form_type
                }
        
        for key, metric in financials.balance_sheet.items():
            if metric:
                fin_dict["balance_sheet"][key] = {
                    "value": metric.value,
                    "date": metric.date or metric.form_type
                }
        
        for key, metric in financials.cashflow.items():
            if metric:
                fin_dict["cashflow"][key] = {
                    "value": metric.value,
                    "date": metric.date or metric.form_type
                }
        
        context["financials"] = fin_dict
    else:
        context["financials"] = {}
    
    # News timeline - return list of dicts
    if state.news_timeline:
        news_items = []
        for article in state.news_timeline[:10]:
            news_items.append({
                "title": article.title,
                "published_date": article.published_date or "Recent",
                "date": article.published_date or "Recent",
                "snippet": article.snippet or ""
            })
        context["news_items"] = news_items
    else:
        context["news_items"] = []
    
    # Sentiment analysis - return structured dict
    if state.social_sentiment:
        agg = state.social_sentiment.get("aggregate")
        if agg:
            context["sentiment_data"] = {
                "aggregate": {
                    "total_analyzed": agg.total_analyzed,
                    "bullish": agg.bullish,
                    "bearish": agg.bearish,
                    "neutral": agg.neutral,
                    "mixed": agg.mixed,
                    "confidence_avg": agg.confidence_avg
                },
                "top_themes": agg.top_themes
            }
        else:
            context["sentiment_data"] = {}
    else:
        context["sentiment_data"] = {}
    
    # Competitors - return list of company names
    if state.competitors:
        comp_list = [comp.get('name', 'Unknown') for comp in state.competitors[:5]]
        context["competitors"] = comp_list
    else:
        context["competitors"] = []
    
    return context


async def manage_context_window(
    context: Dict[str, any],
    max_tokens: int = None
) -> Dict[str, any]:
    """
    Ensure context fits within token limit.
    For structured data, we trim lists and truncate text fields.
    """
    if max_tokens is None:
        max_tokens = settings.SYNTHESIS_CONTEXT_TOKENS
    
    # Trim news items if too many
    if "news_items" in context and len(context["news_items"]) > 10:
        logger.info(f"Trimming news items from {len(context['news_items'])} to 10")
        context["news_items"] = context["news_items"][:10]
    
    # Truncate profile description if too long
    if "profile_description" in context and len(context["profile_description"]) > 2000:
        logger.info("Truncating profile description")
        context["profile_description"] = context["profile_description"][:2000] + "..."
    
    # Trim competitors if too many
    if "competitors" in context and len(context["competitors"]) > 5:
        logger.info(f"Trimming competitors from {len(context['competitors'])} to 5")
        context["competitors"] = context["competitors"][:5]
    
    logger.info("✓ Context prepared for synthesis")
    return context


@retry(max_attempts=3)
async def generate_synthesis_with_gemini(
    company: str,
    context: Dict[str, str],
    use_flash: bool = False
) -> str:
    """
    Generate synthesis using Gemini Pro or Flash with structured output parsing.
    """
    model_name = settings.SECONDARY_MODEL if use_flash else settings.PRIMARY_MODEL
    
    # Initialize LangChain ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.GEMINI_TEMPERATURE_SYNTHESIS,
        max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS_SYNTHESIS,
        top_p=settings.GEMINI_TOP_P,
        top_k=settings.GEMINI_TOP_K
    )
    
    # Set up string output parser for clean text extraction
    parser = StrOutputParser()
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a senior equity research analyst preparing an investment report."),
        ("human", "{prompt}")
    ])
    
    # Build chain with output parser
    chain = prompt_template | llm | parser
    
    # Build prompt content - extract parameters from context
    company_name = context.get("company_name", "Unknown Company")
    ticker = context.get("ticker", "N/A")
    financials = context.get("financials", {})
    news_items = context.get("news_items", [])
    sentiment_data = context.get("sentiment_data", {})
    competitors = context.get("competitors", [])
    profile_desc = context.get("profile_description", "")
    
    prompt_content = prompts.research_synthesis_prompt(
        company_name=company_name,
        ticker=ticker,
        financials=financials,
        news_items=news_items,
        sentiment_data=sentiment_data,
        competitors=competitors,
        profile_description=profile_desc
    )
    
    logger.info(f"Generating synthesis with {model_name}...")
    
    try:
        # Invoke chain with structured output parsing
        synthesis = await chain.ainvoke({"prompt": prompt_content})
        
        # Normalize text to fix spacing issues
        synthesis = normalize_markdown_text(synthesis)
        
        logger.info(f"✓ Generated {len(synthesis)} char synthesis")
        return synthesis
    
    except Exception as e:
        logger.error(f"Synthesis generation failed: {e}")
        raise


async def synthesize_research(state: ResearchState) -> ResearchState:
    """
    Main synthesis node.
    Generates comprehensive research report with fallback.
    """
    logger.info("Synthesizing research report...")
    state.current_node = "synthesis"
    
    if not state.found:
        logger.warning("Company not resolved, skipping synthesis")
        return state
    
    company = state.found.title
    
    # Prepare context
    context = prepare_synthesis_context(state)
    
    if not context:
        logger.warning("No context available for synthesis")
        state.synthesized_insights = "Insufficient data for synthesis."
        return state
    
    # Manage context window
    context = await manage_context_window(context)
    
    # Try with Gemini Pro first
    try:
        synthesis = await generate_synthesis_with_gemini(
            company,
            context,
            use_flash=False
        )
        state.synthesized_insights = synthesis
    
    except Exception as e:
        logger.warning(f"Gemini Pro failed: {e}, falling back to Flash")
        
        # Fallback to Flash
        try:
            synthesis = await generate_synthesis_with_gemini(
                company,
                context,
                use_flash=True
            )
            state.synthesized_insights = synthesis
        
        except Exception as e2:
            logger.error(f"Both models failed: {e2}")
            state.synthesized_insights = "Failed to generate synthesis."
            state.error_message = f"Synthesis node failed: {str(e2)} (fallback attempted)"
    
    return state
