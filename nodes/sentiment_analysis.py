"""
LLM-based sentiment analysis using Gemini Flash.
Analyzes social media sentiment with context-aware understanding.
"""

import asyncio
import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from agents.state import ResearchState, SentimentAggregate
from config.prompts import prompts
from config.settings import settings
from nodes.web_research import search_with_fallback
from utils.retry import retry

logger = logging.getLogger(__name__)


async def scrape_social_snippets(company: str) -> List[str]:
    """
    Scrape social media snippets from Reddit, Twitter, forums (last 1 year).
    Returns list of text snippets.
    """
    snippets = []
    
    # Calculate 1-year timeframe for search
    cutoff_date = datetime.utcnow() - timedelta(days=365)
    date_filter = f"after:{cutoff_date.strftime('%Y-%m-%d')}"
    
    # Search queries for different platforms with time constraint
    queries = [
        f"{company} stock reddit {date_filter}",
        f"{company} twitter discussion {date_filter}",
        f"{company} investor forum {date_filter}",
        f"{company} stocktwits recent"
    ]
    
    for query in queries:
        try:
            results = search_with_fallback(query, max_results=8)
            
            for result in results:
                # Use snippet or try to extract from content
                text = result.snippet or result.content or ""
                
                if len(text) > 100:  # Meaningful content
                    # Truncate to max length
                    snippet = text[:settings.SENTIMENT_SNIPPET_LENGTH]
                    snippets.append(snippet)
                
                if len(snippets) >= settings.MAX_SENTIMENT_SAMPLES:
                    break
            
            if len(snippets) >= settings.MAX_SENTIMENT_SAMPLES:
                break
        
        except Exception as e:
            logger.warning(f"Failed to scrape for query '{query}': {e}")
            continue
    
    logger.info(f"Scraped {len(snippets)} social snippets")
    return snippets


# Define structured output schema
class SentimentResult(BaseModel):
    """Structured sentiment analysis result"""
    snippet_index: int = Field(description="Index of the snippet (1-based)")
    sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL", "MIXED"] = Field(description="Overall sentiment classification")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    themes: List[str] = Field(description="List of 1-3 key themes (lowercase, single words)", max_length=3)

class SentimentBatchResults(BaseModel):
    """Batch of sentiment analysis results"""
    results: List[SentimentResult] = Field(description="List of sentiment results for each snippet")


@retry(max_attempts=3)
async def analyze_sentiment_batch(
    company: str,
    snippets: List[str]
) -> List[Dict]:
    """
    Analyze sentiment using Gemini Flash with structured output.
    Returns list of sentiment results.
    """
    if not snippets:
        return []
    
    # Initialize LangChain ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model=settings.SECONDARY_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.GEMINI_TEMPERATURE_VALIDATION,
        max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS_SENTIMENT
    )
    
    # Set up structured output parser
    parser = JsonOutputParser(pydantic_object=SentimentBatchResults)
    
    # Create prompt template with format instructions
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst reviewing social media sentiment. Respond with valid JSON only."),
        ("human", "{prompt}\n\n{format_instructions}")
    ])
    
    # Build chain with structured output
    chain = prompt_template | llm | parser
    
    # Use optimized prompt
    prompt_text = prompts.sentiment_analysis_prompt(company, snippets)
    
    try:
        # Invoke chain with structured output
        result = await chain.ainvoke({
            "prompt": prompt_text,
            "format_instructions": parser.get_format_instructions()
        })
        
        # Extract results list from structured output
        # Handle different possible return types
        if isinstance(result, list):
            # If result is already a list, use it directly
            results = result
        elif isinstance(result, dict):
            # If result is a dict, extract the "results" key
            results = result.get("results", [])
        else:
            # If result is a Pydantic model, access the results attribute
            results = getattr(result, "results", [])
        
        # Convert Pydantic models to dicts if needed
        if results and hasattr(results[0], 'dict'):
            results = [r.dict() for r in results]
        elif results and hasattr(results[0], 'model_dump'):
            # Pydantic v2 uses model_dump instead of dict
            results = [r.model_dump() for r in results]
        
        logger.info(f"✓ Analyzed {len(results)} snippets with Gemini")
        return results
    
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        # Return empty list on parsing errors instead of raising
        if "parse" in str(e).lower() or "json" in str(e).lower():
            logger.warning("Falling back to empty results due to parsing error")
            return []
        raise


async def analyze_social_sentiment(state: ResearchState) -> ResearchState:
    """
    Main sentiment analysis node.
    Scrapes social media and analyzes with Gemini Flash.
    """
    logger.info("Analyzing social sentiment...")
    state.current_node = "sentiment"
    
    if not state.found:
        logger.warning("Company not resolved, skipping sentiment analysis")
        return state
    
    company = state.found.title
    
    # Scrape social snippets
    snippets = await scrape_social_snippets(company)
    
    if not snippets:
        logger.warning("No social snippets found")
        state.social_sentiment = {
            "aggregate": SentimentAggregate(),
            "top_themes": [],
            "samples": []
        }
        return state
    
    # Analyze in batches of 10
    batch_size = 10
    all_results = []
    
    for i in range(0, len(snippets), batch_size):
        batch = snippets[i:i+batch_size]
        
        try:
            batch_results = await analyze_sentiment_batch(company, batch)
            all_results.extend(batch_results)
            
            # Small delay to respect rate limits
            if i + batch_size < len(snippets):
                await asyncio.sleep(1)
        
        except Exception as e:
            logger.warning(f"Batch {i//batch_size + 1} failed: {e}")
            continue
    
    # Aggregate results
    sentiment_counts = {
        "BULLISH": 0,
        "BEARISH": 0,
        "NEUTRAL": 0,
        "MIXED": 0
    }
    
    all_themes = []
    confidence_sum = 0.0
    
    for result in all_results:
        sentiment = result.get("sentiment", "NEUTRAL")
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        confidence_sum += result.get("confidence", 0.5)
        
        themes = result.get("themes", [])
        all_themes.extend(themes)
    
    # Count theme frequencies
    theme_counter = Counter(all_themes)
    
    # Create aggregate
    aggregate = SentimentAggregate(
        total_analyzed=len(all_results),
        bullish=sentiment_counts["BULLISH"],
        bearish=sentiment_counts["BEARISH"],
        neutral=sentiment_counts["NEUTRAL"],
        mixed=sentiment_counts["MIXED"],
        confidence_avg=confidence_sum / len(all_results) if all_results else 0.0,
        top_themes=theme_counter.most_common(10)
    )
    
    state.social_sentiment = {
        "aggregate": aggregate,
        "top_themes": theme_counter.most_common(10),
        "samples": [
            {"text": snippets[i], "sentiment": all_results[i]}
            for i in range(min(len(snippets), len(all_results), 10))
        ]
    }
    
    logger.info(f"✓ Sentiment: {aggregate.summary()}")
    
    return state
