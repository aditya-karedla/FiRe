"""
LLM-based company validation node.
Uses Gemini to validate if the matched company is correct.
"""

import logging
from typing import Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from agents.state import ResearchState
from config.settings import settings
from utils.retry import retry

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Structured validation result from LLM"""
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(description="Confidence level in the match")
    match: Literal["YES", "NO"] = Field(description="Whether the company matches user intent")
    reasoning: str = Field(description="Brief explanation of the validation decision")


@retry(max_attempts=2)
async def validate_company_match(state: ResearchState) -> ResearchState:
    """
    Use LLM to validate if the matched company is correct.
    
    Flow:
    1. If single match found, validate with LLM
    2. If LLM confirms with high confidence → proceed
    3. If LLM has low confidence or rejects → require HITL
    4. If multiple matches → require HITL (existing behavior)
    """
    logger.info("Validating company match with LLM...")
    state.current_node = "validate_match"
    
    if not state.match_options:
        logger.error("No match options to validate")
        state.error_message = "No company matches found"
        return state
    
    # Get the top match (or only match)
    top_match = state.match_options[0]
    
    # Initialize output parser
    parser = JsonOutputParser(pydantic_object=ValidationResult)
    
    # Build validation prompt
    prompt_text = f"""You are a financial analyst assistant helping to validate company identification.

User Input: "{state.company_name}"

Matched Company:
- Name: {top_match.title}
- Ticker: {top_match.ticker}
- CIK: {top_match.cik_str}

Task: Determine if the matched company is correct for the user's input.

Consider:
1. Are the names similar enough (accounting for abbreviations, legal suffixes)?
2. Is this a well-known company that matches the user's likely intent?
3. Could there be confusion with other companies?

Examples:
- User: "Apple" → Match: "Apple Inc." → HIGH confidence, YES match
- User: "Microsoft" → Match: "Microsoft Corp" → HIGH confidence, YES match
- User: "Meta" → Match: "Meta Platforms Inc" → HIGH confidence, YES match
- User: "Amazon" → Match: "Amazon.com Inc" → HIGH confidence, YES match
- User: "Tesla Motors" → Match: "Tesla Inc" → HIGH confidence, YES match (historical name)

{parser.get_format_instructions()}"""

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=settings.SECONDARY_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.GEMINI_TEMPERATURE_VALIDATION,
        max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS_VALIDATION
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a precise financial data validator. Respond with valid JSON only."),
        ("human", "{prompt}")
    ])
    
    chain = prompt_template | llm | parser
    
    try:
        # Get LLM validation (parser returns dict directly)
        result = await chain.ainvoke({"prompt": prompt_text})
        
        # Extract values from structured output
        confidence = result.get("confidence", "LOW").upper()
        match_result = result.get("match", "NO").upper()
        reasoning = result.get("reasoning", "")
        
        logger.info(f"LLM Validation: CONFIDENCE={confidence}, MATCH={match_result}")
        logger.info(f"Reasoning: {reasoning}")
        
        # Store validation results in state
        if not hasattr(state, 'validation_result'):
            state.validation_result = {}
        
        state.validation_result = {
            "confidence": confidence,
            "match": match_result,
            "reasoning": reasoning,
            "validated_company": top_match.dict()
        }
        
        # Decision logic:
        # HIGH confidence + YES → Auto-select
        # Anything else → Require HITL
        if confidence == "HIGH" and match_result == "YES":
            logger.info(f"✓ High confidence match validated: {top_match.title}")
            state.found = top_match
            state.cik10 = top_match.cik10
            state.llm_validation_passed = True
        else:
            logger.warning(f"Low confidence or rejected match - requiring HITL")
            state.llm_validation_passed = False
            # Keep match_options for HITL selection
        
        return state
        
    except Exception as e:
        logger.error(f"LLM validation failed: {e}")
        # On LLM failure, require HITL for safety
        state.llm_validation_passed = False
        state.error_message = f"Validation error: {str(e)}"
        return state


def prepare_hitl_message(state: ResearchState) -> str:
    """
    Prepare a message for HITL company selection.
    """
    if not state.match_options:
        return "No companies found. Please try a different search."
    
    message_lines = [
        f"Found {len(state.match_options)} possible match(es) for '{state.company_name}':",
        ""
    ]
    
    # Show validation result if available
    if hasattr(state, 'validation_result') and state.validation_result:
        val = state.validation_result
        message_lines.extend([
            "🤖 AI Validation:",
            f"   Confidence: {val.get('confidence', 'N/A')}",
            f"   Match: {val.get('match', 'N/A')}",
            f"   Reasoning: {val.get('reasoning', 'N/A')}",
            ""
        ])
    
    # List options
    for i, match in enumerate(state.match_options, 1):
        message_lines.append(f"{i}. {match.title} ({match.ticker}) - CIK: {match.cik_str}")
    
    message_lines.extend([
        "",
        "Please select the correct company by entering the number (1, 2, etc.)",
        "Or enter 'none' if none of these are correct."
    ])
    
    return "\n".join(message_lines)
