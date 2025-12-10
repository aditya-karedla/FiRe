"""
LangGraph workflow builder with parallel execution and checkpointing.
"""

import logging

from langgraph.graph import END, StateGraph

from agents.state import PipelineStatus, ResearchState
from nodes import (
    analyze_social_sentiment,
    extract_financial_statements,
    fetch_company_profile,
    fetch_news_timeline,
    fetch_sec_data,
    fuzzy_match_companies,
    generate_report,
    identify_competitors,
    extract_investor_materials,
    load_company_tickers,
    resolve_company_selection,
    suggest_company_names,
    search_suggestions_in_tickers,
    synthesize_research,
    validate_company_match,
)

logger = logging.getLogger(__name__)


def check_if_needs_matching(state: ResearchState) -> str:
    """
    Initial router: Check if we need to do company matching.
    If company already selected, skip directly to data collection.
    """
    if state.found:
        logger.info(f"Company pre-selected: {state.found.title}, skipping match")
        return "skip_match"
    else:
        logger.info("No company selected, starting match process")
        return "needs_match"


def should_validate(state: ResearchState) -> str:
    """
    Router: Check if we should validate with LLM or try suggestions.
    """
    # If company already resolved (pre-selected), skip validation
    if state.found:
        logger.info(f"Company pre-selected: {state.found.title}")
        return "data_collection"
    
    if not state.match_options:
        logger.warning("No company matches found, will try LLM suggestions")
        return "suggest"
    
    # Always validate with LLM first (even single match)
    logger.info(f"Found {len(state.match_options)} match(es), validating with LLM...")
    return "validate"


def check_suggestions(state: ResearchState) -> str:
    """
    Router: After LLM suggestions, check if we found any matches.
    """
    if state.match_options:
        logger.info(f"Found {len(state.match_options)} matches from suggestions")
        return "validate"
    else:
        logger.error("No matches found even with LLM suggestions")
        return "end"


def should_resolve(state: ResearchState) -> str:
    """
    Router: After LLM validation, check if we need HITL.
    """
    # If LLM validated successfully, proceed
    if hasattr(state, 'llm_validation_passed') and state.llm_validation_passed:
        logger.info("✓ LLM validation passed, proceeding")
        return "data_collection"
    
    # If validation failed or low confidence, require HITL
    logger.info("LLM validation failed or low confidence, requiring HITL")
    return "resolve"


def is_resolved(state: ResearchState) -> str:
    """
    Router: Check if company is resolved after HITL.
    """
    if state.found:
        return "data_collection"
    else:
        return "end"


async def suggest_and_search(state: ResearchState) -> ResearchState:
    """
    Combined node: Generate LLM suggestions and search them in tickers.
    """
    logger.info("🤖 Generating alternative company names with LLM...")
    
    # Generate suggestions
    state = await suggest_company_names(state)
    
    # Search suggestions in tickers
    state = search_suggestions_in_tickers(state)
    
    return state


async def parallel_initial_data(state: ResearchState) -> ResearchState:
    """
    Fan-out node: Fetch SEC data and company profile in parallel.
    """
    logger.info("🔀 Parallel: SEC data + Company profile")
    
    import asyncio
    
    # Run both in parallel
    results = await asyncio.gather(
        fetch_sec_data(state),
        fetch_company_profile(state),
        return_exceptions=True
    )
    
    # Merge results
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Parallel task failed: {result}")
        elif isinstance(result, ResearchState):
            # Merge state - update from result state
            if result.companyfacts:
                state.companyfacts = result.companyfacts
            if result.submissions:
                state.submissions = result.submissions
            if result.company_profile:
                state.company_profile = result.company_profile
    
    return state


async def parallel_deep_research(state: ResearchState) -> ResearchState:
    """
    Fan-out node: News, competitors, investor materials, sentiment in parallel.
    """
    logger.info("🔀 Parallel: News + Competitors + Materials + Sentiment")
    
    import asyncio
    
    # Run all in parallel
    results = await asyncio.gather(
        fetch_news_timeline(state),
        identify_competitors(state),
        extract_investor_materials(state),
        analyze_social_sentiment(state),
        return_exceptions=True
    )
    
    # Merge results
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Parallel task failed: {result}")
        elif isinstance(result, ResearchState):
            # Merge state
            if result.news_timeline:
                state.news_timeline = result.news_timeline
            if result.competitors:
                state.competitors = result.competitors
            if hasattr(result, 'investor_docs') and result.investor_docs:
                state.investor_docs = result.investor_docs
            if result.social_sentiment:
                state.social_sentiment = result.social_sentiment
    
    return state


async def mark_in_progress(state: ResearchState) -> ResearchState:
    """Helper node to mark pipeline as in progress."""
    state.status = PipelineStatus.RESEARCH_IN_PROGRESS
    return state


async def mark_complete(state: ResearchState) -> ResearchState:
    """Helper node to mark pipeline as complete."""
    state.status = PipelineStatus.COMPLETED
    logger.info("✓ Pipeline complete!")
    return state


def build_graph() -> StateGraph:
    """
    Build the LangGraph workflow with parallel execution.
    
    Flow:
    1. Match companies (fuzzy search)
    2. Resolve selection (HITL if needed)
    3. Parallel: SEC data + Company profile
    4. Extract financial statements
    5. Parallel: News + Competitors + Materials + Sentiment
    6. Synthesize research
    7. Generate report
    """
    
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("load_tickers", load_company_tickers)
    workflow.add_node("match", fuzzy_match_companies)
    workflow.add_node("suggest", suggest_and_search)
    workflow.add_node("validate", validate_company_match)
    workflow.add_node("resolve", resolve_company_selection)
    workflow.add_node("mark_progress", mark_in_progress)
    workflow.add_node("initial_data", parallel_initial_data)
    workflow.add_node("financials", extract_financial_statements)
    workflow.add_node("deep_research", parallel_deep_research)
    workflow.add_node("synthesis", synthesize_research)
    workflow.add_node("report", generate_report)
    workflow.add_node("mark_complete", mark_complete)
    
    # Define flow - start with check
    workflow.set_entry_point("load_tickers")
    
    # Load Tickers → Check if needs matching
    workflow.add_conditional_edges(
        "load_tickers",
        check_if_needs_matching,
        {
            "needs_match": "match",
            "skip_match": "mark_progress"
        }
    )
    
    # Match → (Suggest | Validate | Data Collection)
    workflow.add_conditional_edges(
        "match",
        should_validate,
        {
            "suggest": "suggest",
            "validate": "validate",
            "data_collection": "mark_progress"
        }
    )
    
    # Suggest → (Validate | End)
    workflow.add_conditional_edges(
        "suggest",
        check_suggestions,
        {
            "validate": "validate",
            "end": END
        }
    )
    
    # Validate → (Resolve | Initial Data)
    workflow.add_conditional_edges(
        "validate",
        should_resolve,
        {
            "resolve": "resolve",
            "data_collection": "mark_progress"
        }
    )
    
    # Resolve → (Initial Data | End)
    workflow.add_conditional_edges(
        "resolve",
        is_resolved,
        {
            "data_collection": "mark_progress",
            "end": END
        }
    )
    
    # Mark Progress → Initial Data
    workflow.add_edge("mark_progress", "initial_data")
    
    # Initial Data → Financials
    workflow.add_edge("initial_data", "financials")
    
    # Financials → Deep Research
    workflow.add_edge("financials", "deep_research")
    
    # Deep Research → Synthesis
    workflow.add_edge("deep_research", "synthesis")
    
    # Synthesis → Report
    workflow.add_edge("synthesis", "report")
    
    # Report → Mark Complete
    workflow.add_edge("report", "mark_complete")
    
    # Mark Complete → End
    workflow.add_edge("mark_complete", END)
    
    # Compile
    graph = workflow.compile()
    logger.info("✓ Graph compiled")
    
    return graph


async def run_research_pipeline(
    company_input: str,
    selected_company = None
) -> ResearchState:
    """
    Run complete research pipeline for a company.
    
    Args:
        company_input: Company name from user
        selected_company: Optional pre-selected CompanyMatch (skips matching phase)
    
    Returns:
        Final ResearchState
    """
    
    # Create initial state
    initial_state = ResearchState(company_name=company_input)
    
    # If company already selected, populate it and skip matching
    if selected_company:
        initial_state.found = selected_company
        initial_state.cik10 = selected_company.cik_str.zfill(10)
        logger.info(f"Starting research with pre-selected: {selected_company.title}")
    else:
        logger.info(f"Starting research pipeline for: {company_input}")
    
    # Build graph
    graph = build_graph()
    
    # Run graph
    final_state = None
    
    async for state in graph.astream(initial_state):
        # state is a dict with node name as key
        for node_name, node_state in state.items():
            logger.info(f"✓ Completed: {node_name}")
            final_state = node_state
    
    return final_state
