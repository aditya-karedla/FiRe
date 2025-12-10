"""Validation utilities for ensuring data quality"""

import logging
from typing import Any, Dict, List

from agents.state import ResearchState, ValidationReport

logger = logging.getLogger(__name__)


def validate_state(state: ResearchState) -> ValidationReport:
    """
    Validate research state before synthesis.
    Checks for completeness and data quality.
    """
    warnings = []
    errors = []
    
    # Check company resolution
    if not state.found:
        errors.append("Company not resolved")
    
    # Check financial data
    if state.financials_1yr:
        if not state.financials_1yr.income_statement:
            warnings.append("Missing income statement data")
        if not state.financials_1yr.balance_sheet:
            warnings.append("Missing balance sheet data")
        if not state.financials_1yr.cashflow:
            warnings.append("Missing cash flow data")
    else:
        warnings.append("No financial statements available")
    
    # Check research data
    if not state.company_profile or not state.company_profile.description:
        warnings.append("Company profile incomplete")
    
    if len(state.news_timeline) < 5:
        warnings.append(f"Limited news coverage ({len(state.news_timeline)} items)")
    
    if not state.social_sentiment:
        warnings.append("Sentiment analysis not available")
    
    if len(state.competitors) == 0:
        warnings.append("No competitor data")
    
    # Overall assessment
    passed = len(errors) == 0
    
    if warnings:
        logger.warning(f"Validation warnings: {', '.join(warnings)}")
    if errors:
        logger.error(f"Validation errors: {', '.join(errors)}")
    
    return ValidationReport(
        passed=passed,
        warnings=warnings,
        errors=errors
    )


def check_required_fields(data: Dict[str, Any], required: List[str]) -> List[str]:
    """Check if required fields are present and non-empty"""
    missing = []
    
    for field in required:
        if field not in data or not data[field]:
            missing.append(field)
    
    return missing
