"""
Report generation module.
Creates formatted Markdown and JSON reports.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from agents.state import ResearchState
from config.settings import settings

logger = logging.getLogger(__name__)


def format_markdown_report(state: ResearchState) -> str:
    """
    Generate Markdown-formatted research report.
    """
    if not state.found:
        return "# Research Report\n\nNo company data available."
    
    company = state.found.title
    ticker = state.found.ticker
    cik = state.found.cik_str
    
    # Build report sections
    report_lines = [
        f"# Investment Research Report: {company}",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Ticker:** {ticker} | **CIK:** {cik}",
        f"",
        f"---",
        f""
    ]
    
    # Company Overview
    if state.company_profile:
        profile = state.company_profile
        report_lines.extend([
            f"## Company Overview",
            f"",
            f"**Industry:** {profile.industry or 'N/A'}",
            f"**Founded:** {profile.founded or 'N/A'}",
            f"**Headquarters:** {profile.headquarters or 'N/A'}",
            f"**Employees:** {profile.employees or 'N/A'}",
            f"",
            f"### Business Description",
            f"{profile.description or 'No description available'}",
            f"",
            f"---",
            f""
        ])
    
    # Financial Analysis
    if state.financials_1yr:
        financials = state.financials_1yr
        report_lines.extend([
            f"## Financial Analysis",
            f"",
            f"### Key Metrics",
            f"",
            f"| Statement | Metric | Value | Date |",
            f"|-----------|--------|-------|------|"
        ])
        
        # Income statement
        for key, metric in financials.income_statement.items():
            if metric:
                value_str = f"${metric.value:,.0f}" if metric.value else "N/A"
                date_str = metric.date or metric.form_type or "N/A"
                report_lines.append(f"| Income | {key.replace('_', ' ').title()} | {value_str} | {date_str} |")
        
        # Balance sheet
        for key, metric in financials.balance_sheet.items():
            if metric:
                value_str = f"${metric.value:,.0f}" if metric.value else "N/A"
                date_str = metric.date or metric.form_type or "N/A"
                report_lines.append(f"| Balance | {key.replace('_', ' ').title()} | {value_str} | {date_str} |")
        
        # Cash flow
        for key, metric in financials.cashflow.items():
            if metric:
                value_str = f"${metric.value:,.0f}" if metric.value else "N/A"
                date_str = metric.date or metric.form_type or "N/A"
                report_lines.append(f"| Cashflow | {key.replace('_', ' ').title()} | {value_str} | {date_str} |")
        
        report_lines.extend([f"", f"---", f""])
    
    # Market News
    if state.news_timeline:
        report_lines.extend([
            f"## Recent News & Events",
            f""
        ])
        
        for article in state.news_timeline[:10]:
            report_lines.extend([
                f"### {article.title}",
                f"**Date:** {article.published_date or 'Recent'}",
                f"**Source:** {article.domain or article.source}",
                f"",
                f"{article.snippet or 'No summary available'}",
                f"",
                f"[Read more]({article.url})",
                f""
            ])
        
        report_lines.extend([f"---", f""])
    
    # Social Sentiment
    if state.social_sentiment:
        agg = state.social_sentiment.get("aggregate")
        if agg:
            report_lines.extend([
                f"## Social Sentiment Analysis",
                f"",
                f"**Total Analyzed:** {agg.total_analyzed}",
                f"**Average Confidence:** {agg.confidence_avg:.1%}",
                f"",
                f"### Sentiment Distribution",
                f"",
                f"| Sentiment | Count | Percentage |",
                f"|-----------|-------|------------|",
                f"| 🟢 Bullish | {agg.bullish} | {agg.bullish_ratio:.1%} |",
                f"| 🔴 Bearish | {agg.bearish} | {agg.bearish_ratio:.1%} |",
                f"| ⚪ Neutral | {agg.neutral} | {(agg.neutral/agg.total_analyzed if agg.total_analyzed > 0 else 0):.1%} |",
                f"| 🟡 Mixed | {agg.mixed} | {(agg.mixed/agg.total_analyzed if agg.total_analyzed > 0 else 0):.1%} |",
                f"",
                f"**Overall:** {agg.summary()}",
                f"",
            ])
            
            themes = state.social_sentiment.get("top_themes", [])
            if themes:
                report_lines.extend([
                    f"### Top Discussion Themes",
                    f""
                ])
                for theme, count in themes:
                    report_lines.append(f"- {theme} ({count} mentions)")
                report_lines.append(f"")
            
            report_lines.extend([f"---", f""])
    
    # Competitive Landscape
    if state.competitors:
        report_lines.extend([
            f"## Competitive Landscape",
            f""
        ])
        
        for comp in state.competitors[:5]:
            name = comp.get('name', 'Unknown')
            desc = comp.get('description', 'No description available')
            report_lines.extend([
                f"### {name}",
                f"{desc}",
                f""
            ])
        
        report_lines.extend([f"---", f""])
    
    # Synthesis
    if state.synthesized_insights:
        report_lines.extend([
            f"## Investment Analysis & Synthesis",
            f"",
            state.synthesized_insights,
            f"",
            f"---",
            f""
        ])
    
    # Footer
    report_lines.extend([
        f"",
        f"*This report was generated by Deep Research Agent using data from SEC EDGAR, ",
        f"Tavily web search, and Google Gemini AI. Information is for research purposes only ",
        f"and should not be considered investment advice.*"
    ])
    
    return "\n".join(report_lines)


def format_json_report(state: ResearchState) -> Dict:
    """
    Generate structured JSON report.
    """
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "company_name": state.found.title if state.found else None,
            "ticker": state.found.ticker if state.found else None,
            "cik": state.found.cik_str if state.found else None,
            "pipeline_status": state.status.value if state.status else None
        },
        "company_profile": None,
        "financials": None,
        "news": [],
        "sentiment": None,
        "competitors": state.competitors or [],
        "synthesis": state.synthesized_insights,
        "error_message": state.error_message
    }
    
    # Company profile
    if state.company_profile:
        report["company_profile"] = {
            "industry": state.company_profile.industry,
            "founded": state.company_profile.founded,
            "headquarters": state.company_profile.headquarters,
            "employees": state.company_profile.employees,
            "description": state.company_profile.description,
            "profile_url": state.company_profile.profile_url
        }
    
    # Financials
    if state.financials_1yr:
        financials = state.financials_1yr
        metrics_list = []
        
        for key, metric in financials.income_statement.items():
            if metric:
                metrics_list.append({"statement": "income", "label": key, "value": metric.value, "date": metric.date})
        
        for key, metric in financials.balance_sheet.items():
            if metric:
                metrics_list.append({"statement": "balance", "label": key, "value": metric.value, "date": metric.date})
        
        for key, metric in financials.cashflow.items():
            if metric:
                metrics_list.append({"statement": "cashflow", "label": key, "value": metric.value, "date": metric.date})
        
        report["financials"] = {
            "extraction_date": financials.extraction_date.isoformat(),
            "metrics": metrics_list
        }
    
    # News
    if state.news_timeline:
        report["news"] = [
            {
                "title": n.title,
                "url": n.url,
                "source": n.domain or n.source,
                "date": n.published_date,
                "snippet": n.snippet
            }
            for n in state.news_timeline
        ]
    
    # Sentiment
    if state.social_sentiment:
        agg = state.social_sentiment.get("aggregate")
        if agg:
            report["sentiment"] = {
                "total_analyzed": agg.total_analyzed,
                "bullish": agg.bullish,
                "bearish": agg.bearish,
                "neutral": agg.neutral,
                "mixed": agg.mixed,
                "bullish_ratio": agg.bullish_ratio,
                "bearish_ratio": agg.bearish_ratio,
                "confidence_avg": agg.confidence_avg,
                "summary": agg.summary(),
                "top_themes": state.social_sentiment.get("top_themes", [])
            }
    
    # Competitors
    if state.competitors:
        report["competitors"] = [
            {
                "name": c.get("name", "Unknown") if isinstance(c, dict) else getattr(c, "name", "Unknown"),
                "description": c.get("description", "No description available") if isinstance(c, dict) else getattr(c, "description", "No description available"),
                "website": c.get("website", c.get("source", "")) if isinstance(c, dict) else getattr(c, "website", "")
            }
            for c in state.competitors
        ]
    
    return report


def save_report(
    state: ResearchState,
    output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Save both Markdown and JSON reports to disk.
    Returns dict with file paths.
    """
    if not state.found:
        logger.warning("No company data to save")
        return {}
    
    # Default output directory
    if output_dir is None:
        output_dir = settings.OUTPUTS_DIR / "reports"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize filename
    company_slug = state.found.title.lower().replace(" ", "_")
    company_slug = "".join(c for c in company_slug if c.isalnum() or c == "_")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate reports
    markdown_report = format_markdown_report(state)
    json_report = format_json_report(state)
    
    # Save files
    md_path = output_dir / f"{company_slug}_{timestamp}.md"
    json_path = output_dir / f"{company_slug}_{timestamp}.json"
    
    with open(md_path, "w") as f:
        f.write(markdown_report)
    
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)
    
    logger.info(f"✓ Reports saved:")
    logger.info(f"  - {md_path}")
    logger.info(f"  - {json_path}")
    
    return {
        "markdown": md_path,
        "json": json_path
    }


async def generate_report(state: ResearchState) -> ResearchState:
    """
    Main report generation node.
    Saves reports and updates state.
    """
    logger.info("Generating final report...")
    state.current_node = "report"
    
    try:
        saved_paths = save_report(state)
        state.report_path = str(saved_paths.get("markdown", ""))
        
        logger.info("✓ Report generation complete")
    
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        state.error_message = f"Report generation failed: {str(e)}"
    
    return state
