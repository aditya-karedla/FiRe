"""
Deep Research Agent - Streamlit UI
Provides HITL company selection and live progress tracking.
"""

import asyncio
import logging
from pathlib import Path

import streamlit as st

from agents import run_research_pipeline
from config.settings import settings
from nodes.sec_data import load_company_tickers, fuzzy_match_companies
from agents.state import ResearchState
from utils import markdown_to_pdf_bytes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Page config
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .company-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .company-card:hover {
        border-color: #4CAF50;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .status-in-progress {
        background-color: #FFA726;
        color: white;
    }
    .status-completed {
        background-color: #4CAF50;
        color: white;
    }
    .status-failed {
        background-color: #EF5350;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state."""
    if "step" not in st.session_state:
        st.session_state.step = "input"  # input | selection | processing | results
    
    if "company_input" not in st.session_state:
        st.session_state.company_input = ""
    
    if "match_options" not in st.session_state:
        st.session_state.match_options = []
    
    if "selected_company" not in st.session_state:
        st.session_state.selected_company = None
    
    if "final_state" not in st.session_state:
        st.session_state.final_state = None


def render_header():
    """Render page header."""
    st.markdown('<div class="main-header">🔬 Deep Research Agent</div>', unsafe_allow_html=True)
    st.markdown("---")


def render_input_step():
    """Render company input step."""
    st.subheader("Step 1: Enter Company Name")
    
    company_input = st.text_input(
        "Company Name",
        value=st.session_state.company_input,
        placeholder="e.g., Apple Inc, Tesla, Microsoft",
        help="Enter the company you want to research"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        search_clicked = st.button("🔍 Search", type="primary", width="stretch")
    
    with col2:
        if st.button("🔄 Reset", width="stretch"):
            # Reset session
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    if search_clicked and company_input:
        st.session_state.company_input = company_input
        
        with st.spinner("Searching for companies..."):
            # Create initial state
            initial_state = ResearchState(company_name=company_input)
            
            try:
                # Load tickers (synchronous)
                state_with_tickers = load_company_tickers(initial_state)
                
                # Run fuzzy matching (synchronous)
                result_state = fuzzy_match_companies(state_with_tickers)
                
                # If no matches found, try LLM suggestions
                if not result_state.match_options:
                    st.info("🤖 No direct matches found. Trying AI-powered suggestions...")
                    
                    # Run the suggest_and_search node
                    import asyncio
                    from agents.graph_builder import suggest_and_search
                    
                    # Run async function
                    result_state = asyncio.run(suggest_and_search(result_state))
                    
                    # Check if suggestions found anything
                    if not result_state.match_options:
                        st.error("No companies found even with AI suggestions. Please try a different name.")
                        return
                    
                    # Mark that LLM suggestions were used
                    st.session_state.used_llm_suggestions = True
                
                st.session_state.match_options = result_state.match_options
                
                # Always show selection for HITL verification
                # (whether from fuzzy match or LLM suggestions)
                st.session_state.step = "selection"
                st.rerun()
            
            except Exception as e:
                st.error(f"Error searching companies: {e}")
                logging.exception("Company search error")


def render_selection_step():
    """Render company selection step (HITL)."""
    st.subheader("Step 2: Select Company")
    
    # Show different message if LLM suggestions were used
    if st.session_state.get('used_llm_suggestions', False):
        st.success("🤖 AI found these companies based on your input. Please verify and select:")
    else:
        st.info(f"Found {len(st.session_state.match_options)} possible matches. Please select the correct one:")
    
    # Display matches as cards
    for idx, match in enumerate(st.session_state.match_options):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"""
            <div class="company-card">
                <h4>{match.title}</h4>
                <p><strong>Ticker:</strong> {match.ticker}</p>
                <p><strong>CIK:</strong> {match.cik_str}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button(f"Select", key=f"select_{idx}"):
                st.session_state.selected_company = match
                st.session_state.step = "processing"
                st.rerun()
    
    # Back button
    if st.button("⬅️ Back to Search"):
        st.session_state.step = "input"
        st.rerun()


def render_processing_step():
    """Render processing step with live progress."""
    st.subheader("Step 3: Research in Progress")
    
    company = st.session_state.selected_company
    
    st.success(f"Researching: **{company.title}** ({company.ticker})")
    
    # Progress container
    progress_container = st.container()
    
    with progress_container:
        st.info("🔄 Running research pipeline...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Pipeline stages
        stages = [
            "Matching company name",
            "Validating with AI",
            "Fetching SEC data",
            "Building company profile",
            "Extracting financials",
            "Gathering news",
            "Identifying competitors",
            "Analyzing sentiment",
            "Synthesizing research",
            "Generating report"
        ]
        
        # Simulate progress (in real implementation, use callbacks)
        for i, stage in enumerate(stages):
            status_text.text(f"⚙️ {stage}...")
            progress_bar.progress((i + 1) / len(stages))
        
        # Run pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run pipeline with pre-selected company
            final_state = loop.run_until_complete(
                run_research_pipeline(company.title, selected_company=company)
            )
            
            st.session_state.final_state = final_state
            st.session_state.step = "results"
            
            progress_bar.progress(1.0)
            status_text.text("✅ Research complete!")
            
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")
            logging.exception("Pipeline error")
        
        finally:
            loop.close()


def get_state_attr(state, attr, default=None):
    """Safely get attribute from state whether it's dict or object."""
    if isinstance(state, dict):
        return state.get(attr, default)
    else:
        return getattr(state, attr, default)


def render_results_step():
    """Render results step with report."""
    st.subheader("Step 4: Research Results")
    
    state = st.session_state.final_state
    
    if not state:
        st.error("No results available")
        return
    
    # Get status - handle both dict and ResearchState object
    status = get_state_attr(state, 'status')
    if isinstance(status, dict):
        status_value = status.get('value', 'UNKNOWN')
    elif hasattr(status, 'value'):
        status_value = status.value
    else:
        status_value = str(status) if status else 'UNKNOWN'
    
    # Status badge
    status_color = {
        "COMPLETED": "status-completed",
        "IN_PROGRESS": "status-in-progress",
        "FAILED": "status-failed"
    }.get(status_value, "")
    
    st.markdown(f"""
    <span class="status-badge {status_color}">{status_value}</span>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    found = get_state_attr(state, 'found')
    resolved_company_name = get_state_attr(found, 'title') if found else get_state_attr(state, 'company_name', 'Unknown Company')
    if not resolved_company_name:
        resolved_company_name = "Unknown Company"
    
    # Tabs for different views
    tabs = st.tabs([
        "📊 Overview",
        "💰 Financials",
        "📰 News",
        "💬 Sentiment",
        "🏢 Competitors",
        "📝 Full Report"
    ])
    
    # Overview tab
    with tabs[0]:
        company_profile = get_state_attr(state, 'company_profile')
        
        if company_profile:
            # Handle both dict and object
            if isinstance(company_profile, dict):
                profile = company_profile
            else:
                profile = {
                    'industry': getattr(company_profile, 'industry', None),
                    'founded': getattr(company_profile, 'founded', None),
                    'headquarters': getattr(company_profile, 'headquarters', None),
                    'employees': getattr(company_profile, 'employees', None),
                    'description': getattr(company_profile, 'description', None),
                    'profile_url': getattr(company_profile, 'profile_url', None)
                }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Company", resolved_company_name)
                st.metric("Industry", profile.get('industry') or "N/A")
                st.metric("Founded", profile.get('founded') or "N/A")
            
            with col2:
                st.metric("Headquarters", profile.get('headquarters') or "N/A")
                st.metric("Employees", profile.get('employees') or "N/A")
                if profile.get('profile_url'):
                    st.markdown(f"**Profile:** [{profile['profile_url']}]({profile['profile_url']})")
            
            st.subheader("Business Description")
            st.write(profile.get('description') or "No description available")
            
            # Key Products/Services
            if profile.get('key_products'):
                st.subheader("Key Products & Services")
                for product in profile['key_products']:
                    st.markdown(f"- {product}")
            
            # Geographic Presence
            if profile.get('geographic_presence'):
                st.subheader("Geographic Presence")
                st.write(", ".join(profile['geographic_presence']))
            
            # Management Team
            if profile.get('management_team'):
                st.subheader("Management Overview")
                for exec in profile['management_team']:
                    with st.expander(f"{exec.get('name', 'Unknown')} - {exec.get('title', '')}"):
                        st.write(exec.get('background', 'No background information available'))
        else:
            st.info("No company profile available")
    
    # Financials tab
    with tabs[1]:
        financials = get_state_attr(state, 'financials_1yr')
        if financials:
            st.subheader("Key Metrics")
            
            # Handle both dict and object
            if isinstance(financials, dict):
                income = financials.get('income_statement', {})
                balance = financials.get('balance_sheet', {})
                cashflow = financials.get('cashflow', {})
            else:
                income = getattr(financials, 'income_statement', {})
                balance = getattr(financials, 'balance_sheet', {})
                cashflow = getattr(financials, 'cashflow', {})
            
            # Create metrics table
            metrics_data = []
            
            # Add income statement metrics
            for key, metric in income.items():
                if metric:
                    if isinstance(metric, dict):
                        value = metric.get('value')
                        date = metric.get('date', 'N/A')
                    else:
                        value = getattr(metric, 'value', None)
                        date = getattr(metric, 'date', None) or getattr(metric, 'form_type', 'N/A')
                    
                    metrics_data.append({
                        "Statement": "Income",
                        "Metric": key.replace('_', ' ').title(),
                        "Value": f"${value:,.0f}" if value else "N/A",
                        "Date": date
                    })
            
            # Add balance sheet metrics
            for key, metric in balance.items():
                if metric:
                    if isinstance(metric, dict):
                        value = metric.get('value')
                        date = metric.get('date', 'N/A')
                    else:
                        value = getattr(metric, 'value', None)
                        date = getattr(metric, 'date', None) or getattr(metric, 'form_type', 'N/A')
                    
                    metrics_data.append({
                        "Statement": "Balance",
                        "Metric": key.replace('_', ' ').title(),
                        "Value": f"${value:,.0f}" if value else "N/A",
                        "Date": date
                    })
            
            # Add cashflow metrics
            for key, metric in cashflow.items():
                if metric:
                    if isinstance(metric, dict):
                        value = metric.get('value')
                        date = metric.get('date', 'N/A')
                    else:
                        value = getattr(metric, 'value', None)
                        date = getattr(metric, 'date', None) or getattr(metric, 'form_type', 'N/A')
                    
                    metrics_data.append({
                        "Statement": "Cashflow",
                        "Metric": key.replace('_', ' ').title(),
                        "Value": f"${value:,.0f}" if value else "N/A",
                        "Date": date
                    })
            
            if metrics_data:
                st.dataframe(metrics_data, width="stretch")
            else:
                st.info("No financial metrics available")
        else:
            st.info("No financial data available")
    
    # News tab
    with tabs[2]:
        news_timeline = get_state_attr(state, 'news_timeline', [])
        if news_timeline:
            for article in news_timeline[:10]:
                # Handle both dict and object
                if isinstance(article, dict):
                    title = article.get('title', 'Untitled')
                    date = article.get('published_date') or article.get('date', 'Recent')
                    source = article.get('domain') or article.get('source', 'Unknown')
                    snippet = article.get('snippet', 'No summary available')
                    url = article.get('url', '#')
                else:
                    title = getattr(article, 'title', 'Untitled')
                    date = getattr(article, 'published_date', None) or 'Recent'
                    source = getattr(article, 'domain', None) or getattr(article, 'source', 'Unknown')
                    snippet = getattr(article, 'snippet', 'No summary available')
                    url = getattr(article, 'url', '#')
                
                with st.expander(f"📰 {title}"):
                    st.write(f"**Date:** {date}")
                    st.write(f"**Source:** {source}")
                    st.write(snippet)
                    st.markdown(f"[Read more]({url})")
        else:
            st.info("No news available")
    
    # Sentiment tab
    with tabs[3]:
        social_sentiment = get_state_attr(state, 'social_sentiment')
        if social_sentiment:
            agg = social_sentiment.get("aggregate") if isinstance(social_sentiment, dict) else None
            
            if agg:
                # Handle both dict and object
                if isinstance(agg, dict):
                    total = agg.get('total_analyzed', 0)
                    confidence = agg.get('confidence_avg', 0)
                    bullish = agg.get('bullish', 0)
                    bearish = agg.get('bearish', 0)
                    neutral = agg.get('neutral', 0)
                    mixed = agg.get('mixed', 0)
                    summary = agg.get('summary', 'N/A')
                    bullish_ratio = bullish / total if total > 0 else 0
                    bearish_ratio = bearish / total if total > 0 else 0
                else:
                    total = getattr(agg, 'total_analyzed', 0)
                    confidence = getattr(agg, 'confidence_avg', 0)
                    bullish = getattr(agg, 'bullish', 0)
                    bearish = getattr(agg, 'bearish', 0)
                    neutral = getattr(agg, 'neutral', 0)
                    mixed = getattr(agg, 'mixed', 0)
                    summary = agg.summary() if hasattr(agg, 'summary') else 'N/A'
                    bullish_ratio = getattr(agg, 'bullish_ratio', 0)
                    bearish_ratio = getattr(agg, 'bearish_ratio', 0)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Analyzed", total)
                
                with col2:
                    st.metric("Avg Confidence", f"{confidence:.1%}")
                
                with col3:
                    st.metric("Overall", summary)
                
                st.subheader("Sentiment Distribution")
                
                sentiment_data = {
                    "Sentiment": ["Bullish", "Bearish", "Neutral", "Mixed"],
                    "Count": [bullish, bearish, neutral, mixed],
                    "Percentage": [
                        f"{bullish_ratio:.1%}",
                        f"{bearish_ratio:.1%}",
                        f"{(neutral/total if total > 0 else 0):.1%}",
                        f"{(mixed/total if total > 0 else 0):.1%}"
                    ]
                }
                
                st.dataframe(sentiment_data, width="stretch")
                
                # Top themes
                themes = social_sentiment.get("top_themes", []) if isinstance(social_sentiment, dict) else []
                if themes:
                    st.subheader("Top Discussion Themes")
                    for theme, count in themes[:10]:
                        st.write(f"- {theme} ({count} mentions)")
        else:
            st.info("No sentiment data available")
    
    # Competitors tab
    with tabs[4]:
        competitors = get_state_attr(state, 'competitors', [])
        if competitors:
            for comp in competitors:
                # Handle both dict and object
                if isinstance(comp, dict):
                    name = comp.get('name', 'Unknown')
                    desc = comp.get('description', 'No description available')
                    website = comp.get('website')
                else:
                    name = getattr(comp, 'name', 'Unknown')
                    desc = getattr(comp, 'description', 'No description available')
                    website = getattr(comp, 'website', None)
                
                with st.expander(f"🏢 {name}"):
                    st.write(desc)
                    if website:
                        st.markdown(f"[Website]({website})")
        else:
            st.info("No competitor data available")
    
    # Full Report tab
    with tabs[5]:
        synthesized_insights = get_state_attr(state, 'synthesized_insights')
        if synthesized_insights:
            st.markdown(synthesized_insights)
        else:
            st.info("No synthesis available")
        
        # Download buttons
        report_path_str = get_state_attr(state, 'report_path')
        report_path = Path(report_path_str) if report_path_str else None
        json_path = report_path.with_suffix(".json") if report_path else None

        slug_source = resolved_company_name.lower()
        slug = "".join(ch if ch.isalnum() else "_" for ch in slug_source).strip("_") or "report"
        pdf_file_name = f"{slug}.pdf" if not report_path else f"{report_path.stem}.pdf"

        pdf_content = None
        pdf_error = None

        if isinstance(synthesized_insights, str) and synthesized_insights.strip():
            pdf_content = synthesized_insights
        elif report_path and report_path.exists():
            try:
                pdf_content = report_path.read_text()
            except Exception as err:
                pdf_error = f"Could not read Markdown report: {err}"
                logging.exception("Markdown report read failed")
        else:
            fallback_report = get_state_attr(state, 'final_report')
            if isinstance(fallback_report, str) and fallback_report.strip():
                pdf_content = fallback_report

        pdf_bytes = None
        if pdf_content and not pdf_error:
            try:
                pdf_title = f"{resolved_company_name} Research Report"
                pdf_bytes = markdown_to_pdf_bytes(pdf_content, title=pdf_title)
            except Exception as err:
                pdf_error = f"PDF generation failed: {err}"
                logging.exception("PDF generation error")

        col1, col2, col3 = st.columns(3)

        with col1:
            if report_path and report_path.exists():
                with open(report_path, "r") as f:
                    st.download_button(
                        "📥 Download Markdown Report",
                        f.read(),
                        file_name=report_path.name,
                        mime="text/markdown"
                    )
        
        with col2:
            if json_path and json_path.exists():
                with open(json_path, "r") as f:
                    st.download_button(
                        "📥 Download JSON Report",
                        f.read(),
                        file_name=json_path.name,
                        mime="application/json"
                    )

        with col3:
            if pdf_bytes:
                st.download_button(
                    "📥 Download PDF Report",
                    pdf_bytes,
                    file_name=pdf_file_name,
                    mime="application/pdf"
                )
            elif pdf_error:
                st.warning(pdf_error)
    
    # New search button
    st.markdown("---")
    if st.button("🔍 Start New Research"):
        # Reset session
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def render_sidebar():
    """Render a simplified, task-focused sidebar."""
    with st.sidebar:
        st.header("🎯 Session")

        # Show current step and quick context for clarity
        step_titles = {
            "input": "Enter Company",
            "selection": "Select Company",
            "processing": "Research Running",
            "results": "Review Results"
        }
        current_step = st.session_state.get("step", "input")
        st.write(f"**Step:** {step_titles.get(current_step, current_step).title()}")

        selected = st.session_state.get("selected_company")
        if selected:
            st.write(f"**Company:** {getattr(selected, 'title', 'Unknown')} ({getattr(selected, 'ticker', '—')})")

        # Compact progress indicator for intuitiveness
        if current_step == "processing":
            st.progress(50, text="Working…")

        st.markdown("---")

        # Primary actions
        if st.button("🔍 New Research", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.caption("Tip: Start by entering a company name.")


def main():
    """Main app entry point."""
    init_session_state()
    render_header()
    render_sidebar()
    
    # Route to correct step
    if st.session_state.step == "input":
        render_input_step()
    
    elif st.session_state.step == "selection":
        render_selection_step()
    
    elif st.session_state.step == "processing":
        render_processing_step()
    
    elif st.session_state.step == "results":
        render_results_step()


if __name__ == "__main__":
    main()
