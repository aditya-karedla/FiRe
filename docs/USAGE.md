# Usage Guide - Deep Research Agent

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Valid API keys for Google Gemini and Tavily
- Internet connection for SEC EDGAR and web research

### Initial Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure environment:**
Edit the `.env` file with your credentials:
```env
SEC_USER_AGENT=YourName your@email.com
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

3. **Verify setup:**
```bash
python test_workflow.py
```

## Using the Streamlit UI

### Step 1: Launch the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Step 2: Enter Company Name

In the input field, type the company name you want to research:
- **Examples:** "Apple Inc", "Tesla", "Microsoft Corporation"
- **Tip:** Use the full company name for best results

Click **🔍 Search** to start.

### Step 3: Select Company (HITL)

If multiple companies match your search, you'll see a selection screen:
- Review the company name, ticker, and CIK
- Click **Select** on the correct company
- Click **⬅️ Back to Search** to try a different search

### Step 4: Processing

The agent will now:
1. ✅ Fetch SEC financial data
2. ✅ Build company profile from Wikipedia/IR sites
3. ✅ Extract financial statements (revenue, assets, etc.)
4. ✅ Gather recent news articles
5. ✅ Identify competitors
6. ✅ Analyze social sentiment with Gemini
7. ✅ Synthesize comprehensive report with Gemini Pro
8. ✅ Generate formatted reports

**Progress bar** shows real-time status.

### Step 5: View Results

Once complete, explore results in tabs:

#### 📊 Overview Tab
- Company name, industry, headquarters
- Business description
- Key products and leadership

#### 💰 Financials Tab
- Latest fiscal year data
- Revenue, assets, liabilities
- Key financial metrics
- Filing date and period

#### 📰 News Tab
- Recent articles (past 30 days)
- Source and publication date
- Article summaries
- Direct links to full articles

#### 💬 Sentiment Tab
- Overall sentiment (Bullish/Bearish/Neutral/Mixed)
- Sentiment distribution chart
- Confidence scores
- Top discussion themes

#### 🏢 Competitors Tab
- Major competitors
- Business descriptions
- Website links

#### 📝 Full Report Tab
- Complete AI-generated synthesis
- Investment analysis perspective
- Risk factors and opportunities
- Download as Markdown or JSON

### Step 6: Download Reports

Two download options:
1. **📥 Download Markdown Report** - Human-readable format
2. **📥 Download JSON Report** - Structured data format

Reports are saved to: `outputs/reports/`

### Starting a New Search

Click **🔍 Start New Research** to reset and analyze another company.

## Using the Python API

### Basic Usage

```python
import asyncio
from agents import create_checkpointer, run_research_pipeline

async def research_company(company_name: str):
    # Create checkpointer
    checkpointer = await create_checkpointer("checkpoints.db")
    
    # Run pipeline
    final_state = await run_research_pipeline(
        company_name,
        checkpointer=checkpointer
    )
    
    # Access results
    print(f"Status: {final_state.status}")
    print(f"Company: {final_state.found.title}")
    print(f"Report: {final_state.report_path}")
    
    return final_state

# Run
asyncio.run(research_company("Apple Inc"))
```

### Resuming from Checkpoint

```python
from agents import resume_pipeline, create_checkpointer

async def resume_research(company_name: str):
    checkpointer = await create_checkpointer("checkpoints.db")
    
    # Resume using normalized company name as thread_id
    thread_id = company_name.lower().replace(" ", "_")
    
    final_state = await resume_pipeline(thread_id, checkpointer)
    
    return final_state

asyncio.run(resume_research("apple_inc"))
```

### Accessing State Components

```python
# After running pipeline
state = await run_research_pipeline("Tesla")

# Company profile
if state.company_profile:
    print(f"Industry: {state.company_profile.industry}")
    print(f"Founded: {state.company_profile.founded}")
    print(f"Description: {state.company_profile.description[:200]}")

# Financial statements
if state.financial_statements:
    for metric in state.financial_statements.metrics:
        print(f"{metric.label}: ${metric.value:,.0f}")

# News timeline
if state.news_timeline:
    for article in state.news_timeline[:5]:
        print(f"[{article.date}] {article.title}")

# Sentiment analysis
if state.social_sentiment:
    agg = state.social_sentiment["aggregate"]
    print(f"Sentiment: {agg.summary()}")
    print(f"Bullish: {agg.bullish_ratio:.1%}")
    print(f"Bearish: {agg.bearish_ratio:.1%}")

# Competitors
if state.competitors:
    for comp in state.competitors:
        print(f"- {comp.name}")

# Final synthesis
if state.synthesis:
    print(state.synthesis)
```

### Custom Configuration

```python
from config.settings import settings

# Modify settings before running
settings.MAX_NEWS_RESULTS = 30
settings.MAX_SENTIMENT_SAMPLES = 100
settings.PRIMARY_MODEL = "gemini-1.5-pro-latest"

# Run with custom config
state = await run_research_pipeline("Microsoft")
```

## Advanced Features

### 1. Parallel Execution

The agent automatically runs tasks in parallel:

**Parallel Group 1:**
- SEC data fetching
- Company profile building

**Parallel Group 2:**
- News gathering
- Competitor identification
- Investor materials extraction
- Sentiment analysis

This reduces total execution time by ~3x.

### 2. Checkpointing

All intermediate results are saved to SQLite:

```python
# Check checkpoint history
from agents import get_checkpoint_history

history = await get_checkpoint_history("apple_inc", checkpointer)
print(f"Found {len(history)} checkpoints")
```

**Benefits:**
- Resume after failures
- Inspect intermediate states
- Skip expensive re-computation

### 3. Fallback Strategies

**Web Search:**
- Primary: Tavily (high quality, premium)
- Fallback: DuckDuckGo (free, unlimited)

**LLM Generation:**
- Primary: Gemini Pro 1.5 (highest quality)
- Fallback: Gemini Flash 2.0 (faster, cheaper)

**Automatic switching** when primary fails.

### 4. Schema Validation

All state is validated with Pydantic:

```python
from agents.state import ResearchState

# Invalid state will raise ValidationError
try:
    state = ResearchState(user_input="")  # Empty input
except ValueError as e:
    print(f"Validation error: {e}")
```

### 5. Context Window Management

For companies with massive amounts of data:

```python
from utils.context_manager import prioritize_context

# Automatically prioritizes sections
context = {
    "financial_data": "...",      # Priority: 1.0
    "company_overview": "...",    # Priority: 0.9
    "recent_news": "...",         # Priority: 0.8
    "sentiment": "...",           # Priority: 0.7
    "competitors": "..."          # Priority: 0.6
}

optimized = prioritize_context(context, max_tokens=30000)
```

**Gemini Flash summarization** applied if still over limit.

## Troubleshooting

### Error: "Missing required environment variables"

**Solution:** Check your `.env` file has all required keys:
```bash
cat .env
```

### Error: "No companies found"

**Possible causes:**
1. Company name misspelled
2. Private company (not in SEC database)
3. Non-US company

**Solution:** Try variations:
- "Apple" → "Apple Inc"
- "Tesla" → "Tesla Inc"
- Use ticker symbol in search

### Error: "Tavily API rate limit"

**Solution:** Fallback to DuckDuckGo automatically activates. Or:
1. Upgrade Tavily plan
2. Reduce `MAX_NEWS_RESULTS` in settings

### Error: "Gemini API quota exceeded"

**Solution:**
1. Check your quota: https://aistudio.google.com/
2. Reduce `MAX_SENTIMENT_SAMPLES`
3. Wait for quota reset (usually 24 hours)

### Slow Performance

**Optimizations:**
1. Reduce `MAX_NEWS_RESULTS` (default: 20)
2. Reduce `MAX_SENTIMENT_SAMPLES` (default: 50)
3. Use smaller context windows
4. Enable caching for repeated searches

### Missing Financial Data

**Possible causes:**
1. Company hasn't filed recent 10-K/10-Q
2. Data not available in companyfacts.json
3. Different fiscal year timing

**Solution:** Check SEC EDGAR manually for recent filings.

## Best Practices

### 1. Company Name Input
- Use official company names
- Include "Inc", "Corp", "Ltd" suffixes
- For subsidiaries, use parent company

### 2. API Key Management
- Never commit `.env` to git
- Use separate keys for dev/prod
- Rotate keys periodically

### 3. Rate Limiting
- Space out multiple searches (1-2 min between)
- Use checkpointing to resume
- Monitor API usage dashboards

### 4. Report Storage
- Archive important reports
- Clean old reports periodically
- Use JSON for data processing

### 5. Error Handling
- Check `state.errors` list after completion
- Review logs for warnings
- Test with known companies first

## Example Workflows

### Research Multiple Companies

```python
companies = ["Apple Inc", "Microsoft", "Google", "Amazon"]

for company in companies:
    print(f"\nResearching {company}...")
    state = await run_research_pipeline(company, checkpointer)
    print(f"✓ Complete: {state.report_path}")
```

### Compare Competitors

```python
# Research main company
main_state = await run_research_pipeline("Tesla")

# Research competitors
competitors = main_state.competitors[:3]
competitor_states = []

for comp in competitors:
    comp_state = await run_research_pipeline(comp.name)
    competitor_states.append(comp_state)

# Compare financials, sentiment, etc.
```

### Monitor Company Over Time

```python
import schedule
import time

def daily_research():
    state = await run_research_pipeline("Tesla")
    # Save to time-series database
    # Alert on sentiment changes
    
schedule.every().day.at("09:00").do(daily_research)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Getting Help

- **Documentation:** `docs/` directory
- **Issues:** Check error messages in console
- **Logs:** Review `logs/` directory (if configured)
- **Community:** [Create an issue](https://github.com/your-repo/issues)

## Next Steps

- Explore `docs/DeepResearchAgent.md` for architecture details
- Review `docs/FinanceRetrieval.md` for SEC data specifics
- Customize prompts in `config/prompts.py`
- Add custom nodes to extend functionality
