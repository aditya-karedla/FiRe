# FiRe - Deep Financial Research Agent 🔬

An intelligent AI-powered research agent that performs comprehensive financial analysis by combining SEC filings, web research, news analysis, and sentiment tracking. Built with LangGraph for orchestration and Streamlit for an interactive UI.

## Overview

FiRe (Financial Research) is an autonomous research agent that generates detailed financial reports on publicly traded companies. It leverages multiple data sources including:

- **SEC EDGAR** - Official financial statements and company facts
- **Tavily API** - Advanced web research and content extraction
- **News Sources** - Real-time news timeline and event tracking
- **Social Sentiment** - Market sentiment analysis from various sources
- **Competitor Analysis** - Market context and competitive landscape

The agent uses **Human-in-the-Loop (HITL)** for company selection, ensuring accurate targeting before executing the research pipeline.

## Key Features

✅ **Intelligent Company Resolution** - Fuzzy matching with LLM-powered validation  
✅ **Multi-Source Data Collection** - SEC filings, web research, news, and sentiment  
✅ **Parallel Execution** - Optimized workflow with concurrent node processing  
✅ **Persistent State** - SQLite checkpointing for reliability  
✅ **Interactive UI** - Real-time progress tracking and report generation  
✅ **PDF Export** - Download comprehensive reports in PDF format  

## Architecture

The agent is built using **LangGraph** for workflow orchestration:

```
┌─────────────────┐
│ Company Input   │
└────────┬────────┘
         │
    ┌────▼────────────────┐
    │ Company Resolution  │ (HITL)
    │  • Fuzzy Matching   │
    │  • LLM Validation   │
    │  • User Selection   │
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │ Parallel Research   │
    │  ├─ SEC Data        │
    │  ├─ Web Research    │
    │  ├─ News Timeline   │
    │  └─ Sentiment       │
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │ Synthesis & Report  │
    └─────────────────────┘
```

## Prerequisites

- **Python 3.9+**
- **API Keys**:
  - Google Gemini API (for LLM processing)
  - Tavily API (for web research)
- **Valid Email** (for SEC API user agent)

## Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AdityaK1302/FiRe.git
cd FiRe
```

### 2. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Guide you through environment configuration

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
# SEC EDGAR API (Required)
SEC_USER_AGENT=YourName your@email.com

# Google Gemini API (Required)
GOOGLE_API_KEY=your_gemini_api_key

# Tavily API (Required)
TAVILY_API_KEY=your_tavily_api_key

# Model Configuration (Optional)
PRIMARY_MODEL=gemini-2.5-pro
SECONDARY_MODEL=gemini-2.5-flash
```

**Getting API Keys:**
- **Google Gemini**: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
- **Tavily**: [https://tavily.com/](https://tavily.com/)

### 4. Launch the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Usage

### Step 1: Enter Company Name
Type the company name you want to research (e.g., "Apple Inc", "Tesla", "Microsoft")

### Step 2: Select Company (HITL)
If multiple matches are found, select the correct company from the suggestions

### Step 3: Automatic Research
The agent will:
- Fetch SEC financial data
- Perform web research
- Collect news and events
- Analyze sentiment
- Identify competitors

### Step 4: Review Report
View the comprehensive research report with:
- Executive summary
- Financial highlights
- Market context
- Key findings
- Risk factors

### Step 5: Download PDF
Export the full report as a PDF document

## Project Structure

```
FiRe/
├── app.py                      # Streamlit UI application
├── setup.sh                    # Automated setup script
├── requirements.txt            # Python dependencies
├── agents/                     # LangGraph orchestration
│   ├── graph_builder.py        # Workflow definition
│   └── state.py                # State management
├── nodes/                      # Research nodes
│   ├── company_resolution.py   # Company matching
│   ├── sec_data.py             # SEC EDGAR integration
│   ├── web_research.py         # Tavily web search
│   ├── sentiment_analysis.py   # Sentiment tracking
│   └── report_generation.py    # Report synthesis
├── config/                     # Configuration
│   ├── settings.py             # Environment settings
│   └── prompts.py              # LLM prompts
├── utils/                      # Helper utilities
│   ├── cache.py                # Response caching
│   ├── retry.py                # Retry logic
│   └── pdf_utils.py            # PDF generation
└── docs/                       # Documentation
```

## Technologies Used

- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Agent workflow orchestration
- **[LangChain](https://www.langchain.com/)** - LLM integration framework
- **[Google Gemini](https://deepmind.google/technologies/gemini/)** - LLM for reasoning and synthesis
- **[Tavily](https://tavily.com/)** - AI-optimized search API
- **[Streamlit](https://streamlit.io/)** - Interactive web interface
- **[SEC EDGAR](https://www.sec.gov/edgar)** - Official financial data
- **[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)** - HTML parsing
- **[Pydantic](https://docs.pydantic.dev/)** - Data validation

## Advanced Configuration

### Model Settings

Customize LLM behavior in `.env`:

```env
# Temperature settings (0.0-1.0)
GEMINI_TEMPERATURE_VALIDATION=0.1    # Company validation
GEMINI_TEMPERATURE_EXTRACTION=0.2    # Data extraction
GEMINI_TEMPERATURE_SYNTHESIS=0.4     # Report generation

# Token limits
GEMINI_MAX_OUTPUT_TOKENS_VALIDATION=3000
GEMINI_MAX_OUTPUT_TOKENS_SYNTHESIS=8000
```

### Research Depth

Adjust research parameters in `config/settings.py`:

```python
MAX_SEARCH_RESULTS = 10        # Web search results per query
NEWS_LOOKBACK_DAYS = 30        # News timeline window
SENTIMENT_SOURCES = 5          # Sentiment data sources
```

## Troubleshooting

### Common Issues

**"SEC API Error"**
- Ensure `SEC_USER_AGENT` includes your valid email address
- Check internet connectivity

**"Google API Error"**
- Verify `GOOGLE_API_KEY` is valid
- Check API quota limits

**"Tavily Rate Limit"**
- Free tier has request limits
- Consider upgrading Tavily plan

**"No Company Found"**
- Try using the full legal name
- Check if the company is publicly traded (has a ticker)

### Enable Debug Logging

Set environment variable:
```bash
export LOG_LEVEL=DEBUG
streamlit run app.py
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SEC EDGAR for providing free access to financial data
- Google for the Gemini API
- Tavily for AI-optimized search capabilities
- LangGraph team for the excellent orchestration framework

## Contact

**Aditya Karedla**  
GitHub: [@AdityaK1302](https://github.com/AdityaK1302)

---

**⭐ Star this repo if you find it useful!**
