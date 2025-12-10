"""
Configuration settings for the Deep Research Agent.
Loads environment variables and provides centralized config access.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


class Settings:
    """Central configuration management"""
    
    # SEC API Configuration
    SEC_USER_AGENT: str = os.getenv("SEC_USER_AGENT", "")
    SEC_COMPANY_TICKERS_URL: str = "https://www.sec.gov/files/company_tickers.json"
    SEC_COMPANY_FACTS_URL: str = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    SEC_SUBMISSIONS_URL: str = "https://data.sec.gov/submissions/CIK{cik}.json"
    SEC_FILING_URL_PATTERN: str = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
    SEC_RATE_LIMIT: int = 10  # requests per second
    
    # Google Gemini Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "gemini-2.5-pro")
    SECONDARY_MODEL: str = os.getenv("SECONDARY_MODEL", "gemini-2.5-flash")
    
    # LLM Temperature settings for different tasks
    GEMINI_TEMPERATURE_VALIDATION: float = 0.1  # Low for consistent validation
    GEMINI_TEMPERATURE_EXTRACTION: float = 0.2  # Low for data extraction
    GEMINI_TEMPERATURE_SUGGESTIONS: float = 0.3  # Some creativity for alternatives
    GEMINI_TEMPERATURE_SYNTHESIS: float = 0.4  # Balanced for report generation
    
    # LLM Token limits for different tasks
    GEMINI_MAX_OUTPUT_TOKENS_VALIDATION: int = 3000
    GEMINI_MAX_OUTPUT_TOKENS_SUGGESTIONS: int = 3000
    GEMINI_MAX_OUTPUT_TOKENS_SENTIMENT: int = 3000
    GEMINI_MAX_OUTPUT_TOKENS_SYNTHESIS: int = 10000
    
    # LLM Advanced parameters
    GEMINI_TOP_P: float = 0.95
    GEMINI_TOP_K: int = 40
    
    # Tavily Configuration
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    TAVILY_MAX_RESULTS: int = 10
    TAVILY_SEARCH_DEPTH: str = "advanced"
    
    # Cache Configuration
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", str(DATA_DIR)))
    TICKERS_CACHE_FILE: Path = CACHE_DIR / "company_tickers.json"
    TICKERS_CACHE_TTL: int = 86400  # 24 hours in seconds
    
    
    # Research Configuration
    MAX_NEWS_ITEMS: int = 50
    MAX_SENTIMENT_SAMPLES: int = 50
    SENTIMENT_SNIPPET_LENGTH: int = 500
    MAX_COMPETITORS: int = 10
    
    # Sentiment Analysis
    MAX_SENTIMENT_SAMPLES: int = 50
    SENTIMENT_SNIPPET_LENGTH: int = 1000
    
    # Context Window Management
    MAX_CONTEXT_TOKENS: int = 800_000  # 80% of 1M token limit
    PROFILE_SUMMARY_THRESHOLD: int = 2000  # chars
    SYNTHESIS_CONTEXT_TOKENS: int = 30_000  # Max tokens for synthesis context (leave room for response)
    MAX_SUMMARIZATION_OUTPUT_MULTIPLIER: int = 2  # max_words * 2 for summarization
    
    # Retry Configuration
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_FACTOR: float = 2.0
    RETRY_MAX_WAIT: int = 60  # seconds
    
    # Streamlit
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    STREAMLIT_ADDRESS: str = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate required settings and return list of missing configs"""
        missing = []
        
        if not cls.SEC_USER_AGENT or "your.email" in cls.SEC_USER_AGENT:
            missing.append("SEC_USER_AGENT must be set with your email")
        
        if not cls.GOOGLE_API_KEY or "your_" in cls.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY must be set")
        
        if not cls.TAVILY_API_KEY or "your_" in cls.TAVILY_API_KEY:
            missing.append("TAVILY_API_KEY must be set")
        
        return missing
    
    @classmethod
    def get_headers(cls) -> dict:
        """Get HTTP headers for SEC requests"""
        return {
            "User-Agent": cls.SEC_USER_AGENT,
            "Accept": "application/json"
        }


# Create global settings instance
settings = Settings()

# Validate on import
validation_errors = settings.validate()
if validation_errors:
    print("⚠️  Configuration warnings:")
    for error in validation_errors:
        print(f"   - {error}")
    print("\nPlease update your .env file with proper credentials.")
