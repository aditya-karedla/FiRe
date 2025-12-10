"""
Pydantic models for state management throughout the research pipeline.
These models ensure type safety and validation at every step.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PipelineStatus(str, Enum):
    """Current status of the research workflow"""
    INITIALIZING = "initializing"
    AWAITING_HITL = "awaiting_hitl"
    FETCHING_DATA = "fetching_data"
    RESEARCH_IN_PROGRESS = "research_in_progress"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


class CompanyMatch(BaseModel):
    """A potential company match from SEC database"""
    title: str
    ticker: str
    cik_str: str
    cik10: Optional[str] = None
    
    @field_validator('cik10', mode='before')
    @classmethod
    def compute_cik10(cls, v, info):
        """Automatically format CIK to 10 digits"""
        if v is None and info.data.get('cik_str'):
            try:
                return f"{int(info.data['cik_str']):010d}"
            except (ValueError, TypeError):
                return None
        return v
    
    def __str__(self) -> str:
        return f"{self.title} ({self.ticker})"


class FinancialMetric(BaseModel):
    """Single financial data point from SEC filings"""
    element: str
    value: Optional[float] = None
    date: Optional[str] = None
    unit: str = "USD"
    form_type: Optional[str] = None
    
    def formatted_value(self) -> str:
        """Human-readable value formatting"""
        if self.value is None:
            return "N/A"
        
        # Format large numbers with B/M suffixes
        abs_val = abs(self.value)
        if abs_val >= 1_000_000_000:
            return f"${self.value/1_000_000_000:.2f}B"
        elif abs_val >= 1_000_000:
            return f"${self.value/1_000_000:.2f}M"
        else:
            return f"${self.value:,.0f}"


class FinancialStatements(BaseModel):
    """Complete set of financial statements"""
    income_statement: Dict[str, Optional[FinancialMetric]] = {}
    balance_sheet: Dict[str, Optional[FinancialMetric]] = {}
    cashflow: Dict[str, Optional[FinancialMetric]] = {}
    extraction_date: datetime = Field(default_factory=datetime.utcnow)
    
    def is_complete(self) -> bool:
        """Check if we have reasonable financial data"""
        has_income = any(self.income_statement.values())
        has_balance = any(self.balance_sheet.values())
        return has_income and has_balance


class SearchResult(BaseModel):
    """Normalized search result from any source"""
    title: str
    url: str
    snippet: Optional[str] = None
    domain: Optional[str] = None
    published_date: Optional[str] = None
    content: Optional[str] = None
    source: Literal["tavily", "ddgs"] = "tavily"
    
    @field_validator('snippet', 'content')
    @classmethod
    def limit_text_length(cls, v):
        """Prevent extremely long text fields"""
        if v and len(v) > 5000:
            return v[:5000] + "..."
        return v


class CompanyProfile(BaseModel):
    """Aggregated company profile information"""
    description: str = ""
    profile_url: Optional[str] = None
    investor_docs: List[Dict[str, str]] = []
    founded: Optional[str] = None
    industry: Optional[str] = None
    employees: Optional[int] = None
    headquarters: Optional[str] = None
    # Additional fields for enhanced profiling
    key_products: List[str] = []  # Key products or services
    geographic_presence: List[str] = []  # Markets/regions where company operates
    management_team: List[Dict[str, str]] = []  # Key executives (name, title, background)
    sector: Optional[str] = None  # Broader sector classification


class SentimentAggregate(BaseModel):
    """Aggregated sentiment scores from LLM analysis"""
    total_analyzed: int = 0
    bullish: int = 0
    bearish: int = 0
    neutral: int = 0
    mixed: int = 0
    confidence_avg: float = 0.0
    top_themes: List[tuple[str, int]] = []
    
    @property
    def bullish_ratio(self) -> float:
        """Percentage of bullish sentiment"""
        if self.total_analyzed == 0:
            return 0.0
        return self.bullish / self.total_analyzed
    
    @property
    def bearish_ratio(self) -> float:
        """Percentage of bearish sentiment"""
        if self.total_analyzed == 0:
            return 0.0
        return self.bearish / self.total_analyzed
    
    @property
    def net_sentiment(self) -> float:
        """Net sentiment score (bullish - bearish)"""
        return self.bullish_ratio - self.bearish_ratio
    
    def summary(self) -> str:
        """Human-readable sentiment summary"""
        net = self.net_sentiment
        if net > 0.15:
            overall = "Strongly Bullish"
        elif net > 0.05:
            overall = "Bullish"
        elif net < -0.15:
            overall = "Strongly Bearish"
        elif net < -0.05:
            overall = "Bearish"
        else:
            overall = "Neutral"
        
        return f"{overall} ({self.bullish}/{self.bearish}/{self.neutral} posts)"


class ResearchState(BaseModel):
    """
    Main state object that flows through the entire LangGraph pipeline.
    This is where all research data accumulates.
    """
    
    # Input from user
    company_name: str = ""
    
    # Pipeline tracking
    status: PipelineStatus = PipelineStatus.INITIALIZING
    current_node: str = "start"
    error_message: Optional[str] = None
    
    # Company resolution phase
    company_tickers: List[Dict[str, Any]] = []
    match_options: List[CompanyMatch] = []
    human_response: Optional[str] = None  # User's HITL selection
    found: Optional[CompanyMatch] = None  # Resolved company
    cik10: Optional[str] = None
    
    # LLM validation and suggestions
    llm_validation_passed: bool = False
    validation_result: Optional[Dict[str, Any]] = None
    llm_suggestions: List[str] = []
    
    # SEC financial data
    companyfacts: Dict[str, Any] = {}
    submissions: Dict[str, Any] = {}
    financials_1yr: Optional[FinancialStatements] = None
    
    # Web research results
    company_profile: Optional[CompanyProfile] = None
    news_timeline: List[SearchResult] = []
    social_sentiment: Optional[Dict[str, Any]] = None
    competitors: List[Dict[str, Any]] = []
    
    # Final synthesis
    synthesis_prompt: Optional[str] = None
    synthesized_insights: Optional[str] = None
    final_report: Optional[str] = None
    report_path: Optional[str] = None
    
    # Metadata & tracking
    execution_start: datetime = Field(default_factory=datetime.utcnow)
    execution_end: Optional[datetime] = None
    retry_counts: Dict[str, int] = {}
    checkpoint_ids: List[str] = []
    
    class Config:
        arbitrary_types_allowed = True
    
    def get_thread_id(self) -> str:
        """Generate normalized thread ID from company name"""
        if self.found:
            base = self.found.ticker.lower()
        elif self.company_name:
            base = self.company_name.lower()
        else:
            base = "unknown"
        
        # Normalize: remove spaces, dots, special chars
        normalized = "".join(c for c in base if c.isalnum() or c == "_")
        return normalized or "research"
    
    def duration(self) -> Optional[float]:
        """Calculate execution duration in seconds"""
        if self.execution_end:
            return (self.execution_end - self.execution_start).total_seconds()
        return None
    
    def progress_summary(self) -> str:
        """Human-readable progress summary"""
        checks = []
        if self.found:
            checks.append("✓ Company resolved")
        if self.financials_1yr:
            checks.append("✓ Financials extracted")
        if self.company_profile:
            checks.append("✓ Profile loaded")
        if self.news_timeline:
            checks.append(f"✓ {len(self.news_timeline)} news items")
        if self.social_sentiment:
            checks.append("✓ Sentiment analyzed")
        if self.competitors:
            checks.append(f"✓ {len(self.competitors)} competitors")
        if self.synthesized_insights:
            checks.append("✓ Report synthesized")
        
        return " | ".join(checks) if checks else "Starting..."


# Specialized models for node outputs

class ValidationReport(BaseModel):
    """Results from data validation"""
    passed: bool
    warnings: List[str] = []
    errors: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class NodeError(BaseModel):
    """Structured error information from failed nodes"""
    node_name: str
    error_type: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = 0
