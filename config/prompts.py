"""
Gemini-optimized prompt templates for research synthesis.

These prompts are designed specifically for Gemini's strengths:
- Clear, structured XML/markdown format
- Explicit instructions and constraints
- Role-based framing for better context understanding
- Temperature and tone guidance
"""

from typing import Dict, Any


class PromptTemplates:
    """Collection of optimized prompts for different research tasks"""
    
    @staticmethod
    def sentiment_analysis_prompt(company: str, snippets: list[str]) -> str:
        """
        Gemini-optimized prompt for batch sentiment analysis.
        Uses financial terminology and structured output format.
        """
        snippets_formatted = "\n".join([
            f"{i+1}. {snippet[:500]}"  # Limit snippet length
            for i, snippet in enumerate(snippets)
        ])
        
        return f"""You're a financial analyst reviewing social media sentiment about {company}.

**Task:** Analyze each post below and determine investor sentiment.

**Posts to analyze:**
{snippets_formatted}

**Your analysis should:**
1. Classify sentiment as: BULLISH, BEARISH, NEUTRAL, or MIXED
2. Rate confidence (0.0 to 1.0) based on clarity and context
3. Extract up to 3 key themes per post (single words: "earnings", "product", "competition", etc.)

**Important guidelines:**
- BULLISH = positive outlook, growth expectations, optimism
- BEARISH = negative outlook, concerns, skepticism
- NEUTRAL = factual, balanced, no strong sentiment
- MIXED = contains both positive and negative sentiments
- Be conservative with confidence scores - if unclear, rate lower
- Themes should be lowercase, single words or short phrases

**Output format (JSON array):**
```json
[
  {{"post_num": 1, "sentiment": "BULLISH", "confidence": 0.85, "themes": ["growth", "innovation"]}},
  {{"post_num": 2, "sentiment": "BEARISH", "confidence": 0.70, "themes": ["competition", "margins"]}}
]
```

Return only the JSON array, no additional text."""
    
    @staticmethod
    def research_synthesis_prompt(
        company_name: str,
        ticker: str,
        financials: Dict[str, Any],
        news_items: list[Dict[str, Any]],
        sentiment_data: Dict[str, Any],
        competitors: list[str],
        profile_description: str,
        industry: str = "N/A",
        sector: str = "N/A",
        founded: str = "N/A",
        headquarters: str = "N/A",
        employees: int = None,
        key_products: list[str] = None,
        geographic_presence: list[str] = None,
        management_team: list[Dict[str, str]] = None
    ) -> str:
        """
        Main synthesis prompt for generating investment research report.
        Optimized for Gemini's long-context and analytical capabilities.
        """
        
        # Format financial data
        fin_summary = PromptTemplates._format_financials(financials)
        
        # Format news timeline
        news_summary = PromptTemplates._format_news(news_items[:10])
        
        # Format sentiment
        sentiment_summary = PromptTemplates._format_sentiment(sentiment_data)
        
        # Format competitors
        comp_list = ", ".join(competitors[:5]) if competitors else "Not available"
        
        # Format additional profile fields
        key_products = key_products or []
        geographic_presence = geographic_presence or []
        management_team = management_team or []
        
        products_str = ", ".join(key_products[:5]) if key_products else "Not available"
        geo_str = ", ".join(geographic_presence[:5]) if geographic_presence else "Not available"
        
        # Format management overview
        mgmt_str = ""
        if management_team:
            for exec in management_team[:5]:
                name = exec.get("name", "Unknown")
                title = exec.get("title", "")
                background = exec.get("background", "")
                mgmt_str += f"\n  - **{name}** ({title}): {background}"
        else:
            mgmt_str = "\n  Not available"
        
        return f"""You're a senior equity research analyst at a top-tier investment bank. Write a comprehensive research report on {company_name} (Ticker: {ticker}).

**IMPORTANT:** All data provided below covers the LAST 1 YEAR timeframe (from December 2024 to December 2025). Focus your analysis on this recent period.

## Context & Data

### Company Overview
**Business Description:**
{profile_description[:1000]}

**Key Facts:**
- **Industry/Sector:** {industry} / {sector}
- **Founded:** {founded}
- **Headquarters:** {headquarters}
- **Employees:** {employees if employees else "Not available"}

**Key Products/Services:**
{products_str}

**Geographic Presence:**
{geo_str}

**Management Overview:**{mgmt_str}

### Financial Highlights (Most Recent Period)
{fin_summary}

### Recent News & Developments
{news_summary}

### Market Sentiment Analysis
{sentiment_summary}

### Key Competitors
{comp_list}

---

## Your Assignment

Write an **investment-grade research report** with the following structure:

### 1. Executive Summary (3-4 paragraphs)
Surface the headline narrative by explicitly weaving together the company's business model, key products/services, geographic footprint, management capabilities, financials, news catalysts, sentiment shifts, and competitive dynamics. Highlight how these signals reinforce or contradict each other so an investor immediately sees the integrated picture.

### 2. Integrated Insight Threads
Develop 2-3 cohesive storylines connecting multiple data sources (e.g., link a margin trend to recent product news and investor sentiment). For each thread, explain the causal chain, why it matters now, and what to monitor next.

### 3. Financial Health Assessment
Provide a high-level interpretation and judgment of the financial metrics. Focus on:
- **Revenue Trends:** Direction and quality (not exact precision)
- **Profitability Direction:** Improving, stable, or declining margins
- **Balance Sheet Strength:** Strong, adequate, or concerning leverage
- **Cash Generation:** Ability to fund operations and growth
- **Major Financial Highlights or Red Flags:** Key insights that matter for investment decision

Note: Exact numeric accuracy is less important than sound interpretation and judgment.

### 4. Recent Developments (5 most material events)
From the news timeline (covering the last 1 year), identify and discuss the 5 most significant events that impact the investment thesis.

### 5. Market Perception & Sentiment
Based on social media analysis:
- Overall investor sentiment (bullish/bearish)
- Key themes and concerns in investor discussions
- Alignment (or divergence) between sentiment and fundamentals

### 6. Competitive Positioning & Business Strengths
- How is the company positioned versus competitors? 
- What are its key products/services that drive competitive advantage?
- How does its geographic presence strengthen or limit its position?
- Market share trends, differentiation factors, and competitive moats
- Management team's capability to execute strategy

### 7. Investment Considerations

**Opportunities (3 key points):**
- What could drive upside?

**Risks (3 key points):**
- What could cause problems?

### 8. Analyst Notes
2-3 actionable insights or things to watch going forward.

---

## Critical Guidelines

**DO:**
- Be direct and factual - cite specific numbers from the data provided
- Use professional but readable language (write for an informed investor, not academics)
- Flag missing or incomplete data explicitly (e.g., "Cash flow data not available")
- Make explicit cross-source connections (quant + narrative) and call out reinforcing or conflicting signals
- Use markdown formatting for clarity
- Take the time to reason through complex relationships; depth is preferred over brevity so long as you stay within 2500 words

**DON'T:**
- Invent or hallucinate information not in the provided data
- Give investment recommendations (buy/sell/hold)
- Make predictions about future stock price
- Use marketing language or hype
- Add headers with placeholder names like "[Your Name]", "Analyst:", "Prepared by:", etc.
- Include contact information, role titles, or attribution headers
- Exceed 2500 words total

**Tone:** Professional, analytical, balanced. Think Bloomberg or Goldman Sachs research.

**Format:** Start directly with "## Executive Summary" - no title page, no analyst attribution, no preamble. The report title and metadata will be added separately.

Begin your report below:
"""
    
    @staticmethod
    def _format_financials(financials: Dict[str, Any]) -> str:
        """Format financial data for prompt"""
        if not financials:
            return "Financial data not available."
        
        parts = []
        
        # Income statement
        if "income_statement" in financials:
            inc = financials["income_statement"]
            parts.append("**Income Statement:**")
            for key, metric in inc.items():
                if metric:
                    value = f"${metric.get('value', 0):,.0f}" if metric.get('value') else "N/A"
                    date = metric.get('date', 'Unknown date')
                    parts.append(f"  - {key.replace('_', ' ').title()}: {value} ({date})")
        
        # Balance sheet
        if "balance_sheet" in financials:
            bal = financials["balance_sheet"]
            parts.append("\n**Balance Sheet:**")
            for key, metric in bal.items():
                if metric:
                    value = f"${metric.get('value', 0):,.0f}" if metric.get('value') else "N/A"
                    date = metric.get('date', 'Unknown date')
                    parts.append(f"  - {key.replace('_', ' ').title()}: {value} ({date})")
        
        # Cash flow
        if "cashflow" in financials:
            cf = financials["cashflow"]
            parts.append("\n**Cash Flow:**")
            for key, metric in cf.items():
                if metric:
                    value = f"${metric.get('value', 0):,.0f}" if metric.get('value') else "N/A"
                    date = metric.get('date', 'Unknown date')
                    parts.append(f"  - {key.replace('_', ' ').title()}: {value} ({date})")
        
        return "\n".join(parts) if parts else "Financial metrics not available."
    
    @staticmethod
    def _format_news(news_items: list[Dict[str, Any]]) -> str:
        """Format news timeline for prompt"""
        if not news_items:
            return "No recent news available."
        
        formatted = []
        for i, item in enumerate(news_items[:10], 1):
            title = item.get("title", "Untitled")
            date = item.get("published_date") or item.get("date", "Recent")
            snippet = item.get("snippet", "")[:150]
            formatted.append(f"{i}. [{date}] {title}\n   {snippet}...")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _format_sentiment(sentiment_data: Dict[str, Any]) -> str:
        """Format sentiment analysis for prompt"""
        if not sentiment_data:
            return "Sentiment data not available."
        
        agg = sentiment_data.get("aggregate", {})
        total = agg.get("total_analyzed", 0)
        
        if total == 0:
            return "No sentiment data analyzed."
        
        bullish = agg.get("bullish", 0)
        bearish = agg.get("bearish", 0)
        neutral = agg.get("neutral", 0)
        
        bullish_pct = (bullish / total * 100) if total > 0 else 0
        bearish_pct = (bearish / total * 100) if total > 0 else 0
        neutral_pct = (neutral / total * 100) if total > 0 else 0
        
        net = bullish_pct - bearish_pct
        
        parts = [
            f"**Overall Sentiment:** {'Bullish' if net > 10 else 'Bearish' if net < -10 else 'Neutral'} (Net: {net:+.1f}%)",
            f"  - Bullish: {bullish_pct:.1f}% ({bullish} posts)",
            f"  - Bearish: {bearish_pct:.1f}% ({bearish} posts)",
            f"  - Neutral: {neutral_pct:.1f}% ({neutral} posts)",
        ]
        
        # Add top themes if available
        themes = sentiment_data.get("top_themes", [])
        if themes:
            theme_str = ", ".join([f"{theme[0]} ({theme[1]})" for theme in themes[:5]])
            parts.append(f"\n**Top Discussion Themes:** {theme_str}")
        
        return "\n".join(parts)
    
    @staticmethod
    def summarization_prompt(text: str, max_words: int = 500) -> str:
        """Quick summarization prompt for long content"""
        return f"""Summarize the following content in {max_words} words or less. Focus on the most important facts and insights.

Content:
{text}

Summary (max {max_words} words):"""


# Create global instance for easy import
prompts = PromptTemplates()
