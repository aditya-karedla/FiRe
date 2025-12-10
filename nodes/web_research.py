"""
Web research nodes using Tavily (primary) and DuckDuckGo (fallback).
Handles company profile, news, and general web research.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional

import aiohttp
import html2text
from bs4 import BeautifulSoup
from dateutil import parser
from duckduckgo_search import DDGS
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from tavily import TavilyClient

from agents.state import CompanyProfile, ResearchState, SearchResult
from config.settings import settings
from utils.fallback import FallbackChain
from utils.retry import retry

logger = logging.getLogger(__name__)


# Pydantic models for structured output
class ManagementMember(BaseModel):
    """Executive team member information"""
    name: str = Field(description="Executive's full name")
    title: str = Field(description="Job title/position")
    background: Optional[str] = Field(default=None, description="Brief background or notable achievements")


class CompanyProfileExtraction(BaseModel):
    """Structured company profile data extraction"""
    industry: Optional[str] = Field(default=None, description="Primary industry sector")
    sector: Optional[str] = Field(default=None, description="Broader sector classification")
    founded: Optional[str] = Field(default=None, description="Year founded as string")
    headquarters: Optional[str] = Field(default=None, description="Headquarters location")
    employees: Optional[int] = Field(default=None, description="Number of employees")
    key_products: List[str] = Field(default_factory=list, description="List of major products or services")
    geographic_presence: List[str] = Field(default_factory=list, description="Major markets/regions of operation")
    management_team: List[ManagementMember] = Field(default_factory=list, description="Key executives")


class CompetitorInfo(BaseModel):
    """Individual competitor information"""
    name: str = Field(description="Official company name")
    description: str = Field(description="What they do and competitive relationship")


class CompetitorList(BaseModel):
    """List of competitors"""
    competitors: List[CompetitorInfo] = Field(description="List of competitor companies")


# Initialize clients
tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY) if settings.TAVILY_API_KEY else None
ddgs_client = DDGS()
html_converter = html2text.HTML2Text()
html_converter.ignore_links = False
html_converter.ignore_images = True


def search_with_fallback(query: str, max_results: int = 10) -> List[SearchResult]:
    """
    Search using Tavily first, fallback to DuckDuckGo.
    Returns normalized SearchResult objects.
    """
    chain = FallbackChain(f"search: {query}")
    
    # Strategy 1: Tavily (if available)
    if tavily_client:
        def tavily_search():
            logger.info(f"Searching Tavily: {query}")
            response = tavily_client.search(
                query=query,
                search_depth=settings.TAVILY_SEARCH_DEPTH,
                max_results=max_results
            )
            
            results = []
            for item in response.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    domain=item.get("domain"),
                    content=item.get("raw_content"),
                    source="tavily"
                ))
            
            logger.info(f"Tavily returned {len(results)} results")
            return results
        
        chain.add_strategy("Tavily", tavily_search)
    
    # Strategy 2: DuckDuckGo (always available)
    def ddgs_search():
        logger.info(f"Searching DuckDuckGo: {query}")
        results = []
        
        with ddgs_client as ddgs:
            search_results = ddgs.text(query, max_results=max_results)
            
            for item in search_results:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", ""),
                    snippet=item.get("body", ""),
                    source="ddgs"
                ))
        
        logger.info(f"DuckDuckGo returned {len(results)} results")
        return results
    
    chain.add_strategy("DuckDuckGo", ddgs_search)
    
    # Execute with fallback
    return chain.execute() or []


@retry(max_attempts=2)
async def llm_clean_content(raw_text: str, company_name: str) -> str:
    """
    Use LLM to extract only relevant company information from scraped content.
    Filters out navigation, menus, language selectors, and other irrelevant content.
    """
    # Truncate if too long
    text_to_clean = raw_text[:10000] if len(raw_text) > 10000 else raw_text
    
    llm = ChatGoogleGenerativeAI(
        model=settings.SECONDARY_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.0
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a content extraction expert. Extract ONLY the main business description, removing all navigation and UI elements. Write ONLY in English."),
        ("human", """You are reading a scraped webpage about {company}. The text contains both relevant company information AND irrelevant website navigation elements.

Your task: Write a clean, coherent business description IN ENGLISH ONLY using ONLY the relevant company information.

EXAMPLE OF WHAT TO REMOVE:
- "Jump to content"
- "Search"
- "Contents"
- "Beginning 1 Drugs 2 References"
- Language names (العربية, Deutsch, English, Español, etc.)
- "Change links"
- "Page Talk"
- "Edit"
- Navigation headers
- Any lists of languages
- Any non-English text or characters

WHAT TO INCLUDE:
Write 2-4 paragraphs IN ENGLISH describing:
1. What the company does (business description)
2. Key products, services, or business areas
3. Important facts like founding year, headquarters, industry
4. Major achievements or market position

Write as if you're creating a Wikipedia-style introduction. Be concise and factual. USE ONLY ENGLISH LANGUAGE.

---
SCRAPED TEXT:
{text}
---

Write the clean business description below IN ENGLISH (2-4 paragraphs, no navigation elements, no non-English text):
""")
    ])
    
    chain = prompt | llm
    
    try:
        result = await chain.ainvoke({
            "company": company_name,
            "text": text_to_clean
        })
        cleaned_text = result.content.strip()
        
        # Validate the result - if it still contains navigation artifacts, it failed
        nav_indicators = ['Jump to content', 'Search', 'Contents', 'Beginning', 'Change links', 'Page Talk']
        if any(indicator in cleaned_text for indicator in nav_indicators):
            logger.warning("LLM output still contains navigation elements, retrying with stricter prompt")
            raise ValueError("Navigation elements detected in output")
        
        logger.info(f"✓ LLM cleaned content: {len(raw_text)} -> {len(cleaned_text)} chars")
        return cleaned_text
    except Exception as e:
        logger.warning(f"LLM cleaning failed: {e}, attempting fallback extraction")
        # More aggressive fallback - extract only paragraphs that look like content
        paragraphs = []
        for para in raw_text.split('\n\n'):
            para = para.strip()
            # Skip if it looks like navigation (too short, contains specific patterns)
            if len(para) < 50:
                continue
            if any(word in para for word in ['Jump to', 'Search', 'Contents', 'Beginning', 'Edit', 'Page Talk', 'Change links']):
                continue
            # Skip if it's mostly non-English characters (language selector)
            if re.search(r'[\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF]{20,}', para):
                continue
            paragraphs.append(para)
            if len(paragraphs) >= 3:  # Take first 3 good paragraphs
                break
        
        return '\n\n'.join(paragraphs) if paragraphs else "No relevant company information could be extracted."


async def fetch_and_clean_url(url: str, company_name: str = None) -> str:
    """
    Fetch URL content, convert HTML to markdown, and use LLM to clean it.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url,
            headers={"User-Agent": settings.SEC_USER_AGENT},
            timeout=aiohttp.ClientTimeout(total=15)
        ) as response:
            response.raise_for_status()
            html = await response.text()
    
    # Parse and clean HTML
    soup = BeautifulSoup(html, 'lxml')
    
    # Remove unwanted elements more aggressively
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button", "iframe"]):
        element.decompose()
    
    # Remove elements with navigation-related classes/ids
    for selector in [
        '[class*="nav"]', '[class*="menu"]', '[class*="sidebar"]',
        '[class*="toc"]', '[class*="language"]', '[id*="nav"]',
        '[id*="menu"]', '[id*="sidebar"]', '[class*="footer"]',
        '[class*="header"]', '[role="navigation"]'
    ]:
        for element in soup.select(selector):
            element.decompose()
    
    # For Wikipedia, extract main content only
    if 'wikipedia.org' in url:
        # Try to get the first few paragraphs from the main content
        main_content = soup.find('div', {'id': 'mw-content-text'})
        if main_content:
            # Get just the paragraphs, skip infoboxes and other elements
            paragraphs = main_content.find_all('p', recursive=False)[:5]
            if paragraphs:
                # Create new soup with just these paragraphs
                clean_soup = BeautifulSoup('<div></div>', 'lxml')
                for p in paragraphs:
                    clean_soup.div.append(p)
                soup = clean_soup
    
    # Convert to markdown
    markdown = html_converter.handle(str(soup))
    
    # Basic cleanup
    lines = [line.strip() for line in markdown.split('\n') if line.strip()]
    raw_text = '\n'.join(lines)
    
    # Use LLM to clean if company name provided
    if company_name:
        return await llm_clean_content(raw_text, company_name)
    
    return raw_text


async def fetch_company_profile(state: ResearchState) -> ResearchState:
    """
    Fetch company profile from web sources.
    Prioritizes Wikipedia and official company pages.
    """
    logger.info("Fetching company profile...")
    state.current_node = "web_profile"
    
    if not state.found:
        logger.warning("Company not resolved, skipping profile fetch")
        return state
    
    company = state.found.title
    ticker = state.found.ticker
    
    # Search for company overview - prioritize official company sites
    # Try official company site first, then Wikipedia as fallback
    query = f"{company} about us company overview"
    results = search_with_fallback(query, max_results=10)
    
    profile_text = ""
    profile_url = None
    
    # Prioritize official company domains (.com, .co, investor relations, about pages)
    company_keywords = company.lower().replace(' ', '').replace('.', '').replace(',', '')
    
    for result in results:
        url_lower = result.url.lower()
        # Skip non-English Wikipedia pages
        if 'wikipedia.org' in url_lower and not url_lower.startswith('https://en.wikipedia'):
            continue
        # Prioritize official company sites (about pages, investor relations)
        if any(pattern in url_lower for pattern in ['/about', '/company', '/who-we-are', 'investor', '/ir/']):
            profile_url = result.url
            logger.info(f"Selected official company page: {profile_url}")
            break
        # Check if URL contains company name (likely official site)
        if company_keywords[:10] in url_lower.replace('-', '').replace('_', ''):
            profile_url = result.url
            logger.info(f"Selected company domain: {profile_url}")
            break
    
    # Fallback to English Wikipedia
    if not profile_url:
        for result in results:
            if "en.wikipedia.org" in result.url:
                profile_url = result.url
                logger.info(f"Fallback to Wikipedia: {profile_url}")
                break
    
    # Final fallback to first result
    if not profile_url and results:
        profile_url = results[0].url
        logger.info(f"Using first result: {profile_url}")
    
    # Fetch content from selected URL
    if profile_url:
        try:
            profile_text = await fetch_and_clean_url(profile_url, company_name=company)
            logger.info(f"✓ Fetched and cleaned profile from {profile_url}")
        except Exception as e:
            logger.warning(f"Failed to fetch profile content: {e}")
            # Use snippet as fallback
            profile_text = results[0].snippet if results else ""
    
    # Search for investor relations materials
    investor_query = f"{company} investor relations annual report 10-K"
    investor_results = search_with_fallback(investor_query, max_results=5)
    
    investor_docs = []
    for result in investor_results:
        url_lower = result.url.lower()
        if any(term in url_lower for term in ['.pdf', 'investor', 'ir', '10-k', 'annual-report']):
            investor_docs.append({
                "title": result.title,
                "url": result.url
            })
    
    # Extract structured data using LLM
    industry = None
    founded = None
    headquarters = None
    employees = None
    
    if profile_text:
        try:
            # Initialize parser and LLM chain
            parser = JsonOutputParser(pydantic_object=CompanyProfileExtraction)
            
            llm = ChatGoogleGenerativeAI(
                model=settings.SECONDARY_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=settings.GEMINI_TEMPERATURE_EXTRACTION
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data extraction specialist. Extract company information and respond with valid JSON only. Use English language only."),
                ("human", """Extract comprehensive company profile information about {company} from the text below.

IMPORTANT:
- Extract information in ENGLISH ONLY
- For key_products: Extract 3-5 major products or service lines (in English)
- For geographic_presence: List major markets/regions (e.g., ["United States", "Europe", "Asia-Pacific"])
- For management_team: Extract CEO, CFO, and other key C-suite executives (up to 5)
- If any field cannot be found, use null for strings/ints or empty array [] for lists
- All text fields must be in English

{format_instructions}

Text:
{text}""")
            ])
            
            chain = prompt | llm | parser
            
            extracted = await chain.ainvoke({
                "company": company,
                "text": profile_text[:4000],
                "format_instructions": parser.get_format_instructions()
            })
            
            industry = extracted.get("industry")
            founded = str(extracted.get("founded")) if extracted.get("founded") else None
            headquarters = extracted.get("headquarters")
            employees = extracted.get("employees")
            sector = extracted.get("sector")
            key_products = extracted.get("key_products", [])
            geographic_presence = extracted.get("geographic_presence", [])
            management_team = extracted.get("management_team", [])
            
            logger.info(f"✓ Extracted structured data: industry={industry}, sector={sector}, founded={founded}, hq={headquarters}, employees={employees}, products={len(key_products)}, mgmt={len(management_team)}")
            
        except Exception as e:
            logger.warning(f"Failed to extract structured data with LLM: {e}")
            
            # Fallback to basic pattern matching
            # Try to extract year founded
            founded_patterns = [
                r'founded\s+in\s+(\d{4})',
                r'established\s+in\s+(\d{4})',
                r'incorporated\s+in\s+(\d{4})',
            ]
            for pattern in founded_patterns:
                match = re.search(pattern, profile_text, re.IGNORECASE)
                if match:
                    founded = match.group(1)
                    break
            
            # Try to extract headquarters
            hq_patterns = [
                r'headquartered in ([^.\n]+)',
                r'headquarters[:\s]+([^.\n]+)',
            ]
            for pattern in hq_patterns:
                match = re.search(pattern, profile_text, re.IGNORECASE)
                if match:
                    headquarters = match.group(1).strip()
                    break
    
    state.company_profile = CompanyProfile(
        description=profile_text[:5000],  # Limit size
        profile_url=profile_url,
        investor_docs=investor_docs[:10],
        industry=industry,
        founded=founded,
        headquarters=headquarters,
        employees=employees,
        sector=sector if 'sector' in locals() else None,
        key_products=key_products if 'key_products' in locals() else [],
        geographic_presence=geographic_presence if 'geographic_presence' in locals() else [],
        management_team=management_team if 'management_team' in locals() else []
    )
    
    logger.info(f"✓ Profile complete ({len(profile_text)} chars, {len(investor_docs)} docs, {len(key_products) if 'key_products' in locals() else 0} products, {len(management_team) if 'management_team' in locals() else 0} executives)")
    
    return state


async def fetch_news_timeline(state: ResearchState) -> ResearchState:
    """
    Fetch recent news articles about the company (last 1 year).
    """
    logger.info("Fetching news timeline (last 1 year)...")
    state.current_node = "news"
    
    if not state.found:
        logger.warning("Company not resolved, skipping news fetch")
        return state
    
    company = state.found.title
    
    # Search for recent news with time constraint
    cutoff_date = datetime.utcnow() - timedelta(days=365)
    query = f"{company} news after:{cutoff_date.strftime('%Y-%m-%d')}"
    results = search_with_fallback(query, max_results=settings.MAX_NEWS_ITEMS)
    
    # Filter results by published date if available
    filtered_results = []
    for result in results:
        if result.published_date:
            try:
                # Parse published date and check if within 1 year
                pub_date_str = result.published_date
                # Handle various date formats
                pub_date = parser.parse(pub_date_str)
                if pub_date >= cutoff_date:
                    filtered_results.append(result)
            except:
                # If date parsing fails, include it (better to have data than exclude)
                filtered_results.append(result)
        else:
            # If no published date, include it (assume recent)
            filtered_results.append(result)
    
    results = filtered_results if filtered_results else results
    
    logger.info(f"Filtered to {len(results)} news items within 1-year timeframe")
    
    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    
    for result in results:
        if result.url not in seen_urls:
            seen_urls.add(result.url)
            unique_results.append(result)
    
    state.news_timeline = unique_results
    
    logger.info(f"✓ Found {len(unique_results)} news items")
    
    return state


async def identify_competitors(state: ResearchState) -> ResearchState:
    """
    Identify key competitors using web search and LLM extraction.
    """
    logger.info("Identifying competitors...")
    state.current_node = "competitors"
    
    if not state.found:
        logger.warning("Company not resolved, skipping competitor search")
        return state
    
    company = state.found.title
    industry = state.company_profile.industry if state.company_profile and state.company_profile.industry else ""
    
    # Multiple targeted searches for better competitor discovery
    searches = [
        f"{company} top competitors",
        f"{company} vs competitors comparison",
        f"who competes with {company}",
    ]
    
    if industry:
        searches.append(f"{industry} leading companies")
    
    # Gather context from multiple searches
    all_results = []
    for query in searches[:2]:  # Limit to 2 searches to save API calls
        results = search_with_fallback(query, max_results=8)
        all_results.extend(results)
    
    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for result in all_results:
        if result.url not in seen_urls:
            seen_urls.add(result.url)
            unique_results.append(result)
    
    # Build comprehensive context for LLM
    context_text = ""
    for result in unique_results[:15]:
        context_text += f"Source: {result.title}\n"
        if result.snippet:
            context_text += f"Content: {result.snippet}\n"
        context_text += "\n"
    
    # Use LLM to extract structured competitor information
    competitors = []
    
    if context_text:
        try:
            # Initialize parser and LLM chain
            parser = JsonOutputParser(pydantic_object=CompetitorList)
            
            llm = ChatGoogleGenerativeAI(
                model=settings.SECONDARY_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=settings.GEMINI_TEMPERATURE_EXTRACTION
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a competitive intelligence analyst. Extract competitor information and respond with valid JSON only."),
                ("human", """You are analyzing competitors of {company}.

Based on the search results below, identify the TOP 5-8 DIRECT competitors (actual companies, NOT news articles or generic terms).

CRITICAL RULES:
1. Extract ONLY real company names (e.g., "Apple Inc.", "Amazon", "Google/Alphabet")
2. DO NOT include article titles or news headlines
3. DO NOT include generic terms like "Top competitors", "Analysis", etc.
4. Focus on companies in the same industry/market
5. Each competitor should be a distinct company

For each competitor company, provide:
- name: The official company name (not article title)
- description: 1-2 sentences describing what the company does and why they compete with {company}

If you cannot identify clear competitor companies (only see news articles), return an empty competitors array.

{format_instructions}

Search Results:
{context}""")
            ])
            
            chain = prompt | llm | parser
            
            extracted = await chain.ainvoke({
                "company": company,
                "context": context_text[:4500],
                "format_instructions": parser.get_format_instructions()
            })
            # Filter and validate competitors
            excluded_terms = ["unknown", "news", "article", "analysis", "report", "top ", "list of", "alternatives"]
            
            competitor_list = extracted.get("competitors", [])
            for comp in competitor_list[:settings.MAX_COMPETITORS]:
                if isinstance(comp, dict) and comp.get("name"):
                    name = comp.get("name", "").strip()
                    description = comp.get("description", "").strip()
                    
                    # Skip if name looks like an article title
                    name_lower = name.lower()
                    if any(term in name_lower for term in excluded_terms):
                        continue
                    
                    # Skip very short or very long names (likely not company names)
                    if len(name) < 3 or len(name) > 80:
                        continue
                    
                    competitors.append({
                        "name": name,
                        "description": description if description else f"Competitor of {company}",
                        "source": "AI-extracted from market research"
                    })
            
            logger.info(f"✓ Extracted {len(competitors)} competitors using LLM")
            
        except Exception as e:
            logger.warning(f"Failed to extract competitors with LLM: {e}")
            logger.exception(e)
    
    # If no competitors found, add a note
    if not competitors:
        logger.warning(f"Could not identify competitors for {company}")
        competitors = []
    
    state.competitors = competitors
    
    logger.info(f"✓ Found {len(competitors)} competitors")
    
    return state


async def extract_investor_materials(state: ResearchState) -> ResearchState:
    """
    Extract investor materials from SEC submissions.
    Looks for 10-K, 10-Q, 8-K filings within the last year.
    """
    logger.info("Extracting investor materials...")
    state.current_node = "investor_docs"
    
    if not state.submissions:
        logger.warning("No submissions data available")
        return state
    
    filings = state.submissions.get("filings", {}).get("recent", {})
    
    if not filings:
        logger.warning("No recent filings found")
        return state
    
    # Calculate 1-year cutoff date
    cutoff_date = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    logger.info(f"Filtering filings to last 1 year (from {cutoff_date})")
    
    forms = filings.get("form", [])
    accession_nums = filings.get("accessionNumber", [])
    filing_dates = filings.get("filingDate", [])
    primary_docs = filings.get("primaryDocument", [])
    
    # Look for key filing types
    target_forms = ["10-K", "10-Q", "8-K", "DEF 14A"]
    materials = []
    
    for i, form in enumerate(forms[:50]):  # Check more filings to ensure we get 1-year coverage
        if form in target_forms:
            acc_num = accession_nums[i] if i < len(accession_nums) else ""
            date = filing_dates[i] if i < len(filing_dates) else ""
            doc = primary_docs[i] if i < len(primary_docs) else ""
            
            # Skip filings older than 1 year
            if date and date < cutoff_date:
                continue
            
            # Build SEC filing URL
            acc_num_clean = acc_num.replace("-", "")
            url = settings.SEC_FILING_URL_PATTERN.format(
                cik=state.cik10,
                accession=acc_num_clean,
                document=doc
            )
            
            materials.append({
                "title": f"{form} Filing - {date}",
                "url": url,
                "date": date,
                "form_type": form
            })
    
    # Add to company profile
    if state.company_profile:
        state.company_profile.investor_docs.extend(materials[:10])
    
    logger.info(f"✓ Found {len(materials)} investor materials")
    
    return state
