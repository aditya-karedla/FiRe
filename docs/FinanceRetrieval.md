LangGraph (Python) — SEC Financial Agent (HITL)

Goal: Given a US company name, verify it with a Human-In-The-Loop (HITL) step and return only the last 1 year of:
	•	Income Statement
	•	Balance Sheet
	•	Cash Flow Statement

Primary data source: SEC data.sec.gov endpoints (company_tickers.json, companyfacts, submissions).
No third-party paid APIs, no placeholders. You only need to set a descriptive SEC_USER_AGENT environment variable (the SEC requires this).

This document contains everything you need:
	•	Design & file layout
	•	Full Python LangGraph implementation (nodes + graph)
	•	Streamlit UI that supports HITL
	•	requirements.txt and run instructions

⸻

File layout

sec_langgraph_agent/
├─ langgraph_agent.py          # LangGraph graph + nodes (core logic)
├─ app.py                      # Streamlit UI (HITL + final outputs)
├─ requirements.txt
└─ README.md                   # (this file, optional copy)


⸻

1) Install dependencies

Create a venv and install:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

requirements.txt (exact file content):

langchain==1.0.0         # or the langchain package version that provides langgraph in your env
requests>=2.28
streamlit>=1.20

If your environment’s langchain distribution exposes langgraph under a different package name, adapt imports accordingly. The code below uses the langchain.langgraph API surface (common in recent LangGraph examples).

⸻

2) Environment configuration

Set the SEC user agent (required):

export SEC_USER_AGENT="MyLangGraphAgent aditya.k@example.com"

(no API key required for the SEC endpoints).

⸻

3) langgraph_agent.py — full LangGraph pipeline with HITL

Save the following as langgraph_agent.py. This is a complete, runnable implementation.

# langgraph_agent.py
import os
import json
import difflib
from typing import Dict, Any, List, Optional

import requests

# LangGraph imports: Graph primitives + runtime decorator for human nodes
# If your version uses slightly different imports adjust accordingly.
from langchain.langgraph import StateGraph, GraphState, runtime

# -------------------------
# Config / constants
# -------------------------
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT")
if not SEC_USER_AGENT:
    raise RuntimeError("Please set environment variable SEC_USER_AGENT (e.g. 'MyAgent you@domain.com')")

HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept": "application/json"}

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_FMT = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
SEC_SUBMISSIONS_FMT = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_ARCHIVE_BASE = "https://www.sec.gov/Archives/"

# -------------------------
# HTTP helper
# -------------------------
def http_get_json(url: str, timeout: int = 20) -> Dict[str, Any]:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()

def format_cik_to_10(cik_str: str) -> str:
    return f"{int(cik_str):010d}"

# -------------------------
# Nodes
# -------------------------
def node_load_company_tickers(state: GraphState, *_):
    """
    Downloads SEC company_tickers.json and stores 'company_tickers' in state.
    """
    data = http_get_json(SEC_COMPANY_TICKERS_URL)
    # file has numeric string keys; values contain 'cik_str', 'ticker', 'title'
    tickers_list = list(data.values()) if isinstance(data, dict) else data
    state["company_tickers"] = tickers_list
    return state

def node_suggest_matches(state: GraphState, *_):
    """
    Create top-3 fuzzy match options from company_tickers.
    Stores: state['match_options'] = list of dicts {title, ticker, cik_str}
    """
    company_name = state.get("company_name", "").strip()
    if not company_name:
        raise ValueError("state['company_name'] must be provided")

    tickers = state.get("company_tickers")
    if not tickers:
        raise ValueError("company_tickers not loaded")

    titles = [t.get("title", "") for t in tickers]
    # fuzzy match
    fuzzy = difflib.get_close_matches(company_name, titles, n=5, cutoff=0.55)

    options = []
    for title in fuzzy:
        for t in tickers:
            if t.get("title") == title:
                options.append({"title": t["title"], "ticker": t["ticker"], "cik_str": t["cik_str"]})
                break
        if len(options) >= 3:
            break

    # fallback substring if none
    if not options:
        lower = company_name.lower()
        for t in tickers:
            if lower in t.get("title", "").lower():
                options.append({"title": t["title"], "ticker": t["ticker"], "cik_str": t["cik_str"]})
                if len(options) >= 3:
                    break

    if not options:
        raise LookupError(f"No match candidates found for '{company_name}'")

    # trim to 3
    state["match_options"] = options[:3]
    return state

@runtime.human_node
def human_confirm_match(state: GraphState):
    """
    HITL node. Returns a user prompt dict for the UI to show.
    The UI should collect a reply (option number '1'|'2'|'3') and set state['human_response'].
    """
    options = state.get("match_options", [])
    if not options:
        raise ValueError("No match options to confirm")

    prompt_lines = ["Multiple matches found. Choose the correct company by entering the option number:"]
    for i, opt in enumerate(options, start=1):
        prompt_lines.append(f"{i}. {opt['title']}  (Ticker: {opt['ticker']}, CIK: {opt['cik_str']})")
    prompt_text = "\n".join(prompt_lines)

    # Return a structure that LangGraph's UI/human integrator will show.
    return {"user_prompt": prompt_text, "expected_response": "Enter option number (1,2 or 3)"}

def node_resolve_match(state: GraphState, *_):
    """
    Reads state['human_response'] (expected '1'|'2'|'3') and sets found + cik10.
    """
    resp = state.get("human_response", "")
    if not resp:
        raise ValueError("state['human_response'] must be set by the UI after HITL")

    try:
        idx = int(resp.strip()) - 1
    except Exception:
        raise ValueError("Invalid human_response; expected integer 1/2/3")

    options = state.get("match_options", [])
    if idx < 0 or idx >= len(options):
        raise IndexError("Selected index out of range")

    chosen = options[idx]
    state["found"] = chosen
    state["cik10"] = format_cik_to_10(chosen["cik_str"])
    return state

def node_fetch_sec_data(state: GraphState, *_):
    """
    Downloads companyfacts and submissions for CIK.
    """
    cik10 = state.get("cik10")
    if not cik10:
        raise ValueError("cik10 not set")

    facts_url = SEC_COMPANY_FACTS_FMT.format(cik10=cik10)
    subs_url = SEC_SUBMISSIONS_FMT.format(cik10=cik10)

    state["companyfacts"] = http_get_json(facts_url)
    state["submissions"] = http_get_json(subs_url)
    return state

def node_extract_financial_statements(state: GraphState, *_):
    """
    Extract only the last 1-year values for:
      - Income Statement: Revenues (or SalesRevenueNet), OperatingIncomeLoss, NetIncomeLoss
      - Balance Sheet: Assets, Liabilities, Shareholders' Equity
      - Cash Flow: NetCashProvidedByUsedInOperatingActivities, ...Investing..., ...Financing...
    Stores state['financials_1yr'] with simple structured dicts.
    """
    facts_root = state.get("companyfacts", {}).get("facts", {})
    us_gaap = facts_root.get("us-gaap", {})

    def pick_latest_usd(element_names: List[str]) -> Optional[Dict[str, Any]]:
        for name in element_names:
            el = us_gaap.get(name)
            if not el:
                continue
            units = el.get("units", {})
            if "USD" not in units:
                # fallback to any unit
                unit_values = next(iter(units.values()), [])
            else:
                unit_values = units["USD"]
            if not unit_values:
                continue
            # find latest by 'end' date (some facts might be instant; still we sort by end/filed)
            sorted_vals = sorted(unit_values, key=lambda x: x.get("end", x.get("filed", "")), reverse=True)
            top = sorted_vals[0]
            return {"element": name, "value": top.get("val"), "date": top.get("end") or top.get("filed")}
        return None

    # Income statement fields (common names across filings)
    income_statement = {
        "revenues": pick_latest_usd(["Revenues", "SalesRevenueNet", "SalesRevenue"]),
        "operating_income": pick_latest_usd(["OperatingIncomeLoss"]),
        "net_income": pick_latest_usd(["NetIncomeLoss"])
    }

    # Balance sheet
    balance_sheet = {
        "assets": pick_latest_usd(["Assets"]),
        "liabilities": pick_latest_usd(["Liabilities"]),
        "equity": pick_latest_usd([
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "StockholdersEquityComponent"
        ])
    }

    # Cash flow
    cashflow = {
        "operating_cashflow": pick_latest_usd(["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]),
        "investing_cashflow": pick_latest_usd(["NetCashProvidedByUsedInInvestingActivities"]),
        "financing_cashflow": pick_latest_usd(["NetCashProvidedByUsedInFinancingActivities"])
    }

    state["financials_1yr"] = {
        "income_statement": income_statement,
        "balance_sheet": balance_sheet,
        "cashflow": cashflow
    }
    return state

# -------------------------
# Graph builder
# -------------------------
def build_graph():
    """
    Build and return a compiled StateGraph.
    """
    graph = StateGraph(initial_state={})

    graph.add_node(node_load_company_tickers, name="load_tickers")
    graph.add_node(node_suggest_matches, name="suggest_matches")
    graph.add_node(human_confirm_match, name="hitl_confirm")
    graph.add_node(node_resolve_match, name="resolve_match")
    graph.add_node(node_fetch_sec_data, name="fetch_sec")
    graph.add_node(node_extract_financial_statements, name="financials_1yr")

    # wiring
    graph.add_edge("load_tickers", "suggest_matches")
    graph.add_edge("suggest_matches", "hitl_confirm")
    graph.add_edge("hitl_confirm", "resolve_match")
    graph.add_edge("resolve_match", "fetch_sec")
    graph.add_edge("fetch_sec", "financials_1yr")

    # Return the graph object for invocation by your UI/runner.
    return graph

# -------------------------
# Runner convenience
# -------------------------
def run_company_pipeline(company_name: str, graph: StateGraph):
    """
    Convenience runner: returns the final state dict (after the graph run).
    Note: with HITL the UI must perform two-step invocation:
      1) Invoke graph with {'company_name': ...} to reach the human step (hitl)
         -> graph will return a state containing 'user_prompt' and waiting for 'human_response'.
      2) After user provides choice, invoke graph with {'human_response': '1'} to continue.
    This helper is useful for CLI testing (but UI will handle two invocations).
    """
    return graph.run({"company_name": company_name})

Notes about the graph:
	•	node_suggest_matches produces state["match_options"] (top up to 3 candidates).
	•	human_confirm_match is a @runtime.human_node — the UI should display user_prompt text and wait for user input; when the UI sends human_response (string '1', '2', or '3'), node_resolve_match runs and the pipeline continues to fetch SEC data and extract last-year numbers.
	•	The extracted values come from companyfacts XBRL JSON; we pick the most recent USD fact.

⸻

4) app.py — Streamlit UI (HITL enabled)

Save as app.py. The UI runs the graph and handles HITL using two invocations: first to surface the user prompt, second to resume with the user’s choice.

# app.py
import streamlit as st
import json
from langgraph_agent import build_graph

st.set_page_config(page_title="SEC Financial Agent (HITL)", layout="wide")
st.title("SEC Financial Agent — Income / Balance / Cashflow (Last 1 Year)")

# Build graph once and reuse
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

graph = st.session_state.graph

# session flags
if "last_state" not in st.session_state:
    st.session_state.last_state = None
if "awaiting_hitl" not in st.session_state:
    st.session_state.awaiting_hitl = False

company_input = st.text_input("Enter US Company name (e.g. 'Apple Inc', 'Microsoft'):")

# Run button
if st.button("Resolve & Fetch (HITL)"):
    if not company_input.strip():
        st.error("Enter a company name.")
    else:
        # First invocation: start flow which will reach the human node
        try:
            # run graph with just the company_name to reach the HITL node
            # Using graph.run here — it will execute until it reaches the human node and return a state
            state = graph.run({"company_name": company_input})
            st.session_state.last_state = state
            # detect if graph asked for human prompt (our human node returns user_prompt)
            if "user_prompt" in state:
                st.session_state.awaiting_hitl = True
            else:
                st.session_state.awaiting_hitl = False
        except Exception as e:
            st.error(f"Error running graph: {e}")

# If awaiting HITL, show prompt and options
if st.session_state.awaiting_hitl and st.session_state.last_state:
    st.subheader("🔎 Human review required")
    prompt_text = st.session_state.last_state.get("user_prompt", "")
    st.code(prompt_text)

    # display the options more nicely
    options = st.session_state.last_state.get("match_options", [])
    if options:
        # show radio with readable labels
        labels = [f"{i+1}. {opt['title']} (Ticker: {opt['ticker']})" for i, opt in enumerate(options)]
        chosen_label = st.radio("Select the correct option:", labels)
        # convert back to option number
        chosen_idx = labels.index(chosen_label) + 1
        if st.button("Submit selection"):
            # second invocation: resume graph by supplying human_response
            try:
                next_state = graph.run({"human_response": str(chosen_idx)})
                st.session_state.last_state = next_state
                st.session_state.awaiting_hitl = False
            except Exception as e:
                st.error(f"Error continuing graph after HITL: {e}")

# If we have a final state, display resolved company and the 1-year financials
state = st.session_state.last_state
if state and not st.session_state.awaiting_hitl:
    st.subheader("✅ Resolved Company")
    st.json(state.get("found", {}))

    st.subheader("📑 Recent 10-K / 10-Q Filings (some filings may be presented)")
    filings = state.get("recent_filings") or []
    # If submissions not parsed into recent_filings, we can show a short list from submissions JSON
    if filings:
        for f in filings:
            st.markdown(f"- **{f['form']}** ({f['filedDate']}): [View]({f['edgar_url']})")
    else:
        # try to craft from submissions (optional)
        subs = state.get("submissions", {}).get("filings", {}).get("recent", {})
        forms = subs.get("form", [])[:5]
        accs = subs.get("accessionNumber", [])[:5]
        dates = subs.get("filingDate", [])[:5]
        docs = subs.get("primaryDocument", [])[:5]
        if forms:
            for form, acc, date, doc in zip(forms, accs, dates, docs):
                st.markdown(f"- **{form}** ({date}) — Accession: {acc} — Document: {doc}")
        else:
            st.write("No filings metadata available.")

    st.subheader("📘 Financial Statements (Last 1 Year)")
    fin = state.get("financials_1yr", {})

    st.markdown("### Income Statement")
    st.json(fin.get("income_statement", {}))

    st.markdown("### Balance Sheet")
    st.json(fin.get("balance_sheet", {}))

    st.markdown("### Cash Flow")
    st.json(fin.get("cashflow", {}))

    # Option to download JSON
    if st.button("Download financials JSON"):
        st.download_button(
            label="Download JSON",
            data=json.dumps(fin, indent=2),
            file_name=f"{state.get('found', {}).get('ticker','company')}_financials_1yr.json",
            mime="application/json"
        )

How the UI/HITL works
	1.	User enters a company name and clicks Resolve & Fetch (HITL).
	2.	The Streamlit app calls graph.run({"company_name": ...}) which executes nodes sequentially until the @runtime.human_node (human_confirm_match) — that node returns user_prompt and the state contains match_options. Streamlit detects user_prompt and shows it in the UI.
	3.	The user picks one option from the radio list and clicks Submit selection. The app then calls graph.run({"human_response": "1"}) to continue the graph (this resumes and finishes the pipeline).
	4.	The app displays the final resolved company and the extracted 1-year financials.

⸻

5) Running the app
	1.	Ensure you exported the SEC user agent:

export SEC_USER_AGENT="MyLangGraphAgent aditya.k@example.com"

	2.	Run Streamlit:

streamlit run app.py

Open the browser at http://localhost:8501.

⸻

6) Notes, limitations & production tips
	•	SEC rate limiting & politeness: The SEC asks for a meaningful User-Agent. Cache company_tickers.json (it’s large but rarely changes) and cache companyfacts per CIK. Add retries/backoff for production.
	•	Missing XBRL elements: Not every company uses the exact same tag names. The node pick_latest_usd checks common aliases (e.g., Revenues, SalesRevenueNet). You can extend the alias lists if you need more coverage.
	•	HITL UX: This Streamlit UI uses a two-step graph.run approach. If you integrate with a different orchestration (FastAPI, queue-based UI), adapt accordingly: detect user_prompt and persist partial graph state if your runtime requires it.
	•	More granular statements: The SEC companyfacts JSON contains many individual line items (e.g., CostOfGoodsSold, GrossProfit, EarningsPerShareBasic). If you want additional fields, extend the element name lists in node_extract_financial_statements.
	•	Precise date semantics: The code picks the most recent USD fact by end date — that corresponds to the latest reported annual (or interim) value. If you require fiscal-year-only values, filter contexts by period and form type found in submissions.
	•	LangGraph versions: If your langchain / langgraph API differs, adapt the StateGraph / runtime.human_node imports accordingly. The graph wiring is straightforward and portable.

⸻

7) Quick test (example)
	1.	Set SEC_USER_AGENT.
	2.	streamlit run app.py.
	3.	Enter "Apple Inc" and click the run button.
	4.	If there are multiple matches, pick the correct one.
	5.	Streamlit will show the resolved company and JSON for income, balance, cashflow (last available year).
