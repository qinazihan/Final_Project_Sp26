#!/usr/bin/env python3
"""
Title: MBA Strategy Agent v1 — LangGraph + OpenRouter

A multi-agent business strategy assistant using LangGraph with ChatOpenAI
routed through OpenRouter for access to any LLM provider.

Input:  Business question + parameters (edit SCENARIO below)
Output: MBA-course/results/ (JSON + markdown report)
"""

# %% [markdown]
# # MBA Strategy Agent v1
#
# **Stack:** LangGraph + ChatOpenAI + OpenRouter
#
# OpenRouter gives us one API key to access any model:
# GPT-4o, Claude Sonnet, Gemini Pro, Llama, etc.
#
# **Pipeline:**
# ```
# Clarify -> [Gate 1] -> Plan Topics -> Research Loop ->
# Merge -> Propose -> Critique -> [Gate 2] -> Synthesize -> Action Plan
# ```

# %% Setup and Imports
import os
import json
import operator
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

print("=" * 80)
print("SETUP COMPLETE")
print("=" * 80)

# Optional: Tavily for web search
try:
    from tavily import TavilyClient
    _tavily_key = os.getenv("TAVILY_API_KEY", "")
    _tavily = TavilyClient(api_key=_tavily_key) if _tavily_key else None
    TAVILY_AVAILABLE = bool(_tavily_key)
except ImportError:
    _tavily = None
    TAVILY_AVAILABLE = False

print(f"  Web search: {'Tavily' if TAVILY_AVAILABLE else 'Disabled (set TAVILY_API_KEY)'}")

# %% [markdown]
# ## Configuration
#
# **Students: edit this cell to customize your agent.**
#
# Change the MODEL to try different LLMs via OpenRouter:
# - `"openai/gpt-4o"` — fast, good quality
# - `"openai/gpt-4.1-mini"` — cheapest, fast
# - `"anthropic/claude-sonnet-4-20250514"` — strong reasoning
# - `"google/gemini-2.5-pro-preview"` — Google's best
# - `"meta-llama/llama-4-maverick"` — open source

# %% Configuration — EDIT THIS CELL

# ── OpenRouter setup ──
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Model — pick any model from https://openrouter.ai/models ──
MODEL = os.getenv("MBA_AGENT_MODEL", "openai/gpt-4o")
TEMPERATURE = 0.7

# ── Agent behavior ──
AUTO_APPROVE = True        # True = skip human gates (for batch runs)
MAX_ROUNDS = 2             # max human-feedback cycles
MAX_QUERIES_PER_TOPIC = 3  # web search results per topic
MAX_RESEARCH_TOPICS = 4    # cap on LLM-generated topics

# ── Business scenario — students edit these ──
SCENARIO = {
    "user_query":        "How should I start a beverage business in Sweden?",
    "country_or_market": "Sweden",
    "product_idea":      "Premium organic cold-brew coffee",
    "target_customer":   "Health-conscious urban professionals 25-45",
    "budget_range":      "50k-150k EUR",
    "time_horizon":      "12 months",
    "risk_tolerance":    "moderate",
    "constraints":       "None",
}

# ── Default fallback topics ──
DEFAULT_TOPICS = [
    "Market size and consumer trends",
    "Competitor landscape and go-to-market strategies",
    "Regulatory requirements and operational setup",
    "Unit economics and funding options",
]

print("=" * 80)
print("CONFIGURATION")
print("=" * 80)
print(f"  Model:        {MODEL}")
print(f"  Auto-approve: {AUTO_APPROVE}")
print(f"  Max rounds:   {MAX_ROUNDS}")
print(f"  Query:        {SCENARIO['user_query']}")

# %% [markdown]
# ## Initialize LLM
#
# Create the ChatOpenAI instance pointing at OpenRouter.

# %% Initialize LLM
llm = ChatOpenAI(
    model=MODEL,
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    temperature=TEMPERATURE,
    max_tokens=4096,
)

# Test connection
print("=" * 80)
print("LLM INITIALIZED")
print("=" * 80)
print(f"  Provider: OpenRouter ({OPENROUTER_BASE_URL})")
print(f"  Model:    {MODEL}")

# %% [markdown]
# ## Pydantic Models for Structured Output
#
# These define the expected JSON shape for each LLM node.
# `llm.with_structured_output(Model)` returns parsed objects directly —
# no manual JSON parsing needed.

# %% Pydantic Output Models
class ClarifyOutput(BaseModel):
    """Output of the Clarify Problem node."""
    problem_framing: str = Field(description="Clear 2-3 sentence problem statement")
    assumptions: list[str] = Field(description="List of 4-6 key assumptions")
    constraints_noted: str = Field(description="Key constraints identified")

class TopicsOutput(BaseModel):
    """Output of the Plan Research Topics node."""
    topics: list[str] = Field(description="List of research topics to investigate")

class FindingSource(BaseModel):
    title: str = ""
    url: str = ""

class Finding(BaseModel):
    claim: str = Field(description="Specific research finding")
    confidence: str = Field(description="high, medium, or low")
    sources: list[FindingSource] = Field(default_factory=list)

class ResearchOutput(BaseModel):
    """Output of the Run Topic Research node."""
    topic: str = Field(description="The topic researched")
    findings: list[Finding] = Field(description="List of findings with confidence")
    summary: str = Field(description="2-3 sentence summary")

class OptionItem(BaseModel):
    name: str = Field(description="Option name")
    description: str = Field(description="One-line description")
    advantages: list[str] = Field(description="Key advantages")
    risks: list[str] = Field(description="Key risks")
    investment: str = Field(description="low, medium, or high")
    fit_notes: str = Field(default="", description="How this fits constraints")

class ProposerOutput(BaseModel):
    """Output of the Proposer node."""
    options: list[OptionItem] = Field(description="Exactly 3 strategic options")

print("Pydantic models defined (6 output schemas)")

# %% [markdown]
# ## State Schema

# %% State Schema
class MBAAgentState(TypedDict):
    # User inputs
    user_query: str
    country_or_market: str
    product_idea: str
    target_customer: str
    budget_range: str
    time_horizon: str
    risk_tolerance: str
    constraints: str
    # Clarification
    problem_framing: str
    assumptions: list[str]
    # Research
    research_topics: list[str]
    current_topic_idx: int
    evidence: Annotated[list[dict], operator.add]
    merged_findings: str
    # Proposal & Critique
    proposal_options: list[dict]
    critic_feedback: str
    # Human feedback
    human_feedback_1: str
    human_feedback_2: str
    rerun_target: str
    # Control
    round_num: int
    max_rounds: int
    status: str
    # Final outputs
    recommendation: str
    action_plan: str
    final_output: str

print("State schema defined")

# %% [markdown]
# ## Helper: Web Search

# %% Web Search Helper
def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Search the web via Tavily."""
    if not TAVILY_AVAILABLE or not _tavily:
        return []
    try:
        results = _tavily.search(query=query, max_results=max_results)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in results.get("results", [])
        ]
    except Exception as e:
        print(f"  [SEARCH ERROR] {e}")
        return []

# %% [markdown]
# ## Node Functions
#
# Each node takes the full state and returns only the fields it modifies.

# %% Node: Clarify Problem
CLARIFY_PROMPT = textwrap.dedent("""\
    You are a business strategy consultant helping frame a business question.
    Given the user's input, produce a clear problem framing with key assumptions.
""")

def clarify_problem(state: dict) -> dict:
    print("=" * 80)
    print("NODE: Clarify Problem")
    print("=" * 80)

    feedback_note = ""
    hf = state.get("human_feedback_1", "")
    if hf and hf != "approved":
        feedback_note = f"\nHuman feedback to incorporate: {hf}"

    user_input = f"""Business question: {state['user_query']}
Market: {state.get('country_or_market', 'N/A')}
Product: {state.get('product_idea', 'N/A')}
Target customer: {state.get('target_customer', 'N/A')}
Budget: {state.get('budget_range', 'N/A')}
Time horizon: {state.get('time_horizon', 'N/A')}
Risk tolerance: {state.get('risk_tolerance', 'N/A')}
Constraints: {state.get('constraints', 'None')}{feedback_note}"""

    try:
        structured = llm.with_structured_output(ClarifyOutput)
        result = structured.invoke([SystemMessage(content=CLARIFY_PROMPT),
                                    HumanMessage(content=user_input)])
        framing = result.problem_framing
        assumptions = result.assumptions
    except Exception as e:
        print(f"  [Structured output failed: {e}] Falling back to text")
        resp = llm.invoke([SystemMessage(content=CLARIFY_PROMPT),
                           HumanMessage(content=user_input)])
        framing = resp.content
        assumptions = []

    print(f"  Problem: {framing[:200]}...")
    print(f"  Assumptions: {len(assumptions)} identified")
    return {"problem_framing": framing, "assumptions": assumptions}


# %% Node: Human Gate 1
def human_gate_1(state: dict) -> dict:
    print("=" * 80)
    print("NODE: Human Gate 1 -- Review Problem Framing")
    print("=" * 80)

    print(f"\n  Problem: {state.get('problem_framing', 'N/A')}")
    print(f"  Assumptions:")
    for i, a in enumerate(state.get("assumptions", []), 1):
        print(f"    {i}. {a}")

    if AUTO_APPROVE:
        print("  -> Auto-approved")
        return {"human_feedback_1": "approved"}

    feedback = input("  [Enter]=approve, or type feedback > ").strip()
    if feedback:
        return {"human_feedback_1": feedback}
    return {"human_feedback_1": "approved"}


# %% Node: Plan Research Topics
PLAN_TOPICS_PROMPT = textwrap.dedent("""\
    You are a research planning assistant for business strategy.
    Generate exactly {n} specific, searchable research topics for this business problem.
""")

def plan_research_topics(state: dict) -> dict:
    print("=" * 80)
    print("NODE: Plan Research Topics")
    print("=" * 80)

    prompt = f"""Problem: {state.get('problem_framing', state['user_query'])}
Assumptions: {json.dumps(state.get('assumptions', []))}
Market: {state.get('country_or_market', 'N/A')}"""

    try:
        structured = llm.with_structured_output(TopicsOutput)
        result = structured.invoke([
            SystemMessage(content=PLAN_TOPICS_PROMPT.format(n=MAX_RESEARCH_TOPICS)),
            HumanMessage(content=prompt)])
        topics = result.topics[:MAX_RESEARCH_TOPICS]
    except Exception as e:
        print(f"  [Structured output failed: {e}] Using defaults")
        topics = DEFAULT_TOPICS

    print(f"  Topics ({len(topics)}):")
    for i, t in enumerate(topics, 1):
        print(f"    {i}. {t}")
    return {"research_topics": topics, "current_topic_idx": 0}


# %% Node: Run Topic Research
RESEARCH_PROMPT = textwrap.dedent("""\
    You are a business research analyst. Research the given topic thoroughly.
    For each finding, include a clear claim, confidence level (high/medium/low),
    and supporting sources if web results are provided.
""")

def run_topic_research(state: dict) -> dict:
    idx = state["current_topic_idx"]
    topics = state["research_topics"]
    topic = topics[idx]

    print("=" * 80)
    print(f"NODE: Research Topic {idx+1}/{len(topics)}: {topic}")
    print("=" * 80)

    # Web search context
    search_context = ""
    if TAVILY_AVAILABLE:
        market = state.get("country_or_market", "")
        query = f"{topic} {market}" if market else topic
        results = web_search(query, max_results=MAX_QUERIES_PER_TOPIC)
        if results:
            search_context = "\n\nWeb search results:\n" + "\n".join(
                f"- [{r['title']}]({r['url']}): {r['content'][:300]}" for r in results)
            print(f"  Found {len(results)} web sources")
    else:
        print("  (No web search -- using model knowledge)")

    user_msg = f"""Research this topic: {topic}
Context: {state.get('problem_framing', state['user_query'])}
Market: {state.get('country_or_market', 'N/A')}
{search_context}"""

    try:
        structured = llm.with_structured_output(ResearchOutput)
        result = structured.invoke([SystemMessage(content=RESEARCH_PROMPT),
                                    HumanMessage(content=user_msg)])
        evidence_item = {
            "topic": result.topic,
            "findings": [f.model_dump() for f in result.findings],
            "summary": result.summary,
        }
    except Exception as e:
        print(f"  [Structured output failed: {e}] Falling back to text")
        resp = llm.invoke([SystemMessage(content=RESEARCH_PROMPT),
                           HumanMessage(content=user_msg)])
        evidence_item = {"topic": topic, "findings": [], "summary": resp.content[:500]}

    print(f"  -> {len(evidence_item.get('findings', []))} findings")
    print(f"  -> {evidence_item['summary'][:150]}...")
    return {"evidence": [evidence_item], "current_topic_idx": idx + 1}


# %% Node: Merge Findings
MERGE_PROMPT = textwrap.dedent("""\
    You are a research synthesizer. Combine the evidence into a coherent
    briefing for business decision-makers. Organize by theme, note confidence
    levels, flag data gaps. Target 400-600 words. Be specific.
""")

def merge_findings(state: dict) -> dict:
    print("=" * 80)
    print("NODE: Merge Research Findings")
    print("=" * 80)

    evidence_text = json.dumps(state.get("evidence", []), indent=2, default=str)
    resp = llm.invoke([
        SystemMessage(content=MERGE_PROMPT),
        HumanMessage(content=f"Synthesize:\n{evidence_text}\n\nContext: {state.get('problem_framing', '')}")])

    merged = resp.content
    print(f"  -> Merged {len(state.get('evidence', []))} topics ({len(merged)} chars)")
    return {"merged_findings": merged}


# %% Node: Proposer
PROPOSER_PROMPT = textwrap.dedent("""\
    You are a business strategy consultant. Based on the evidence briefing,
    propose exactly 3 strategic options with advantages, risks, investment
    level, and fit with client constraints.
""")

def proposer(state: dict) -> dict:
    print("=" * 80)
    print("NODE: Propose Strategic Options")
    print("=" * 80)

    user_msg = f"""Problem: {state.get('problem_framing', state['user_query'])}
Assumptions: {json.dumps(state.get('assumptions', []))}
Evidence: {state.get('merged_findings', 'N/A')}
Budget: {state.get('budget_range', 'N/A')}
Risk tolerance: {state.get('risk_tolerance', 'N/A')}
Time horizon: {state.get('time_horizon', 'N/A')}"""

    try:
        structured = llm.with_structured_output(ProposerOutput)
        result = structured.invoke([SystemMessage(content=PROPOSER_PROMPT),
                                    HumanMessage(content=user_msg)])
        options = [opt.model_dump() for opt in result.options]
    except Exception as e:
        print(f"  [Structured output failed: {e}] Falling back to text")
        resp = llm.invoke([SystemMessage(content=PROPOSER_PROMPT),
                           HumanMessage(content=user_msg)])
        options = [{"name": "See response", "description": resp.content[:500],
                    "advantages": [], "risks": [], "investment": "unknown"}]

    for i, opt in enumerate(options, 1):
        print(f"  Option {i}: {opt['name']} — {opt['description'][:100]}")
    return {"proposal_options": options}


# %% Node: Critic
CRITIC_PROMPT = textwrap.dedent("""\
    You are a demanding business strategy critic. Stress-test the options:
    challenge assumptions, identify data gaps, flag underestimated risks,
    compare fairly, note low-confidence claims. Be constructive but rigorous.
""")

def critic(state: dict) -> dict:
    print("=" * 80)
    print("NODE: Critic Review")
    print("=" * 80)

    user_msg = f"""Problem: {state.get('problem_framing', state['user_query'])}
Evidence: {state.get('merged_findings', 'N/A')}
Options: {json.dumps(state.get('proposal_options', []), indent=2, default=str)}"""

    resp = llm.invoke([SystemMessage(content=CRITIC_PROMPT),
                       HumanMessage(content=user_msg)])
    feedback = resp.content
    print(f"  -> Critique ({len(feedback)} chars)")
    print(f"  {feedback[:200]}...")
    return {"critic_feedback": feedback}


# %% Node: Human Gate 2
def human_gate_2(state: dict) -> dict:
    print("=" * 80)
    print("NODE: Human Gate 2 -- Review Options & Critique")
    print("=" * 80)

    for i, opt in enumerate(state.get("proposal_options", []), 1):
        print(f"  Option {i}: {opt.get('name', '?')} — {opt.get('description', '')[:100]}")
    print(f"  Critique: {state.get('critic_feedback', 'N/A')[:200]}...")

    if AUTO_APPROVE:
        rerun, feedback = "approve", "approved"
        print("  -> Auto-approved")
    else:
        print("\n  [Enter]=approve, 'research'=redo research, 'propose'=redo options")
        feedback = input("  > ").strip().lower()
        if not feedback:
            rerun, feedback = "approve", "approved"
        elif feedback in ("research", "propose"):
            rerun = feedback
        else:
            rerun = "research"

    round_num = state.get("round_num", 0) + 1
    if round_num >= state.get("max_rounds", MAX_ROUNDS) and rerun != "approve":
        print(f"  ** Max rounds reached. Forcing finalize. **")
        rerun = "approve"

    print(f"  -> Decision: {rerun} | Round: {round_num}/{MAX_ROUNDS}")
    return {"human_feedback_2": feedback, "rerun_target": rerun, "round_num": round_num}


# %% Node: Synthesizer
SYNTH_PROMPT = textwrap.dedent("""\
    You are a senior business consultant writing a final recommendation report.
    Produce these sections:
    1. **Executive Summary** (5 bullet points)
    2. **Problem Framing & Assumptions**
    3. **Evidence Snapshot** (by theme, with citations)
    4. **Strategic Options** (3 paths with tradeoffs)
    5. **Recommended Option & Rationale**
    6. **Risks & Mitigations**
    Be specific, evidence-based, actionable. Use markdown. Note data gaps explicitly.
""")

def synthesizer(state: dict) -> dict:
    print("=" * 80)
    print("NODE: Synthesize Recommendation")
    print("=" * 80)

    findings = state.get('merged_findings', 'N/A')[:4000]
    critique = state.get('critic_feedback', 'N/A')[:3000]
    options_json = json.dumps(state.get('proposal_options', []), indent=2, default=str)[:3000]

    user_msg = f"""Problem: {state.get('problem_framing', state['user_query'])}
Assumptions: {json.dumps(state.get('assumptions', []))}
Evidence: {findings}
Options: {options_json}
Critique: {critique}
Human preference: {state.get('human_feedback_2', 'N/A')}
Write a comprehensive recommendation report with all 6 sections."""

    synth_llm = ChatOpenAI(
        model=MODEL, base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY,
        temperature=0.5, max_tokens=8192,
    )
    resp = synth_llm.invoke([SystemMessage(content=SYNTH_PROMPT),
                             HumanMessage(content=user_msg)])
    recommendation = resp.content

    if not recommendation:
        print("  [WARNING] Empty response, retrying with shorter prompt...")
        resp = synth_llm.invoke([SystemMessage(content=SYNTH_PROMPT),
                                 HumanMessage(content=f"Problem: {state.get('problem_framing', '')}\nEvidence: {findings[:2000]}\nWrite recommendation.")])
        recommendation = resp.content

    print(f"  -> Report generated ({len(recommendation)} chars)")
    return {"recommendation": recommendation}


# %% Node: Action Plan
ACTION_PLAN_PROMPT = textwrap.dedent("""\
    You are a business execution planner. Create a 90-day action plan:
    ### Days 1-30: Foundation & Validation (3-5 actions, milestones, KPIs)
    ### Days 31-60: Build & Test (3-5 actions, milestones, KPIs)
    ### Days 61-90: Launch & Measure (3-5 actions, milestones, KPIs)
    ### Data Gaps & Next Steps
    Be specific to the business context. Use markdown.
""")

def action_plan_90d(state: dict) -> dict:
    print("=" * 80)
    print("NODE: 90-Day Action Plan")
    print("=" * 80)

    user_msg = f"""Recommendation: {state.get('recommendation', 'N/A')}
Market: {state.get('country_or_market', 'N/A')}
Product: {state.get('product_idea', 'N/A')}
Budget: {state.get('budget_range', 'N/A')}
Timeline: {state.get('time_horizon', 'N/A')}"""

    resp = llm.invoke([SystemMessage(content=ACTION_PLAN_PROMPT),
                       HumanMessage(content=user_msg)])
    plan = resp.content

    final = f"""{'='*60}
FINAL MBA STRATEGY REPORT
{'='*60}

{state.get('recommendation', '[No recommendation]')}

{'='*60}
90-DAY ACTION PLAN
{'='*60}

{plan}

{'='*60}
METADATA
{'='*60}
Model: {MODEL}
Rounds: {state.get('round_num', 1)}/{MAX_ROUNDS}
Topics: {len(state.get('evidence', []))}
Web search: {'Tavily' if TAVILY_AVAILABLE else 'Disabled'}
"""

    print(f"  -> Action plan generated ({len(plan)} chars)")
    return {"action_plan": plan, "final_output": final, "status": "finalize"}


print("All node functions defined (10 nodes)")

# %% [markdown]
# ## Routing Functions

# %% Routing Functions
def route_after_gate_1(state: dict) -> Literal["clarify_problem", "plan_research_topics"]:
    return "plan_research_topics" if state.get("human_feedback_1") == "approved" else "clarify_problem"

def route_after_research(state: dict) -> Literal["run_topic_research", "merge_findings"]:
    return "run_topic_research" if state["current_topic_idx"] < len(state["research_topics"]) else "merge_findings"

def route_after_gate_2(state: dict) -> Literal["plan_research_topics", "proposer", "synthesizer"]:
    t = state.get("rerun_target", "approve")
    if t == "research": return "plan_research_topics"
    if t == "propose": return "proposer"
    return "synthesizer"

print("Routing functions defined")

# %% [markdown]
# ## Build Graph
#
# ```
# START -> clarify -> gate_1 --approve--> plan_topics -> research (loop) -> merge
#                            --revise---> clarify
# merge -> proposer -> critic -> gate_2 --approve---> synthesizer -> action_plan -> END
#                                       --research--> plan_topics
#                                       --propose---> proposer
# ```

# %% Build Graph
def build_graph():
    g = StateGraph(MBAAgentState)
    g.add_node("clarify_problem", clarify_problem)
    g.add_node("human_gate_1", human_gate_1)
    g.add_node("plan_research_topics", plan_research_topics)
    g.add_node("run_topic_research", run_topic_research)
    g.add_node("merge_findings", merge_findings)
    g.add_node("proposer", proposer)
    g.add_node("critic", critic)
    g.add_node("human_gate_2", human_gate_2)
    g.add_node("synthesizer", synthesizer)
    g.add_node("action_plan_90d", action_plan_90d)

    g.add_edge(START, "clarify_problem")
    g.add_edge("clarify_problem", "human_gate_1")
    g.add_conditional_edges("human_gate_1", route_after_gate_1,
                            {"clarify_problem": "clarify_problem",
                             "plan_research_topics": "plan_research_topics"})
    g.add_edge("plan_research_topics", "run_topic_research")
    g.add_conditional_edges("run_topic_research", route_after_research,
                            {"run_topic_research": "run_topic_research",
                             "merge_findings": "merge_findings"})
    g.add_edge("merge_findings", "proposer")
    g.add_edge("proposer", "critic")
    g.add_edge("critic", "human_gate_2")
    g.add_conditional_edges("human_gate_2", route_after_gate_2,
                            {"plan_research_topics": "plan_research_topics",
                             "proposer": "proposer",
                             "synthesizer": "synthesizer"})
    g.add_edge("synthesizer", "action_plan_90d")
    g.add_edge("action_plan_90d", END)
    return g.compile()

agent = build_graph()

print("=" * 80)
print("GRAPH BUILT")
print("=" * 80)
print(f"  Nodes: {len(agent.get_graph().nodes)}")
print(f"  Edges: {len(agent.get_graph().edges)}")

# %% [markdown]
# ## Run the Agent

# %% Run Agent
print("=" * 80)
print("RUNNING MBA STRATEGY AGENT")
print(f"  Model: {MODEL}")
print(f"  Query: {SCENARIO['user_query']}")
print("=" * 80)

initial_state = {
    **SCENARIO,
    "assumptions": [], "research_topics": [], "current_topic_idx": 0,
    "evidence": [], "merged_findings": "", "proposal_options": [],
    "critic_feedback": "", "human_feedback_1": "", "human_feedback_2": "",
    "rerun_target": "", "round_num": 0, "max_rounds": MAX_ROUNDS,
    "status": "continue", "problem_framing": "", "recommendation": "",
    "action_plan": "", "final_output": "",
}

result = agent.invoke(initial_state)

print("=" * 80)
print("AGENT COMPLETE")
print("=" * 80)

# %% [markdown]
# ## View Final Report

# %% Display Report
print(result.get("final_output", "No output generated."))

# %% [markdown]
# ## Save Results

# %% Save Results
print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

json_path = output_dir / f"mba_output_{ts}.json"
serializable = {k: v for k, v in result.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
with open(json_path, "w") as f:
    json.dump(serializable, f, indent=2, ensure_ascii=False)
print(f"  JSON:     {json_path}")

md_path = output_dir / f"mba_report_{ts}.md"
with open(md_path, "w") as f:
    f.write(f"# MBA Strategy Report\n\n")
    f.write(f"**Generated:** {datetime.now().isoformat()}\n")
    f.write(f"**Model:** {MODEL}\n\n---\n\n")
    f.write(result.get("recommendation", ""))
    f.write(f"\n\n---\n\n## 90-Day Action Plan\n\n")
    f.write(result.get("action_plan", ""))
print(f"  Markdown: {md_path}")

print("=" * 80)
print("COMPLETE")
print("=" * 80)
