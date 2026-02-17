# -*- coding: utf-8 -*-
"""
MBA LangGraph Agent v1 — Single-file implementation

A multi-agent business strategy assistant built with LangGraph + LiteLLM.
Designed for MBA students studying business decision-making with AI.

Flow:
  Clarify -> [Human Gate 1] -> Plan Research -> Research Loop ->
  Merge -> Propose -> Critique -> [Human Gate 2] -> Synthesize -> Action Plan

Models: Uses LiteLLM (supports OpenAI, Anthropic, Gemini, OpenRouter, etc.)

Usage:
  source env.sh
  .venv/bin/python MBA-course/mba_agent_v1.py
"""

import os
import json
import operator
import textwrap
from datetime import datetime
from typing import TypedDict, Annotated, Literal

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage, HumanMessage
import litellm

litellm.telemetry = False

# ── Optional: Tavily for web search ──────────────────────────────
try:
    from tavily import TavilyClient
    _tavily_key = os.getenv("TAVILY_API_KEY", "")
    _tavily = TavilyClient(api_key=_tavily_key) if _tavily_key else None
    TAVILY_AVAILABLE = bool(_tavily_key)
except ImportError:
    _tavily = None
    TAVILY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

class Config:
    """Central configuration — edit these to customize behavior.

    Model format (LiteLLM): "provider/model" or just "model"
      - OpenAI:      "gpt-5.2", "gpt-4o"
      - Anthropic:   "claude-opus-4-5-20251101", "claude-sonnet-4-20250514"
      - Gemini:      "gemini/gemini-3-pro-preview"
      - OpenRouter:  "openrouter/openai/gpt-4o"  (set OPENROUTER_API_KEY)
    """

    # Model selection — override with MBA_AGENT_MODEL env var
    MODEL: str = os.getenv("MBA_AGENT_MODEL", "gpt-4o")
    TEMPERATURE: float = 0.7

    # Research limits
    MAX_ROUNDS: int = 2            # max human-feedback cycles
    MAX_QUERIES_PER_TOPIC: int = 3  # web searches per topic

    # Default research topics (fallback)
    DEFAULT_TOPICS: list = [
        "Market size and consumer trends",
        "Competitor landscape and go-to-market strategies",
        "Regulatory requirements and operational setup",
        "Unit economics and funding options",
    ]


print(f"MBA Agent v1 | Model: {Config.MODEL} | Max rounds: {Config.MAX_ROUNDS}")
print(f"Web search: {'Tavily' if TAVILY_AVAILABLE else 'Disabled (set TAVILY_API_KEY)'}")


# ═══════════════════════════════════════════════════════════════════
# 2. STATE SCHEMA
# ═══════════════════════════════════════════════════════════════════

class MBAAgentState(MessagesState):
    # Inherits: messages: Annotated[list[AnyMessage], add_messages]
    # This enables the LangGraph Studio Chat tab.

    # ── User inputs ──
    user_query: str
    country_or_market: str
    product_idea: str
    target_customer: str
    budget_range: str
    time_horizon: str
    risk_tolerance: str
    constraints: str

    # ── Clarification ──
    problem_framing: str
    assumptions: list[str]

    # ── Research ──
    research_topics: list[str]
    current_topic_idx: int
    evidence: Annotated[list[dict], operator.add]   # accumulates across topics
    merged_findings: str

    # ── Proposal & Critique ──
    proposal_options: list[dict]
    critic_feedback: str

    # ── Human feedback ──
    human_feedback_1: str
    human_feedback_2: str
    rerun_target: str          # "research" | "propose" | "approve"

    # ── Control ──
    round_num: int
    max_rounds: int
    status: str                # "continue" | "finalize"

    # ── Final outputs ──
    recommendation: str
    action_plan: str
    final_output: str


# ═══════════════════════════════════════════════════════════════════
# 3. UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def call_llm(system_prompt: str, user_prompt: str,
             model: str = None, temperature: float = None,
             max_tokens: int = 4096) -> str:
    """Call any LLM via LiteLLM."""
    model = model or Config.MODEL
    temperature = temperature if temperature is not None else Config.TEMPERATURE
    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        if content is None:
            print(f"  [LLM WARNING] Model returned None content (finish_reason: {resp.choices[0].finish_reason})")
            return ""
        return content.strip()
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return f"[Error calling {model}: {e}]"


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Search the web via Tavily. Returns list of {title, url, content}."""
    if not TAVILY_AVAILABLE or not _tavily:
        return []
    try:
        results = _tavily.search(query=query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in results.get("results", [])
        ]
    except Exception as e:
        print(f"  [SEARCH ERROR] {e}")
        return []


def extract_json(text: str):
    """Try to extract JSON from LLM response (handles markdown code blocks)."""
    # Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Extract from ```json ... ``` blocks
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _sep(title: str):
    """Print a section separator."""
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ═══════════════════════════════════════════════════════════════════
# 4. NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

# ── 4.0 Intake (Chat → Structured Fields) ─────────────────────

INTAKE_SYSTEM = textwrap.dedent("""\
    You are a business strategy assistant. Parse the user's natural language
    business question into structured fields.

    Respond in JSON:
    {
      "user_query": "The core business question",
      "country_or_market": "Country or market mentioned, or 'Not specified'",
      "product_idea": "Product or service idea, or 'Not specified'",
      "target_customer": "Target customer segment, or 'Not specified'",
      "budget_range": "Budget mentioned, or 'Not specified'",
      "time_horizon": "Time horizon mentioned, or 'Not specified'",
      "risk_tolerance": "Risk tolerance mentioned, or 'Not specified'",
      "constraints": "Any constraints mentioned, or 'None'"
    }
""")


def intake(state: dict) -> dict:
    """Parse the user's initial HumanMessage into structured fields.
    No-op if user_query is already populated (Graph-mode / script-mode)."""
    # If structured fields already populated, skip parsing
    if state.get("user_query"):
        return {}

    # Find the latest HumanMessage from chat
    user_text = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            user_text = m.content.strip()
            break

    if not user_text:
        return {}

    _sep("Intake: Parsing Chat Message")
    response = call_llm(INTAKE_SYSTEM, user_text)
    parsed = extract_json(response)

    if parsed:
        result = {
            "user_query": parsed.get("user_query", user_text),
            "country_or_market": parsed.get("country_or_market", "Not specified"),
            "product_idea": parsed.get("product_idea", "Not specified"),
            "target_customer": parsed.get("target_customer", "Not specified"),
            "budget_range": parsed.get("budget_range", "Not specified"),
            "time_horizon": parsed.get("time_horizon", "Not specified"),
            "risk_tolerance": parsed.get("risk_tolerance", "Not specified"),
            "constraints": parsed.get("constraints", "None"),
        }
        print(f"  Parsed: {result['user_query'][:100]}")
        return result

    # Fallback: use raw text as user_query
    return {"user_query": user_text}


# ── 4.1 Clarify Problem ─────────────────────────────────────────

CLARIFY_SYSTEM = textwrap.dedent("""\
    You are a business strategy consultant helping frame a business question.
    Given the user's input, produce a clear problem framing.

    Respond in JSON format:
    {
      "problem_framing": "A clear 2-3 sentence problem statement",
      "assumptions": ["assumption 1", "assumption 2", "..."],
      "constraints_noted": "Key constraints identified"
    }
""")


def clarify_problem(state: dict) -> dict:
    _sep("Clarify Problem")

    # Include human feedback if this is a re-run
    feedback_note = ""
    hf = state.get("human_feedback_1", "")
    if hf and hf != "approved":
        feedback_note = f"\n\nHuman feedback to incorporate: {hf}"

    user_input = f"""Business question: {state['user_query']}
Market/Country: {state.get('country_or_market', 'Not specified')}
Product idea: {state.get('product_idea', 'Not specified')}
Target customer: {state.get('target_customer', 'Not specified')}
Budget range: {state.get('budget_range', 'Not specified')}
Time horizon: {state.get('time_horizon', 'Not specified')}
Risk tolerance: {state.get('risk_tolerance', 'Not specified')}
Constraints: {state.get('constraints', 'None')}{feedback_note}"""

    response = call_llm(CLARIFY_SYSTEM, user_input)
    parsed = extract_json(response)

    if parsed:
        framing = parsed.get("problem_framing", response)
        assumptions = parsed.get("assumptions", [])
    else:
        framing = response
        assumptions = []

    print(f"\n  Problem: {framing[:200]}...")
    print(f"  Assumptions: {len(assumptions)} identified")

    # Build chat summary for Studio Chat tab
    assumptions_text = "\n".join(f"  {i}. {a}" for i, a in enumerate(assumptions, 1))
    chat_summary = (
        f"**Problem Framing:**\n{framing}\n\n"
        f"**Assumptions:**\n{assumptions_text}\n\n"
        f"Please review. Type **approve** to proceed, or provide feedback to revise."
    )

    return {
        "problem_framing": framing,
        "assumptions": assumptions,
        "messages": [AIMessage(content=chat_summary)],
    }


# ── 4.2 Human Gate 1 ────────────────────────────────────────────

def human_gate_1(state: dict) -> dict:
    """Script mode: prompts via input()."""
    _sep("Human Gate 1 -- Review Problem Framing")

    print(f"\n  Problem framing:\n  {state.get('problem_framing', 'N/A')}")
    print(f"\n  Assumptions:")
    for i, a in enumerate(state.get("assumptions", []), 1):
        print(f"    {i}. {a}")

    print("\n  [Press Enter to approve, or type feedback to revise]")
    feedback = input("  > ").strip()

    if feedback:
        print(f"  -> Feedback received, will re-clarify.")
        return {"human_feedback_1": feedback}
    else:
        print(f"  -> Approved.")
        return {"human_feedback_1": "approved"}


def human_gate_1_studio(state: dict) -> dict:
    """Studio Chat mode: reads the latest HumanMessage for feedback.
    User types 'approve' (or similar) to proceed, or feedback text to revise."""
    # Find latest HumanMessage (user's chat response after the interrupt)
    feedback = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            feedback = m.content.strip()
            break

    if not feedback or feedback.lower() in ("approve", "approved", "ok", "yes", "looks good", "lgtm"):
        return {"human_feedback_1": "approved"}
    return {"human_feedback_1": feedback}


# ── 4.3 Route after Gate 1 ──────────────────────────────────────

def route_after_gate_1(state: dict) -> Literal["clarify_problem", "plan_research_topics"]:
    if state.get("human_feedback_1", "") == "approved":
        return "plan_research_topics"
    return "clarify_problem"


# ── 4.4 Plan Research Topics ────────────────────────────────────

PLAN_TOPICS_SYSTEM = textwrap.dedent("""\
    You are a research planning assistant for business strategy.
    Given a business problem, generate exactly 4 specific, searchable research topics
    that need investigation to make a sound business decision.
    Keep it to 4 topics maximum for efficiency.

    Respond in JSON:
    {"topics": ["topic 1", "topic 2", "..."]}
""")


def plan_research_topics(state: dict) -> dict:
    _sep("Plan Research Topics")

    prompt = f"""Problem: {state.get('problem_framing', state['user_query'])}
Assumptions: {json.dumps(state.get('assumptions', []))}
Market: {state.get('country_or_market', 'Not specified')}"""

    response = call_llm(PLAN_TOPICS_SYSTEM, prompt)
    parsed = extract_json(response)

    topics = parsed.get("topics", Config.DEFAULT_TOPICS) if parsed else Config.DEFAULT_TOPICS

    print(f"\n  Research topics ({len(topics)}):")
    for i, t in enumerate(topics, 1):
        print(f"    {i}. {t}")

    return {
        "research_topics": topics,
        "current_topic_idx": 0,
    }


# ── 4.5 Run Topic Research ──────────────────────────────────────

RESEARCH_SYSTEM = textwrap.dedent("""\
    You are a business research analyst. Research the given topic thoroughly.

    For each major finding, include:
    - A clear claim
    - Supporting sources (if web search results are provided)
    - Confidence level: high (3+ consistent sources), medium (2 sources), low (weak/conflicting)

    Respond in JSON:
    {
      "topic": "the topic researched",
      "findings": [
        {
          "claim": "specific finding",
          "confidence": "high|medium|low",
          "sources": [{"title": "...", "url": "..."}]
        }
      ],
      "summary": "2-3 sentence summary of key insights"
    }
""")


def run_topic_research(state: dict) -> dict:
    idx = state["current_topic_idx"]
    topics = state["research_topics"]
    topic = topics[idx]

    _sep(f"Research Topic {idx+1}/{len(topics)}: {topic}")

    # Web search if available
    search_context = ""
    if TAVILY_AVAILABLE:
        market = state.get("country_or_market", "")
        query = f"{topic} {market}" if market else topic
        results = web_search(query, max_results=Config.MAX_QUERIES_PER_TOPIC)
        if results:
            search_context = "\n\nWeb search results:\n" + "\n".join(
                f"- [{r['title']}]({r['url']}): {r['content'][:300]}"
                for r in results
            )
            print(f"  Found {len(results)} web sources")
    else:
        print("  (No web search -- using model knowledge)")

    prompt = f"""Research this business topic: {topic}
Business context: {state.get('problem_framing', state['user_query'])}
Market: {state.get('country_or_market', 'Not specified')}
{search_context}

Provide specific, actionable findings with confidence levels."""

    response = call_llm(RESEARCH_SYSTEM, prompt)
    parsed = extract_json(response)

    if parsed:
        evidence_item = {
            "topic": parsed.get("topic", topic),
            "findings": parsed.get("findings", []),
            "summary": parsed.get("summary", ""),
        }
    else:
        evidence_item = {
            "topic": topic,
            "findings": [],
            "summary": response[:500],
        }

    n_findings = len(evidence_item.get("findings", []))
    print(f"  -> {n_findings} findings")
    print(f"  -> Summary: {evidence_item['summary'][:150]}...")

    return {
        "evidence": [evidence_item],         # appended via operator.add reducer
        "current_topic_idx": idx + 1,
    }


# ── 4.6 Route: More Topics? ─────────────────────────────────────

def route_after_research(state: dict) -> Literal["run_topic_research", "merge_findings"]:
    if state["current_topic_idx"] < len(state["research_topics"]):
        return "run_topic_research"
    return "merge_findings"


# ── 4.7 Merge Findings ──────────────────────────────────────────

MERGE_SYSTEM = textwrap.dedent("""\
    You are a research synthesizer. Combine the evidence below into a
    coherent briefing document for business decision-makers.

    Organize by theme. Note confidence levels. Flag data gaps explicitly.
    Include source citations where available.

    Target 400-600 words. Be specific, not generic.
""")


def merge_findings(state: dict) -> dict:
    _sep("Merge Research Findings")

    evidence_text = json.dumps(state.get("evidence", []), indent=2, default=str)
    prompt = f"""Synthesize these research findings into a coherent evidence briefing:

{evidence_text}

Business context: {state.get('problem_framing', state['user_query'])}"""

    merged = call_llm(MERGE_SYSTEM, prompt)
    n_topics = len(state.get('evidence', []))
    print(f"  -> Merged {n_topics} topic(s) into briefing ({len(merged)} chars)")

    chat_update = f"Research complete — synthesized findings from {n_topics} topics. Generating strategic options..."

    return {
        "merged_findings": merged,
        "messages": [AIMessage(content=chat_update)],
    }


# ── 4.8 Proposer ────────────────────────────────────────────────

PROPOSER_SYSTEM = textwrap.dedent("""\
    You are a business strategy consultant. Based on the evidence briefing,
    propose exactly 3 strategic options.

    For each option include:
    - Name and one-line description
    - Key advantages (2-3)
    - Key risks (2-3)
    - Estimated investment level (low/medium/high)
    - Fit with client constraints

    Respond in JSON:
    {
      "options": [
        {
          "name": "Option Name",
          "description": "One-line description",
          "advantages": ["adv 1", "adv 2"],
          "risks": ["risk 1", "risk 2"],
          "investment": "low|medium|high",
          "fit_notes": "How this fits constraints"
        }
      ]
    }
""")


def proposer(state: dict) -> dict:
    _sep("Propose Strategic Options")

    prompt = f"""Problem: {state.get('problem_framing', state['user_query'])}
Assumptions: {json.dumps(state.get('assumptions', []))}

Evidence briefing:
{state.get('merged_findings', 'No evidence available')}

Client parameters:
- Budget: {state.get('budget_range', 'Not specified')}
- Risk tolerance: {state.get('risk_tolerance', 'Not specified')}
- Time horizon: {state.get('time_horizon', 'Not specified')}
- Constraints: {state.get('constraints', 'None')}

Propose exactly 3 distinct strategic options."""

    response = call_llm(PROPOSER_SYSTEM, prompt)
    parsed = extract_json(response)

    options = parsed.get("options", []) if parsed else []

    if options:
        for i, opt in enumerate(options, 1):
            print(f"\n  Option {i}: {opt.get('name', 'Unnamed')}")
            print(f"    {opt.get('description', '')[:120]}")
            print(f"    Investment: {opt.get('investment', '?')} | Risks: {len(opt.get('risks', []))}")
    else:
        print("  -> Could not parse structured options, storing raw response")
        options = [{"name": "See raw response", "description": response[:500]}]

    return {"proposal_options": options}


# ── 4.9 Critic ───────────────────────────────────────────────────

CRITIC_SYSTEM = textwrap.dedent("""\
    You are a demanding business strategy critic. Stress-test the proposed options:

    1. Challenge weak or unverified assumptions
    2. Identify missing evidence or data gaps
    3. Flag underestimated risks
    4. Compare options fairly on key criteria
    5. Note which claims have low confidence and need validation

    Be constructive but rigorous. State which option is strongest and why.
    If evidence is insufficient for any recommendation, say so clearly.
""")


def critic(state: dict) -> dict:
    _sep("Critic Review")

    prompt = f"""Problem: {state.get('problem_framing', state['user_query'])}

Evidence briefing:
{state.get('merged_findings', 'N/A')}

Proposed options:
{json.dumps(state.get('proposal_options', []), indent=2, default=str)}

Stress-test these options rigorously."""

    feedback = call_llm(CRITIC_SYSTEM, prompt)
    print(f"  -> Critique ({len(feedback)} chars):")
    print(f"  {feedback[:300]}...")

    # Build chat summary with options + critique
    options_summary = ""
    for i, opt in enumerate(state.get("proposal_options", []), 1):
        advs = ", ".join(opt.get("advantages", [])[:2])
        risks = ", ".join(opt.get("risks", [])[:2])
        options_summary += (
            f"\n**Option {i}: {opt.get('name', 'Unnamed')}**\n"
            f"  {opt.get('description', '')}\n"
            f"  Advantages: {advs}\n"
            f"  Risks: {risks}\n"
            f"  Investment: {opt.get('investment', '?')}\n"
        )

    chat_summary = (
        f"**Strategic Options:**\n{options_summary}\n\n"
        f"**Critic Assessment:**\n{feedback[:1500]}\n\n"
        f"Your decision:\n"
        f"- **approve** — generate final recommendation\n"
        f"- **research** — revise research and re-analyze\n"
        f"- **propose** — keep research, generate new options\n"
        f"- Or type free-form feedback"
    )

    return {
        "critic_feedback": feedback,
        "messages": [AIMessage(content=chat_summary)],
    }


# ── 4.10 Human Gate 2 ───────────────────────────────────────────

def _parse_gate_2_feedback(state: dict, feedback: str) -> dict:
    """Shared logic for gate 2: parse feedback into rerun_target."""
    feedback = (feedback or "").strip().lower()

    if not feedback or feedback == "approved":
        rerun = "approve"
        feedback = "approved"
    elif feedback in ("research", "propose"):
        rerun = feedback
    else:
        rerun = "research"

    round_num = state.get("round_num", 0) + 1

    # Guardrail: enforce max rounds
    if round_num >= state.get("max_rounds", Config.MAX_ROUNDS) and rerun != "approve":
        rerun = "approve"

    return {
        "human_feedback_2": feedback,
        "rerun_target": rerun,
        "round_num": round_num,
    }


def human_gate_2(state: dict) -> dict:
    """Script mode: prompts via input()."""
    _sep("Human Gate 2 -- Review Options & Critique")

    print("\n  === Proposed Options ===")
    for i, opt in enumerate(state.get("proposal_options", []), 1):
        print(f"\n  Option {i}: {opt.get('name', 'Unnamed')}")
        print(f"    {opt.get('description', '')[:150]}")
        for adv in opt.get("advantages", []):
            print(f"    + {adv}")
        for risk in opt.get("risks", []):
            print(f"    - {risk}")

    print(f"\n  === Critic Feedback ===")
    print(f"  {state.get('critic_feedback', 'N/A')[:500]}")

    print(f"\n  === Your Decision ===")
    print("    [Enter]      -> Approve and generate final recommendation")
    print("    'research'   -> Revise research topics and re-research")
    print("    'propose'    -> Keep research, revise options only")
    print("    Or type free-form feedback")

    feedback = input("  > ").strip()
    result = _parse_gate_2_feedback(state, feedback)
    print(f"  -> Decision: {result['rerun_target']} | Round: {result['round_num']}/{state.get('max_rounds', Config.MAX_ROUNDS)}")
    return result


def human_gate_2_studio(state: dict) -> dict:
    """Studio Chat mode: reads the latest HumanMessage for feedback.
    User types 'approve', 'research', 'propose', or free-form feedback."""
    # Find latest HumanMessage (user's chat response after the interrupt)
    feedback = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            feedback = m.content.strip()
            break
    return _parse_gate_2_feedback(state, feedback)


# ── 4.11 Route after Gate 2 ─────────────────────────────────────

def route_after_gate_2(state: dict) -> Literal["plan_research_topics", "proposer", "synthesizer"]:
    target = state.get("rerun_target", "approve")
    if target == "research":
        return "plan_research_topics"
    elif target == "propose":
        return "proposer"
    return "synthesizer"


# ── 4.12 Synthesizer ────────────────────────────────────────────

SYNTH_SYSTEM = textwrap.dedent("""\
    You are a senior business consultant writing a final recommendation report.

    Based on all evidence, options, and critique, produce these sections:

    1. **Executive Summary** (5 bullet points)
    2. **Problem Framing & Assumptions**
    3. **Evidence Snapshot** (organized by theme, with citations)
    4. **Strategic Options** (3 paths with tradeoffs table)
    5. **Recommended Option & Rationale**
    6. **Risks & Mitigations**

    Be specific, evidence-based, and actionable.
    If evidence is weak on any point, note it explicitly as a data gap.

    Use markdown formatting.
""")


def synthesizer(state: dict) -> dict:
    _sep("Synthesize Recommendation")

    # Truncate long fields to fit within model context
    findings = state.get('merged_findings', 'N/A')[:4000]
    critique = state.get('critic_feedback', 'N/A')[:3000]
    options_json = json.dumps(state.get('proposal_options', []), indent=2, default=str)[:3000]

    prompt = f"""Problem: {state.get('problem_framing', state['user_query'])}
Assumptions: {json.dumps(state.get('assumptions', []))}

Evidence briefing (summarized):
{findings}

Proposed options:
{options_json}

Critic feedback (summarized):
{critique}

Human preference: {state.get('human_feedback_2', 'N/A')}

Write a comprehensive recommendation report with all 6 sections."""

    recommendation = call_llm(SYNTH_SYSTEM, prompt, temperature=0.5, max_tokens=8192)

    if not recommendation:
        print("  [WARNING] Empty response from synthesizer, retrying with shorter prompt...")
        short_prompt = f"""Problem: {state.get('problem_framing', state['user_query'])}

Key evidence: {findings[:2000]}

Recommended option: Based on the analysis, write a recommendation report."""
        recommendation = call_llm(SYNTH_SYSTEM, short_prompt, temperature=0.5, max_tokens=8192)

    print(f"  -> Recommendation report generated ({len(recommendation)} chars)")

    return {
        "recommendation": recommendation,
        "messages": [AIMessage(content=f"**Recommendation Report:**\n\n{recommendation}")],
    }


# ── 4.13 Action Plan (90-day) ────────────────────────────────────

ACTION_PLAN_SYSTEM = textwrap.dedent("""\
    You are a business execution planner. Create a 90-day action plan.

    Structure:
    ## 90-Day Action Plan

    ### Days 1-30: Foundation & Validation
    - 3-5 specific actions with owners
    - Key milestones
    - KPIs to track

    ### Days 31-60: Build & Test
    - 3-5 specific actions
    - Key milestones
    - KPIs to track

    ### Days 61-90: Launch & Measure
    - 3-5 specific actions
    - Key milestones
    - KPIs to track

    ### Data Gaps & Next Steps
    - Items needing further validation
    - Recommended actions beyond 90 days

    Be specific to the business context. Use markdown formatting.
""")


def action_plan_90d(state: dict) -> dict:
    _sep("90-Day Action Plan")

    prompt = f"""Recommendation:
{state.get('recommendation', 'N/A')}

Business context:
- Market: {state.get('country_or_market', 'N/A')}
- Product: {state.get('product_idea', 'N/A')}
- Budget: {state.get('budget_range', 'N/A')}
- Timeline: {state.get('time_horizon', 'N/A')}
- Risk tolerance: {state.get('risk_tolerance', 'N/A')}

Create a detailed, actionable 90-day plan."""

    plan = call_llm(ACTION_PLAN_SYSTEM, prompt, temperature=0.5)

    # Assemble final output
    divider = "=" * 60
    final = f"""{divider}
FINAL MBA STRATEGY REPORT
{divider}

{state.get('recommendation', '[No recommendation generated]')}

{divider}
90-DAY ACTION PLAN
{divider}

{plan}

{divider}
METADATA
{divider}
Rounds completed: {state.get('round_num', 1)} of {state.get('max_rounds', Config.MAX_ROUNDS)}
Research topics covered: {len(state.get('evidence', []))}
Model: {Config.MODEL}
Web search: {'Enabled (Tavily)' if TAVILY_AVAILABLE else 'Disabled'}
"""

    print(f"  -> Action plan generated ({len(plan)} chars)")

    return {
        "action_plan": plan,
        "final_output": final,
        "status": "finalize",
        "messages": [AIMessage(content=f"**Final MBA Strategy Report:**\n\n{final}")],
    }


# ═══════════════════════════════════════════════════════════════════
# 5. BUILD GRAPH
# ═══════════════════════════════════════════════════════════════════

def build_graph(studio_mode: bool = False):
    """Build and compile the MBA agent graph.

    Args:
        studio_mode: If True, use Studio-compatible human gates with interrupt_before.
                     If False, use input()-based gates for script mode.

    Graph structure (studio_mode):
      START -> intake -> clarify_problem -> human_gate_1 --(approve)--> plan_research_topics
                                                         --(revise)---> clarify_problem

    Graph structure (script_mode):
      START -> clarify_problem -> human_gate_1 --(approve)--> plan_research_topics
                                               --(revise)---> clarify_problem

      plan_research_topics -> run_topic_research --(more topics)--> run_topic_research
                                                 --(done)--------> merge_findings

      merge_findings -> proposer -> critic -> human_gate_2 --(approve)---> synthesizer
                                                           --(research)--> plan_research_topics
                                                           --(propose)---> proposer

      synthesizer -> action_plan_90d -> END
    """

    g = StateGraph(MBAAgentState)

    # Add nodes — human gates differ by mode
    g.add_node("clarify_problem", clarify_problem)
    g.add_node("plan_research_topics", plan_research_topics)
    g.add_node("run_topic_research", run_topic_research)
    g.add_node("merge_findings", merge_findings)
    g.add_node("proposer", proposer)
    g.add_node("critic", critic)
    g.add_node("synthesizer", synthesizer)
    g.add_node("action_plan_90d", action_plan_90d)

    if studio_mode:
        g.add_node("intake", intake)
        g.add_node("human_gate_1", human_gate_1_studio)
        g.add_node("human_gate_2", human_gate_2_studio)
    else:
        g.add_node("human_gate_1", human_gate_1)
        g.add_node("human_gate_2", human_gate_2)

    # Edges: start -> [intake ->] clarify -> human gate 1
    if studio_mode:
        g.add_edge(START, "intake")
        g.add_edge("intake", "clarify_problem")
    else:
        g.add_edge(START, "clarify_problem")
    g.add_edge("clarify_problem", "human_gate_1")

    # Gate 1: approve or loop back to clarify
    g.add_conditional_edges(
        "human_gate_1", route_after_gate_1,
        {"clarify_problem": "clarify_problem", "plan_research_topics": "plan_research_topics"},
    )

    # Research loop
    g.add_edge("plan_research_topics", "run_topic_research")
    g.add_conditional_edges(
        "run_topic_research", route_after_research,
        {"run_topic_research": "run_topic_research", "merge_findings": "merge_findings"},
    )

    # Merge -> propose -> critique -> human gate 2
    g.add_edge("merge_findings", "proposer")
    g.add_edge("proposer", "critic")
    g.add_edge("critic", "human_gate_2")

    # Gate 2: approve, re-research, or re-propose
    g.add_conditional_edges(
        "human_gate_2", route_after_gate_2,
        {
            "plan_research_topics": "plan_research_topics",
            "proposer": "proposer",
            "synthesizer": "synthesizer",
        },
    )

    # Final: synthesize -> action plan -> end
    g.add_edge("synthesizer", "action_plan_90d")
    g.add_edge("action_plan_90d", END)

    if studio_mode:
        return g.compile(interrupt_before=["human_gate_1", "human_gate_2"])
    return g.compile()


# ═══════════════════════════════════════════════════════════════════
# Module-level graph for `langgraph dev` / Studio
# ═══════════════════════════════════════════════════════════════════
graph = build_graph(studio_mode=True)


# ═══════════════════════════════════════════════════════════════════
# 6. RUN
# ═══════════════════════════════════════════════════════════════════

def run_demo():
    """Interactive demo — run the MBA agent with a business question."""

    print("\n" + "=" * 60)
    print("  MBA STRATEGY AGENT v1")
    print(f"  Model: {Config.MODEL}")
    print("=" * 60)

    # Collect inputs
    print("\n  Enter your business scenario (press Enter for defaults):\n")

    query = input("  Business question [How should I start a beverage business in Sweden?]: ").strip()
    if not query:
        query = "How should I start a beverage business in Sweden?"

    market = input("  Country/Market [Sweden]: ").strip() or "Sweden"
    product = input("  Product idea [Premium organic cold-brew coffee]: ").strip() or "Premium organic cold-brew coffee"
    customer = input("  Target customer [Health-conscious urban professionals 25-45]: ").strip() or "Health-conscious urban professionals 25-45"
    budget = input("  Budget range [50k-150k EUR]: ").strip() or "50k-150k EUR"
    timeline = input("  Time horizon [12 months]: ").strip() or "12 months"
    risk = input("  Risk tolerance [moderate]: ").strip() or "moderate"
    constraints = input("  Constraints [none]: ").strip() or "None"

    # Initial state
    initial_state = {
        "messages": [],
        "user_query": query,
        "country_or_market": market,
        "product_idea": product,
        "target_customer": customer,
        "budget_range": budget,
        "time_horizon": timeline,
        "risk_tolerance": risk,
        "constraints": constraints,
        "assumptions": [],
        "research_topics": [],
        "current_topic_idx": 0,
        "evidence": [],
        "merged_findings": "",
        "proposal_options": [],
        "critic_feedback": "",
        "human_feedback_1": "",
        "human_feedback_2": "",
        "rerun_target": "",
        "round_num": 0,
        "max_rounds": Config.MAX_ROUNDS,
        "status": "continue",
        "problem_framing": "",
        "recommendation": "",
        "action_plan": "",
        "final_output": "",
    }

    # Build and run
    agent = build_graph()
    print("\n  Starting agent pipeline...\n")

    result = agent.invoke(initial_state)

    # Print final output
    print("\n" + result.get("final_output", "No output generated."))

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(outdir, f"mba_output_{ts}.json")

    serializable = {}
    for k, v in result.items():
        if k == "messages":
            # Convert LangChain message objects to dicts for JSON
            serializable[k] = [
                {"role": getattr(m, "type", "unknown"), "content": getattr(m, "content", str(m))}
                for m in v
            ]
        elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
            serializable[k] = v

    with open(outfile, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {outfile}")


if __name__ == "__main__":
    run_demo()
