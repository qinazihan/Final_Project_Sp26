# Sweden Beverage Business Agent — MBA Course Design Requirements (v1)

## 1) Objective
Build an MBA-student-friendly LangGraph agent for the case:
**“How should I start a beverage business in Sweden?”**

The agent should support structured business thinking (not generic chat), with human checkpoints and evidence-backed recommendations.

---

## 2) Notebook Baseline (from LearnLangGraph/langchain-academy)

### Primary base (required)
- `module-4/research-assistant.ipynb`
  - Reason: best fit for multi-step research + synthesis workflow.

### Human-in-the-loop support (required)
- `module-3/edit-state-human-feedback.ipynb`
  - Reason: add approval/edit checkpoints.

### Parallel/advanced support (optional)
- `module-4/map-reduce.ipynb`
  - Reason: optional parallelization for advanced version.

---

## 3) Scope for MBA v1 (complexity control)

### Keep in v1
- Single graph (clear, readable)
- Sequential topic research loop
- 2 required human gates
- Structured final output template

### Do NOT require in v1
- `send()`-based fan-out/fan-in
- subgraph orchestration
- advanced concurrency patterns

Rationale: MBA students should focus on decision quality, not graph-engineering complexity.

---

## 4) Agent Structure (v1)

`START`
→ `Intake`
→ `Clarify_Problem`
→ `Human_Gate_1`
→ `Plan_Research_Topics`
→ loop: (`Pick_Next_Topic` → `Run_Topic_Search` → `Check_More_Topics`)
→ `Merge_Findings`
→ `Proposer`
→ `Critic`
→ `Human_Gate_2`
→ `Synthesizer`
→ `ActionPlan_90D`
→ `END`

### Required research topics (default order)
1. Market
2. Competitor + GTM
3. Regulation + Operations
4. Economics

---

## 5) Multi-round behavior (required)

The design must support iterative improvement:
- Run proposer/critic cycle
- Human can request revisions
- Re-run impacted nodes only (not full restart)

### Loop policy
- Max rounds: 2–3
- If user requests assumption change or missing evidence, re-enter research/proposer/critic as needed

---

## 6) Node responsibilities: LLM vs non-LLM

## LLM-heavy nodes
- `Clarify_Problem`
- `Plan_Research_Topics`
- `Run_Topic_Search` (with tool use)
- `Merge_Findings`
- `Proposer`
- `Critic`
- `Synthesizer`
- `ActionPlan_90D`

## Non-LLM / deterministic nodes
- `Human_Gate_1`, `Human_Gate_2` (interrupt + collect edits)
- `Pick_Next_Topic` (index router)
- `Check_More_Topics` (conditional edge)
- schema/format validation
- stopping criteria and guardrails

---

## 7) Tooling requirement: Web search

Use web search for evidence gathering (required).

### Evidence policy
- At least 2–3 sources per major claim
- Capture URL/source metadata in findings
- Mark confidence level per section (high/medium/low)
- Critic flags unsupported claims and triggers rework

---

## 8) Human-in-the-loop checkpoints (required)

### Gate 1 (after Clarify)
Human approves/edits:
- business question framing
- assumptions
- constraints (budget/timeline/risk)

### Gate 2 (after Critic)
Human approves/edits:
- option preference
- risk posture
- final output emphasis (board/investor/operator style)

Optional Gate 3 (advanced): pre-submission final signoff.

---

## 9) Student input UX (minimal fields)

1. Product idea
2. Target customer
3. Budget range
4. Time horizon
5. Risk tolerance
6. Constraints (optional)

---

## 10) Required final output format (submission-ready)

1. Executive summary (5 bullets)
2. Problem framing + assumptions
3. Evidence snapshot (with citations)
4. Strategic options (3 paths) + tradeoffs
5. Recommended option + rationale
6. Risks + mitigations
7. 90-day action plan + KPIs
8. Data gaps + validation next steps

---

## 11) Optional implementation style for non-coders

Use Node Cards + config-driven assembly:
- Node name
- Purpose
- Inputs
- Outputs
- Prompt style
- Human approval required (yes/no)

This allows students to design workflow logic with minimal Python edits.

---

## 12) v1 Acceptance Criteria

The v1 build is accepted if:
- Runs in Colab with minimal setup
- Includes 2 HITL gates
- Performs topic-by-topic web research sequentially
- Produces the 8-section structured output
- Supports at least one revise-and-rerun cycle
- Keeps graph understandable for non-coder MBA students
