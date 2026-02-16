# LangGraph MBA Agent Build Plan (Local First → Colab)

## Goal
Design a **low-code MBA-friendly LangGraph agent** that is easy to teach and reliable to run.

Core workflow (v1):

`Clarify -> Research -> Propose -> Critic -> Human Feedback -> selective rerun -> Final`

Constraints:
- v1 is **sequential only** (no `send()`, no subgraphs)
- Keep logic transparent for non-technical students
- Add evidence quality and confidence labels

---

## Environment Plan

### Local development environment (primary)
Use this Python environment for build/debug:

`/Users/floydluo/Desktop/OpenClawServer/SPACE-HAI-Agent/HAI-Agent/.venv`

Why local first:
- Faster iteration
- Better logs/debugging
- Easy to test graph transitions
- Easy to use `langgraph dev`

### Colab environment (secondary)
Use Colab **after** local graph is stable.

Why second:
- Colab is best for teaching/demo
- Less suitable for deep debugging
- Reduces “infra debugging during class” risk

---

## Build Order (Recommended)

1. Build **one single Python script** skeleton with mock outputs (local)
2. Add real research + citations in the same script (local)
3. Add critic + human feedback loop (same script)
4. Add guardrails (round/token/query limits)
5. Freeze stable v1 script
6. Convert the script to Colab notebook cells (student-facing)
7. Optional: build v2 with parallel subgraphs (`send()`)

---

## Agent Architecture (v1)

```mermaid
flowchart TD
    A[Start] --> B[Clarify_Problem (LLM)]
    B --> C[Plan_Research_Topics (LLM)]
    C --> D[Research_Loop_Router (deterministic)]
    D -->|has_topic| E[Run_Topic_Research (LLM + web search)]
    E --> F[Update_Topic_Index (deterministic)]
    F --> D
    D -->|done| G[Propose_Options (LLM)]
    G --> H[Critic_Review (LLM)]
    H --> I[Human_Gate (interrupt/input)]
    I --> J{Rerun? (deterministic)}
    J -->|yes| K[Selective_Rerun_Router]
    K --> C
    K --> G
    J -->|no| L[Finalize_Recommendation (LLM)]
    L --> M[End]
```

---

## State Schema (v1)

Use a minimal typed state (TypedDict or Pydantic model):

- `user_query: str`
- `country_or_market: str | None`
- `assumptions: list[str]`
- `research_topics: list[str]`
- `current_topic_idx: int`
- `max_queries_per_topic: int`
- `evidence: list[dict]`
  - suggested evidence item fields:
    - `topic: str`
    - `claim: str`
    - `sources: list[dict]` (`title`, `url`, `snippet`)
    - `confidence: str` (`high|med|low`)
- `proposal_options: list[dict]`
- `critic_feedback: str`
- `human_feedback: str | None`
- `rerun_targets: list[str]` (e.g., `research`, `propose`)
- `round_num: int`
- `max_rounds: int` (default 2, max 3)
- `status: str` (`continue|finalize|halted_by_guardrail`)
- `budget_tokens_used: int | None`
- `budget_time_sec_used: int | None`

---

## Node Design: LLM vs Non-LLM

### LLM nodes
1. `Clarify_Problem`
   - normalize query
   - generate assumptions and scope
2. `Plan_Research_Topics`
   - produce 3–6 practical topics
3. `Run_Topic_Research`
   - issue search queries
   - synthesize findings for one topic
4. `Propose_Options`
   - produce strategic options
5. `Critic_Review`
   - stress-test options (risks, missing evidence)
6. `Finalize_Recommendation`
   - concise final recommendation + action plan

### Deterministic nodes
1. `Research_Loop_Router`
2. `Update_Topic_Index`
3. `Rerun_Decision`
4. `Selective_Rerun_Router`
5. `Guardrail_Check`

---

## Web Search Policy

For each major claim:
- require **2–3 sources** minimum
- store source metadata (`title`, `url`, `snippet`)
- if fewer than 2 sources, mark claim as weak

Confidence policy:
- `high`: >=3 consistent, credible sources
- `med`: 2 sources, minor conflicts
- `low`: weak/conflicting/outdated evidence

If confidence is low:
- critic should request targeted re-research
- human gate can choose accept/revise

---

## Guardrails

Set these in state/config:
- `max_rounds = 2` (allow 3 only if needed)
- `max_queries_per_topic = 2` (or 3 max)
- optional token/time budget threshold

Guardrail behavior:
- if round limit reached -> force finalize with limitations
- if budget exceeded -> finalize with “known gaps” note
- if no sufficient evidence -> return explicit uncertainty

---

## Routing Logic (v1)

### Research loop routing
- if `current_topic_idx < len(research_topics)` -> `Run_Topic_Research`
- else -> `Propose_Options`

### Post-human routing
- if user asks to revise assumptions/topics -> rerun `Plan_Research_Topics` then research
- if user asks to improve options only -> rerun `Propose_Options` (+ optional `Critic_Review`)
- if user approves -> `Finalize_Recommendation`

### Round checks
- every loop increments `round_num`
- if `round_num >= max_rounds` -> `Finalize_Recommendation`

---

## Local Implementation Structure

### v1 decision: single-file implementation
Use one file first:

- `LearnLangGraph/MBA-course/Design/mba_langgraph_v1_single_file.py`

Put everything in this file:
- state schema
- all node functions
- routing functions
- graph builder + compile
- `run_demo()` entrypoint
- simple config block (model, rounds, query caps)

Run mode:
- script mode for reproducibility
- optional `langgraph dev` for interactive testing

### Why single file first
- faster to debug during early design
- easier to teach and explain in class
- straightforward to convert into notebook cells later

---

## `langgraph dev` vs scripts vs Colab

### `langgraph dev` (local)
Best for:
- graph debugging
- state inspection
- flow validation

### Plain Python scripts (local)
Best for:
- stable execution
- tests and repeatability
- production-like behavior

### Colab notebook
Best for:
- teaching/demo
- student exercises
- low-friction setup

Recommendation:
- Build with **one single script** + optional `langgraph dev`
- Teach with Colab cells

---

## Colab Conversion Plan

After v1 stabilizes locally:

1. Create notebook with 3 blocks:
   - install/import
   - config + key input via `getpass`
   - run graph + display outputs
2. Keep only student knobs:
   - question
   - country/market
   - max rounds
3. Hide complex internals
4. Print outputs in sections:
   - assumptions
   - evidence table/list
   - options
   - critique
   - final recommendation

---

## 90-Minute Class Flow

1. **10 min**: concept and graph mental model
2. **15 min**: run baseline query in notebook
3. **20 min**: inspect assumptions + evidence quality
4. **20 min**: human feedback and rerun behavior
5. **15 min**: discuss recommendation tradeoffs
6. **10 min**: recap + extension ideas (v2 parallel)

---

## MVP Checklist

- [ ] Single-file script runs end-to-end with sequential loop
- [ ] Human feedback can trigger selective rerun
- [ ] Research includes source metadata
- [ ] Confidence labels appear in outputs
- [ ] Guardrails prevent infinite/expensive loops
- [ ] Final output includes explicit uncertainty notes
- [ ] Colab notebook reproduces same workflow

---

## Risks & Mitigations

1. **Too complex for MBA audience**
   - Mitigation: keep v1 sequential and transparent

2. **Weak or hallucinated evidence**
   - Mitigation: source minimum + confidence labels + critic gate

3. **Infinite/long loops**
   - Mitigation: max rounds/query budget + forced finalize behavior

4. **Colab instability in class**
   - Mitigation: validate locally first; keep notebook minimal

5. **Feature creep into subgraphs too early**
   - Mitigation: gate v2 as optional advanced module only

---

## v2 Upgrade Path (Optional, Advanced)

After students master v1:
- introduce `send()` and subgraphs for parallel topic lanes
- keep top-level business flow unchanged
- only replace internal research block with fan-out/fan-in

This preserves pedagogy while improving performance.
