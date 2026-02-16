# 0-Course-Project-Design

## Project Goal
Design an MBA-student-friendly, low-code LangGraph project where students can ask business questions (including, but not limited to, “How to start a beverage business in Sweden?”), collaborate with AI agents, and get structured decision-quality outputs.

---

## What We Have Done So Far

### 1) Environment Setup
- Created project folder: `LearnLangGraph`
- Initialized git repository
- Added `langchain-academy` as a git submodule
- Committed setup:
  - Commit: `0c98da8`
  - Message: `Initialize LearnLangGraph with langchain-academy submodule`

### 2) Notebook Review and Base Selection
From `langchain-academy`, selected:
- **Primary base:** `module-4/research-assistant.ipynb`
  - Best fit for multi-step research + synthesis workflows
- **Human-in-the-loop controls:**
  - `module-3/breakpoints.ipynb`
  - `module-3/edit-state-human-feedback.ipynb`
- **Optional for scaling/parallel research:** `module-4/map-reduce.ipynb`

### 3) Initial Domain-Specific Design (Sweden Beverage Case)
Early design included role-based specialists:
- Market Analyst
- Competition & GTM Analyst
- Regulation & Operations Analyst
- Finance Strategist
- Human Decision Gate at key points

### 4) Shift to General-Purpose Design
Team decided to make the design suitable for broader question types.
Recommended general multi-agent flow:
- Clarifier
- Researcher
- Proposer
- Critic
- Synthesizer
- Optional Human Approval Gate

### 5) Low-Code Learning Design for MBA Students
Agreed approach:
- Keep student interaction minimal and guided
- Start with **Colab** for fastest onboarding
- Use prebuilt “Node Cards” and config-driven graph assembly
- Avoid requiring deep coding knowledge

### 6) Deployment Direction
After notebook prototype:
- **Phase 1:** Streamlit web app (fastest classroom usability)
- **Phase 2 (optional):** FastAPI + frontend for more product-like architecture

### 7) Collaboration Structure in Discord
Created topic threads and posted kickoff messages for each:
1. Project Setup: LearnLangGraph + Submodule
2. Use Case: Sweden Beverage Business Agent
3. MBA-Friendly Colab Experience (Low-Code)
4. Easy Node Design: Node Cards + Config
5. Deploy as Web Service/App (Streamlit/FastAPI)
6. General Multi-Agent Design (Proposer/Critic/HITL)

---

## Current Recommended Architecture (v1)

### Graph Skeleton
`Input -> Clarifier -> Researcher -> Proposer -> Critic -> (Optional Human Gate) -> Synthesizer -> Final Output`

### Why This Works for MBA Courses
- Clear role separation and reasoning transparency
- Structured outputs for decision making (not just chat)
- Human control at critical decision points
- Reusable across industries and problem types

### Standard Output Format
- Executive summary
- Key assumptions
- Options and tradeoffs
- Recommended path
- Risks and mitigations
- 30/60/90-day action plan with KPIs

---

## Next Actions
1. Build a student-ready Colab template around the selected notebooks.
2. Create a Node Card template and config schema (YAML/JSON).
3. Implement v1 general agent graph (Clarifier/Researcher/Proposer/Critic/Synthesizer).
4. Add optional human approval checkpoints.
5. Wrap in Streamlit for simple web interaction.
6. Prepare teaching materials: walkthrough, troubleshooting, and grading rubric.

---

## Notes
- Keep v1 intentionally simple and robust.
- Prioritize usability and reproducibility over feature complexity.
- Ensure each node has explicit input/output contract to reduce student confusion.
