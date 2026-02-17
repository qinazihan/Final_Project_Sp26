"""
Chainlit Chat Interface for MBA Strategy Agent v3.

Uses astream() to show real-time progress as each node completes.

Run from MBA-course directory:
    source ../env.sh && python -m chainlit run mba_chainlit.py

Or with the venv explicitly:
    source ../env.sh && ../.venv/bin/python -m chainlit run mba_chainlit.py
"""
import chainlit as cl
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# Import the graph builder from v3
from mba_agent_v3_nb import build_graph, MAX_ROUNDS

# Build graph with in-memory checkpointer (required for interrupt/resume)
agent = build_graph(checkpointer=MemorySaver())


@cl.on_chat_start
async def start():
    """Initialize a new conversation thread."""
    cl.user_session.set("thread_id", str(uuid4()))
    cl.user_session.set("started", False)

    await cl.Message(
        content=(
            "**MBA Strategy Agent**\n\n"
            "Describe your business idea and I'll create a full MBA strategy analysis "
            "with research, strategic options, and a 90-day action plan.\n\n"
            "Example: *I want to start a premium cold-brew coffee business in Sweden "
            "with a budget of 100k EUR, targeting health-conscious professionals aged 25-45.*"
        )
    ).send()


async def stream_graph(graph_input, config):
    """Stream graph execution, showing AI messages as each node completes."""
    async for chunk in agent.astream(graph_input, config, stream_mode="updates"):
        for node_name, updates in chunk.items():
            if not isinstance(updates, dict):
                continue
            # Show any AIMessages produced by this node
            for m in updates.get("messages", []):
                if isinstance(m, AIMessage) and m.content:
                    await cl.Message(content=m.content).send()

    # After stream ends, check if interrupted or completed
    snapshot = agent.get_state(config)
    if snapshot.next:
        # Graph is paused at an interrupt — show the prompt
        for task in snapshot.tasks:
            if hasattr(task, "interrupts"):
                for intr in task.interrupts:
                    await cl.Message(
                        content=f"---\n**Your input needed:**\n\n{intr.value}"
                    ).send()
    else:
        # Graph completed
        await cl.Message(
            content="---\nAnalysis complete! Click **New Chat** to start another analysis."
        ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle each user message — either start the graph or resume from interrupt."""
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}
    started = cl.user_session.get("started")

    # Check if graph already completed
    if started:
        snapshot = agent.get_state(config)
        if not snapshot.next:
            await cl.Message(
                content="The analysis is complete. Click **New Chat** to analyze a different business idea."
            ).send()
            return

    if not started:
        # First message — start the graph with the user's business question
        initial_state = {
            "messages": [HumanMessage(content=message.content)],
            "user_query": "",  # empty so intake parses from message
            "country_or_market": "", "product_idea": "", "target_customer": "",
            "budget_range": "", "time_horizon": "", "risk_tolerance": "",
            "constraints": "",
            "assumptions": [], "research_topics": [], "current_topic_idx": 0,
            "evidence": [], "merged_findings": "", "proposal_options": [],
            "critic_feedback": "", "human_feedback_1": "", "human_feedback_2": "",
            "rerun_target": "", "round_num": 0, "max_rounds": MAX_ROUNDS,
            "status": "continue", "problem_framing": "", "recommendation": "",
            "action_plan": "", "final_output": "",
        }
        cl.user_session.set("started", True)
        await stream_graph(initial_state, config)
    else:
        # Resume from interrupt with user's feedback
        await stream_graph(Command(resume=message.content), config)
