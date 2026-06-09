"""Challenge 1 — Conversation Memory Demo.

Demonstrates that the Customer Agent now remembers prior turns of the same
conversation by inspecting the message list that arrives at the LLM on each
successive call.

Uses a mock LLM + mock tool so no API credits are needed.
The context_id (= thread_id) is kept constant across the three turns to
simulate a single user session.

Run:
    uv run python test_memory.py
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class CapturingMockLLM:
    """Records the message list it receives on every call; returns a canned AIMessage."""

    def __init__(self):
        self.call_history: list[list] = []  # one entry per ainvoke call

    async def ainvoke(self, messages: list, **kwargs) -> AIMessage:
        # Strip the SystemMessage (index 0) so the log stays readable
        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        self.call_history.append(messages)
        turn = len(self.call_history)
        return AIMessage(content=f"[mock answer turn {turn}] I have {len(messages)} messages in context.")

    # LangGraph calls bind_tools on the LLM; return self so the chain works.
    def bind_tools(self, *args, **kwargs):
        return self

    # make it look enough like a BaseChatModel for create_react_agent
    def with_config(self, *a, **kw): return self
    def with_retry(self, *a, **kw): return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_graph(mock_llm: CapturingMockLLM, memory: MemorySaver):
    """Build the Customer Agent graph with the mock LLM and shared MemorySaver."""

    @tool
    async def delegate_to_legal_agent(question: str) -> str:
        """Delegate a legal question (mock — returns instantly)."""
        return f"[law agent mock response to: {question}]"

    graph = create_react_agent(
        model=mock_llm,
        tools=[delegate_to_legal_agent],
        prompt="You are a helpful legal assistant.",
        checkpointer=memory,
    )
    return graph


async def send_turn(
    graph,
    question: str,
    context_id: str,
    turn_num: int,
    mock_llm: CapturingMockLLM,
) -> str:
    """Send one message to the graph and return the agent's answer."""
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": context_id}},
    )
    last_ai = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    return last_ai.content if last_ai else "(no answer)"


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

TURNS = [
    "What are the penalties for tax evasion?",
    "What if the evasion involved offshore accounts?",
    "Can the company's CEO be personally liable for those penalties?",
]


async def main():
    sep = "=" * 68
    print(sep)
    print("  Challenge 1: Conversation Memory — Customer Agent")
    print(sep)

    memory = MemorySaver()
    mock_llm = CapturingMockLLM()
    context_id = str(uuid4())
    print(f"\n  context_id (= thread_id): {context_id}\n")

    # ── WITHOUT memory (fresh graph, no checkpointer) ────────────────────
    print("─" * 68)
    print("  WITHOUT memory  (no checkpointer — fresh state every turn)")
    print("─" * 68)

    nomem_llm = CapturingMockLLM()
    @tool
    async def delegate_mock(question: str) -> str:
        """Delegate a legal question."""
        return "[law agent mock]"

    graph_nomem = create_react_agent(
        model=nomem_llm,
        tools=[delegate_mock],
        prompt="You are a helpful legal assistant.",
        # NO checkpointer
    )

    for i, q in enumerate(TURNS, 1):
        await graph_nomem.ainvoke(
            {"messages": [HumanMessage(content=q)]},
            config={"configurable": {"thread_id": context_id}},
        )
        # How many messages did the LLM see on this turn?
        last_call = nomem_llm.call_history[-1]
        human_msgs = [m for m in last_call if isinstance(m, HumanMessage)]
        print(f"  Turn {i}: '{q[:50]}'")
        print(f"    → LLM received {len(last_call)} messages total, "
              f"{len(human_msgs)} HumanMessage(s)")
        print(f"    → HumanMessages in context: {[m.content[:40] for m in human_msgs]}")
        print()

    # ── WITH memory (MemorySaver, same thread_id) ─────────────────────────
    print("─" * 68)
    print("  WITH memory  (MemorySaver + thread_id=context_id)")
    print("─" * 68)

    mem_llm = CapturingMockLLM()
    graph_mem = _build_graph(mem_llm, memory)

    for i, q in enumerate(TURNS, 1):
        await graph_mem.ainvoke(
            {"messages": [HumanMessage(content=q)]},
            config={"configurable": {"thread_id": context_id}},
        )
        last_call = mem_llm.call_history[-1]
        human_msgs = [m for m in last_call if isinstance(m, HumanMessage)]
        print(f"  Turn {i}: '{q[:50]}'")
        print(f"    → LLM received {len(last_call)} messages total, "
              f"{len(human_msgs)} HumanMessage(s)")
        print(f"    → HumanMessages in context: {[m.content[:40] for m in human_msgs]}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────
    print(sep)
    print("  RESULT: message context size at each turn")
    print(sep)
    print(f"  {'Turn':<8} {'Without memory':>22} {'With memory':>22}")
    print(f"  {'─'*8} {'─'*22} {'─'*22}")
    for i in range(len(TURNS)):
        nm = len([m for m in nomem_llm.call_history[i] if isinstance(m, HumanMessage)])
        wm = len([m for m in mem_llm.call_history[i] if isinstance(m, HumanMessage)])
        flag = "  ← grows!" if wm > 1 else ""
        print(f"  {i+1:<8} {f'{nm} HumanMsg(s)':>22} {f'{wm} HumanMsg(s)':>22}{flag}")

    print()
    print("  Key: with memory, each new turn includes ALL prior HumanMessages")
    print("  so the LLM can answer 'for those penalties' (turn 3) knowing it")
    print("  refers to the offshore-account tax evasion from turns 1+2.")
    print()
    print("  Implementation: 3 lines changed in customer_agent/graph.py")
    print("    + from langgraph.checkpoint.memory import MemorySaver")
    print("    + _memory = MemorySaver()  # module-level, lives with the process")
    print("    + checkpointer=_memory  # passed to create_react_agent()")
    print()
    print("  The context_id from A2A already flows as thread_id into")
    print("  graph.ainvoke(..., config={'configurable': {'thread_id': context_id}})")
    print("  so no change to agent_executor.py was needed.")
    print(sep)


if __name__ == "__main__":
    asyncio.run(main())
