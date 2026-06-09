"""Customer Agent LangGraph definition.

Uses create_react_agent with a `delegate_to_legal_agent` tool that:
1. Discovers the Law Agent via the registry
2. Sends the question to it via A2A
3. Returns the comprehensive legal response to the user

The tool accepts context propagation data (trace_id, context_id, depth)
via a closure — these are bound per-request in agent_executor.py.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from common.llm import get_llm

# Shared in-process memory store — keyed by thread_id (= context_id).
# Persists as long as the customer agent process is alive.
_memory = MemorySaver()

logger = logging.getLogger(__name__)

CUSTOMER_SYSTEM_PROMPT = """You are a helpful legal assistant at the front desk of a multi-agent
legal services platform. Your job is to:

1. Understand the user's legal question
2. Determine if it needs specialist legal analysis (contract issues, tax law,
   regulatory compliance, corporate liability, etc.)
3. If so, use the `delegate_to_legal_agent` tool to send it to the Law Agent,
   which will coordinate specialist sub-agents (Tax and Compliance) as needed
4. Present the comprehensive response clearly to the user

You have access to the full conversation history. When the user asks a follow-up
question, reference prior exchanges so the response is contextually coherent.
If a follow-up clearly builds on a previous question, include that context when
calling `delegate_to_legal_agent` so the Law Agent can provide a more relevant answer.

Always use the `delegate_to_legal_agent` tool for any substantive legal question.
Do not attempt to answer complex legal questions from your own knowledge alone.

Be professional, clear, and make the specialist response accessible to the user.
"""


def build_graph(trace_id: str, context_id: str, depth: int) -> Any:
    """Build a create_react_agent graph with trace context bound into the tool closure.

    Args:
        trace_id: UUID generated at this request's entry point.
        context_id: A2A context_id for this conversation.
        depth: Delegation depth (0 at customer agent).

    Returns:
        A compiled LangGraph agent.
    """

    @tool
    async def delegate_to_legal_agent(question: str) -> str:
        """Send a legal question to the Law Agent for comprehensive analysis.

        The Law Agent will coordinate Tax and Compliance sub-agents in parallel
        and return a synthesised response covering all relevant legal dimensions.

        Args:
            question: The legal question to analyse.

        Returns:
            A comprehensive legal analysis from the multi-agent system.
        """
        from common.a2a_client import delegate
        from common.registry_client import discover

        logger.info(
            "Customer delegate_to_legal_agent | trace=%s context=%s depth=%d",
            trace_id, context_id, depth,
        )

        try:
            endpoint = await discover("legal_question")
            result = await delegate(
                endpoint=endpoint,
                question=question,
                context_id=context_id,
                trace_id=trace_id,
                depth=depth + 1,
            )
            if not result:
                return "The Law Agent returned an empty response. Please try again."
            return result
        except Exception as exc:
            logger.exception("delegate_to_legal_agent failed: %s", exc)
            return f"Could not reach the Law Agent: {exc}"

    llm = get_llm()
    graph = create_react_agent(
        model=llm,
        tools=[delegate_to_legal_agent],
        prompt=CUSTOMER_SYSTEM_PROMPT,
        checkpointer=_memory,
    )
    return graph