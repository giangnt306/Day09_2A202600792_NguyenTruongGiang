"""Compliance Agent LangGraph definition.

Uses create_react_agent with a regulatory-compliance-specialised system prompt.
No tools — it answers purely from LLM knowledge.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from common.llm import get_llm

COMPLIANCE_SYSTEM_PROMPT = """You are a senior regulatory compliance officer.

Answer compliance questions concisely in 3-5 bullet points maximum. Cover:
- Relevant regulation and enforcing agency
- Key penalty range (fines / prison if applicable)
- Individual vs. corporate liability distinction

No preamble. No disclaimer. Plain bullets only.
"""


def create_graph():
    """Return a compiled LangGraph create_react_agent for compliance questions."""
    llm = get_llm()
    graph = create_react_agent(
        model=llm,
        tools=[],
        prompt=COMPLIANCE_SYSTEM_PROMPT,
    )
    return graph