"""Law Agent LangGraph StateGraph definition.

Graph topology:
    analyze_law → check_routing → (parallel) call_tax + call_compliance → aggregate → END

The parallel branches (call_tax / call_compliance) are dispatched via LangGraph's
Send API so that both sub-agent calls happen concurrently.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from common.llm import get_llm

logger = logging.getLogger(__name__)

MAX_DELEGATION_DEPTH = 3


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

def _last_wins(a: str, b: str) -> str:
    """Reducer: keep the most recently written value."""
    return b if b else a


class LawState(TypedDict):
    question: str
    context_id: str
    trace_id: str
    delegation_depth: int
    law_analysis: str
    needs_tax: bool
    needs_compliance: bool
    # Annotated so parallel branches can both write without conflict
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    final_answer: str


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def analyze_and_route(state: LawState) -> dict:
    """Single LLM call: produce legal analysis AND routing decision together.

    Replaces the original two-step analyze_law → check_routing sequence,
    saving one serial LLM round-trip (~2s) per request.
    """
    depth = state.get("delegation_depth", 0)
    if depth >= MAX_DELEGATION_DEPTH:
        logger.info("Max delegation depth reached (%d); skipping sub-agents", depth)
        return {
            "law_analysis": "Max delegation depth reached.",
            "needs_tax": False,
            "needs_compliance": False,
        }

    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                "You are a senior corporate litigation attorney.\n\n"
                "Given the legal question, do TWO things in ONE response:\n"
                "1. Write a concise legal analysis (contract law, tort, liability) under 150 words.\n"
                "2. On the LAST line, output ONLY a JSON object (no markdown, no extra text):\n"
                '   {"needs_tax": <true|false>, "needs_compliance": <true|false>}\n\n'
                "needs_tax=true → question involves tax law, IRS, tax evasion\n"
                "needs_compliance=true → question involves SEC, SOX, AML, FCPA, or regulatory compliance\n\n"
                "Format strictly: analysis text first, then the JSON object on its own line."
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(messages)
    raw = result.content.strip()

    # Split: last line is the JSON routing decision; everything above is the analysis
    lines = raw.splitlines()
    json_line = ""
    analysis_lines = []
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            json_line = stripped
            break
        analysis_lines.insert(0, line)
    else:
        analysis_lines = lines

    if json_line:
        try:
            parsed = json.loads(json_line)
        except json.JSONDecodeError:
            logger.warning("analyze_and_route JSON parse failed: %r", json_line)
            parsed = {"needs_tax": True, "needs_compliance": True}
    else:
        logger.warning("analyze_and_route: no JSON line found, defaulting both=True")
        parsed = {"needs_tax": True, "needs_compliance": True}
        analysis_lines = lines

    law_analysis = "\n".join(analysis_lines).strip()
    needs_tax = bool(parsed.get("needs_tax", True))
    needs_compliance = bool(parsed.get("needs_compliance", True))
    logger.info(
        "analyze_and_route: needs_tax=%s needs_compliance=%s analysis_len=%d",
        needs_tax, needs_compliance, len(law_analysis),
    )
    return {
        "law_analysis": law_analysis,
        "needs_tax": needs_tax,
        "needs_compliance": needs_compliance,
    }


def route_to_subagents(state: LawState) -> list[Send]:
    """Routing function: dispatch parallel Send objects based on routing flags.

    This function is used with add_conditional_edges; it returns a list of
    Send objects which LangGraph executes as parallel branches.
    """
    sends: list[Send] = []
    if state.get("needs_tax"):
        sends.append(Send("call_tax", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance", state))
    if not sends:
        # No sub-agents needed — go straight to aggregation
        sends.append(Send("aggregate", state))
    return sends


async def call_tax(state: LawState) -> dict:
    """Delegate to the Tax Agent via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover

    try:
        endpoint = await discover("tax_question")
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
        )
        logger.info("Tax Agent returned %d chars", len(result))
        return {"tax_result": result}
    except Exception as exc:
        logger.exception("call_tax failed: %s", exc)
        return {"tax_result": f"[Tax analysis unavailable: {exc}]"}


async def call_compliance(state: LawState) -> dict:
    """Delegate to the Compliance Agent via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover

    try:
        endpoint = await discover("compliance_question")
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
        )
        logger.info("Compliance Agent returned %d chars", len(result))
        return {"compliance_result": result}
    except Exception as exc:
        logger.exception("call_compliance failed: %s", exc)
        return {"compliance_result": f"[Compliance analysis unavailable: {exc}]"}


async def aggregate(state: LawState) -> dict:
    """Combine law_analysis, tax_result, and compliance_result into a final answer."""
    llm = get_llm()

    sections: list[str] = []
    if state.get("law_analysis"):
        sections.append(f"## Legal Analysis\n{state['law_analysis']}")
    if state.get("tax_result"):
        sections.append(f"## Tax Analysis\n{state['tax_result']}")
    if state.get("compliance_result"):
        sections.append(f"## Regulatory Compliance Analysis\n{state['compliance_result']}")

    combined = "\n\n---\n\n".join(sections)

    messages = [
        SystemMessage(
            content=(
                "You are a senior legal counsel synthesising specialist analyses into a "
                "comprehensive, well-structured response for the client. Combine the following "
                "analyses into a cohesive answer with clear sections. Avoid redundancy. "
                "End with a brief disclaimer that the analysis is educational and the client "
                "should consult licensed attorneys for their specific situation."
            )
        ),
        HumanMessage(content=combined),
    ]
    result = await llm.ainvoke(messages)
    return {"final_answer": result.content}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_graph():
    """Build and compile the Law Agent StateGraph.

    Optimised topology: analyze_and_route replaces the old two-step
    analyze_law → check_routing sequence, saving one serial LLM call.
    """
    graph = StateGraph(LawState)

    graph.add_node("analyze_and_route", analyze_and_route)
    graph.add_node("call_tax", call_tax)
    graph.add_node("call_compliance", call_compliance)
    graph.add_node("aggregate", aggregate)

    graph.set_entry_point("analyze_and_route")
    graph.add_conditional_edges(
        "analyze_and_route",
        route_to_subagents,
        ["call_tax", "call_compliance", "aggregate"],
    )

    graph.add_edge("call_tax", "aggregate")
    graph.add_edge("call_compliance", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()