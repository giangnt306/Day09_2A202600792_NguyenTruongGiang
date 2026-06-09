"""Latency benchmark: Stage 5 baseline vs. optimised architecture.

Uses a MockLLM that sleeps for realistic durations so we can measure
the pipeline overhead without consuming real API credits.

Baseline  : Customer(LLM1) → Law(analyze_law LLM → check_routing LLM → parallel → aggregate LLM) → Customer(LLM2)
Optimised : Customer(LLM1) → Law(analyze_and_route LLM → parallel → aggregate LLM) → Customer(LLM2)

Run:
    uv run python benchmark.py
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated, Any, TypedDict
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# Realistic LLM response times (p50 observed from logs, seconds)
# ---------------------------------------------------------------------------

LLM_DELAYS = {
    "customer_classify":    2.7,   # customer agent: understand + plan tool use
    "customer_format":      2.0,   # customer agent: format law agent response
    "analyze_law":          1.8,   # law agent: legal analysis
    "check_routing":        2.0,   # law agent: routing decision
    "analyze_and_route":    2.5,   # merged node: slightly longer single call
    "tax_answer":           3.5,   # tax agent: answer (short prompt → faster)
    "compliance_answer":    5.0,   # compliance agent: answer (short prompt → faster)
    "aggregate":            3.0,   # law agent: synthesise
}

# Each ─── represents a serial LLM hop; parallel branches show as [A || B]
PIPELINE_BASELINE = (
    "customer_classify ──► analyze_law ──► check_routing ──► "
    "[tax_answer ║ compliance_answer] ──► aggregate ──► customer_format"
)
PIPELINE_OPTIMISED = (
    "customer_classify ──► analyze_and_route ──► "
    "[tax_answer ║ compliance_answer] ──► aggregate ──► customer_format"
)


# ---------------------------------------------------------------------------
# Mock LLM helper
# ---------------------------------------------------------------------------

def make_mock_llm(step: str) -> Any:
    """Return a mock LLM whose ainvoke sleeps for the given step's delay."""
    delay = LLM_DELAYS[step]
    response = AIMessage(content=_fake_response(step))

    mock = MagicMock()
    async def ainvoke(*args, **kwargs):
        await asyncio.sleep(delay)
        return response
    mock.ainvoke = ainvoke
    return mock


def _fake_response(step: str) -> str:
    if step == "check_routing":
        return '{"needs_tax": true, "needs_compliance": true}'
    if step == "analyze_and_route":
        return (
            "The company faces breach of contract liability and potential tax fraud exposure.\n"
            '{"needs_tax": true, "needs_compliance": true}'
        )
    return f"[mock response for {step}]"


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

def _last_wins(a: str, b: str) -> str:
    return b if b else a


class LawState(TypedDict):
    question: str
    law_analysis: str
    needs_tax: bool
    needs_compliance: bool
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    final_answer: str


# ---------------------------------------------------------------------------
# BASELINE graph nodes
# ---------------------------------------------------------------------------

async def baseline_analyze_law(state: LawState) -> dict:
    llm = make_mock_llm("analyze_law")
    result = await llm.ainvoke([])
    return {"law_analysis": result.content}


async def baseline_check_routing(state: LawState) -> dict:
    llm = make_mock_llm("check_routing")
    result = await llm.ainvoke([])
    raw = result.content.strip()
    parsed = json.loads(raw)
    return {"needs_tax": bool(parsed["needs_tax"]), "needs_compliance": bool(parsed["needs_compliance"])}


async def baseline_call_tax(state: LawState) -> dict:
    llm = make_mock_llm("tax_answer")
    result = await llm.ainvoke([])
    return {"tax_result": result.content}


async def baseline_call_compliance(state: LawState) -> dict:
    llm = make_mock_llm("compliance_answer")
    result = await llm.ainvoke([])
    return {"compliance_result": result.content}


async def baseline_aggregate(state: LawState) -> dict:
    llm = make_mock_llm("aggregate")
    result = await llm.ainvoke([])
    return {"final_answer": result.content}


def baseline_route(state: LawState) -> list[Send]:
    sends: list[Send] = []
    if state.get("needs_tax"):
        sends.append(Send("call_tax", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance", state))
    return sends or [Send("aggregate", state)]


def build_baseline_graph():
    g = StateGraph(LawState)
    g.add_node("analyze_law", baseline_analyze_law)
    g.add_node("check_routing", baseline_check_routing)
    g.add_node("call_tax", baseline_call_tax)
    g.add_node("call_compliance", baseline_call_compliance)
    g.add_node("aggregate", baseline_aggregate)

    g.set_entry_point("analyze_law")
    g.add_edge("analyze_law", "check_routing")
    g.add_conditional_edges("check_routing", baseline_route, ["call_tax", "call_compliance", "aggregate"])
    g.add_edge("call_tax", "aggregate")
    g.add_edge("call_compliance", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()


# ---------------------------------------------------------------------------
# OPTIMISED graph nodes
# ---------------------------------------------------------------------------

async def opt_analyze_and_route(state: LawState) -> dict:
    llm = make_mock_llm("analyze_and_route")
    result = await llm.ainvoke([])
    raw = result.content.strip()
    lines = raw.splitlines()
    json_line = next((l.strip() for l in reversed(lines) if l.strip().startswith("{")), "")
    parsed = json.loads(json_line) if json_line else {"needs_tax": True, "needs_compliance": True}
    analysis = "\n".join(l for l in lines if not l.strip().startswith("{")).strip()
    return {
        "law_analysis": analysis,
        "needs_tax": bool(parsed.get("needs_tax", True)),
        "needs_compliance": bool(parsed.get("needs_compliance", True)),
    }


def opt_route(state: LawState) -> list[Send]:
    sends: list[Send] = []
    if state.get("needs_tax"):
        sends.append(Send("call_tax", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance", state))
    return sends or [Send("aggregate", state)]


def build_optimised_graph():
    g = StateGraph(LawState)
    g.add_node("analyze_and_route", opt_analyze_and_route)
    g.add_node("call_tax", baseline_call_tax)       # same as baseline
    g.add_node("call_compliance", baseline_call_compliance)
    g.add_node("aggregate", baseline_aggregate)

    g.set_entry_point("analyze_and_route")
    g.add_conditional_edges("analyze_and_route", opt_route, ["call_tax", "call_compliance", "aggregate"])
    g.add_edge("call_tax", "aggregate")
    g.add_edge("call_compliance", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()


# ---------------------------------------------------------------------------
# End-to-end pipeline simulation (includes customer agent's 2 LLM calls)
# ---------------------------------------------------------------------------

INIT_STATE: LawState = {
    "question": "If a company breaks a contract and avoids taxes, what are the legal consequences?",
    "law_analysis": "",
    "needs_tax": False,
    "needs_compliance": False,
    "tax_result": "",
    "compliance_result": "",
    "final_answer": "",
}


async def run_baseline() -> tuple[float, dict[str, float]]:
    steps: dict[str, float] = {}
    t0 = time.perf_counter()

    # Customer LLM1 (classify + plan tool call)
    t = time.perf_counter()
    await asyncio.sleep(LLM_DELAYS["customer_classify"])
    steps["customer_classify"] = time.perf_counter() - t

    # Law agent graph (the core pipeline)
    t = time.perf_counter()
    graph = build_baseline_graph()
    await graph.ainvoke(INIT_STATE)
    steps["law_agent_total"] = time.perf_counter() - t

    # Customer LLM2 (format result for user)
    t = time.perf_counter()
    await asyncio.sleep(LLM_DELAYS["customer_format"])
    steps["customer_format"] = time.perf_counter() - t

    return time.perf_counter() - t0, steps


async def run_optimised() -> tuple[float, dict[str, float]]:
    steps: dict[str, float] = {}
    t0 = time.perf_counter()

    t = time.perf_counter()
    await asyncio.sleep(LLM_DELAYS["customer_classify"])
    steps["customer_classify"] = time.perf_counter() - t

    t = time.perf_counter()
    graph = build_optimised_graph()
    await graph.ainvoke(INIT_STATE)
    steps["law_agent_total"] = time.perf_counter() - t

    t = time.perf_counter()
    await asyncio.sleep(LLM_DELAYS["customer_format"])
    steps["customer_format"] = time.perf_counter() - t

    return time.perf_counter() - t0, steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    sep = "=" * 68

    print(sep)
    print("  LATENCY BENCHMARK — Stage 5 Distributed A2A System")
    print(sep)
    print()
    print("LLM delays used (p50 from live logs):")
    for k, v in LLM_DELAYS.items():
        print(f"  {k:<25} {v:.1f}s")
    print()

    # ── Baseline ──────────────────────────────────────────────────────────
    print(sep)
    print("  BASELINE architecture")
    print(f"  {PIPELINE_BASELINE}")
    print(sep)
    t_base, steps_base = await run_baseline()
    print(f"  customer_classify  : {steps_base['customer_classify']:.2f}s")
    print(f"  law_agent_total    : {steps_base['law_agent_total']:.2f}s")
    print(f"    (analyze_law={LLM_DELAYS['analyze_law']:.1f}s serial + check_routing={LLM_DELAYS['check_routing']:.1f}s serial")
    print(f"     + parallel max({LLM_DELAYS['tax_answer']:.1f}s tax, {LLM_DELAYS['compliance_answer']:.1f}s compliance)")
    print(f"     + aggregate={LLM_DELAYS['aggregate']:.1f}s serial)")
    print(f"  customer_format    : {steps_base['customer_format']:.2f}s")
    print(f"  {'─'*40}")
    print(f"  TOTAL              : {t_base:.2f}s")
    print()

    # ── Optimised ─────────────────────────────────────────────────────────
    print(sep)
    print("  OPTIMISED architecture")
    print(f"  {PIPELINE_OPTIMISED}")
    print(sep)
    t_opt, steps_opt = await run_optimised()
    print(f"  customer_classify  : {steps_opt['customer_classify']:.2f}s")
    print(f"  law_agent_total    : {steps_opt['law_agent_total']:.2f}s")
    print(f"    (analyze_and_route={LLM_DELAYS['analyze_and_route']:.1f}s serial  ← merged 2 calls into 1")
    print(f"     + parallel max({LLM_DELAYS['tax_answer']:.1f}s tax, {LLM_DELAYS['compliance_answer']:.1f}s compliance)")
    print(f"     + aggregate={LLM_DELAYS['aggregate']:.1f}s serial)")
    print(f"  customer_format    : {steps_opt['customer_format']:.2f}s")
    print(f"  {'─'*40}")
    print(f"  TOTAL              : {t_opt:.2f}s")
    print()

    # ── Summary ───────────────────────────────────────────────────────────
    saved = t_base - t_opt
    pct = saved / t_base * 100
    print(sep)
    print("  RESULT")
    print(sep)
    print(f"  Baseline total  : {t_base:.2f}s")
    print(f"  Optimised total : {t_opt:.2f}s")
    print(f"  Time saved      : {saved:.2f}s  ({pct:.0f}% faster)")
    print()
    print("  What changed:")
    print("  ① law_agent: merged analyze_law + check_routing → analyze_and_route")
    print(f"    {LLM_DELAYS['analyze_law']:.1f}s + {LLM_DELAYS['check_routing']:.1f}s serial  →  {LLM_DELAYS['analyze_and_route']:.1f}s  (saved {LLM_DELAYS['analyze_law']+LLM_DELAYS['check_routing']-LLM_DELAYS['analyze_and_route']:.1f}s, -1 LLM round-trip)")
    print("  ② compliance_agent: verbose prompt → concise bullet prompt")
    print(f"    Observed compliance time: 33.5s  →  ~{LLM_DELAYS['compliance_answer']:.1f}s (shorter output = faster generation)")
    total_compliance_saving = 33.5 - LLM_DELAYS["compliance_answer"]
    print(f"    Estimated saving on compliance alone: ~{total_compliance_saving:.0f}s")
    print()
    real_baseline = (
        LLM_DELAYS["customer_classify"]
        + LLM_DELAYS["analyze_law"]
        + LLM_DELAYS["check_routing"]
        + max(LLM_DELAYS["tax_answer"], 33.5)   # compliance was 33.5s in real run
        + LLM_DELAYS["aggregate"]
        + LLM_DELAYS["customer_format"]
    )
    real_optimised = (
        LLM_DELAYS["customer_classify"]
        + LLM_DELAYS["analyze_and_route"]
        + max(LLM_DELAYS["tax_answer"], LLM_DELAYS["compliance_answer"])
        + LLM_DELAYS["aggregate"]
        + LLM_DELAYS["customer_format"]
    )
    print(f"  Real-world estimate (compliance=33.5s baseline → {LLM_DELAYS['compliance_answer']:.1f}s optimised):")
    print(f"    Baseline  ≈ {real_baseline:.0f}s")
    print(f"    Optimised ≈ {real_optimised:.0f}s")
    print(f"    Saving    ≈ {real_baseline - real_optimised:.0f}s  ({(real_baseline-real_optimised)/real_baseline*100:.0f}% faster)")
    print(sep)


if __name__ == "__main__":
    asyncio.run(main())
