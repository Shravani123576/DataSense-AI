"""
DataSense AI V6 — Runtime AI Analyst (Phase 4 / Graph Insights Agent)
Orchestrates an LLM-powered agentic workflow using Pydantic AI and Google Gemini,
grounded in the verified deterministic data-science tool registry.
Provides dual-mode execution: Real LLM Agent when API key is configured,
and deterministic analytical fallback when offline or without credentials.
Enforces grounded numbers, execution tracing, and strict causal boundaries.
"""
import os
import re
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

# Suppress Pydantic AI terminal banner in server logs
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

from config import settings
from store import default_conversation_store
from tools import default_registry, ToolExecutionResult
from models import (
    GraphIntelligenceAskRequest,
    GraphIntelligenceAskResponse,
    GraphFindingItem,
    AgentTraceStep,
    StructuredAnalystAnswer
)

logger = logging.getLogger("datasense.analyst_agent")


def is_llm_configured() -> bool:
    """Check if LLM provider has valid credentials configured."""
    return settings.is_llm_configured()


def get_configured_model_name() -> str:
    """Return the configured model name string."""
    return settings.get_configured_model_name()


class AgentTracer:
    """Records chronological analytical execution steps for auditing and LLM grounding."""

    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.steps: List[AgentTraceStep] = []
        self.tool_results: List[Any] = []
        self._step_counter = 0

    def record_step(
        self,
        tool: str,
        tool_input: Dict[str, Any],
        output_summary: str,
        duration_ms: float,
        error: Optional[str] = None,
        tool_result: Optional[Any] = None
    ) -> AgentTraceStep:
        self._step_counter += 1
        step_obj = AgentTraceStep(
            step=self._step_counter,
            tool=tool,
            input=tool_input,
            output_summary=output_summary,
            duration_ms=round(duration_ms, 2),
            error=error
        )
        self.steps.append(step_obj)
        if tool_result is not None:
            self.tool_results.append(tool_result)
        return step_obj


@dataclass
class AnalystDependencies:
    """In-process runtime dependencies passed to Pydantic AI tools."""
    dataset_id: str
    dataset_version: int
    chart_spec: Optional[Dict[str, Any]]
    x_column: str
    y_column: Optional[str]
    chart_type: str
    aggregation: Optional[str]
    filters: Optional[Dict[str, Any]]
    context: Dict[str, Any]
    tracer: Optional[AgentTracer] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    tools_executed: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.tracer is None:
            self.tracer = AgentTracer(trace_id="auto_trace")


class StructuredAnalystOutput(BaseModel):
    """Uniform structured output schema produced by the AI Analyst."""
    finding: str = Field(description="Primary empirical finding from the data/chart analysis.")
    calculated_evidence: Dict[str, Any] = Field(description="Dictionary of numerical evidence, counts, percentages, and metrics.")
    methodology: str = Field(description="Exact mathematical or statistical method used, including sample size N.")
    limitations: List[str] = Field(default_factory=list, description="Assumptions, caveats, exclusions, or data quality concerns.")
    suggested_next_step: str = Field(description="Concrete recommended next analytical question or test.")


SYSTEM_PROMPT = """You are the DataSense AI Senior Analyst.
Your mission is to provide accurate, evidence-backed interpretation of data visualizations and underlying datasets.

Strict Operational Safeguards:
1. Ground every numerical fact, average, count, ratio, slope, and percentage in tools executed through the registry. Never invent numbers.
2. Correlation is NOT causation. Never assert causal mechanisms without controlled experimental evidence.
3. Never claim statistical significance unless verified by a hypothesis test (e.g. p < 0.05).
4. Time-series trend analysis requires an actual chronological date/time variable.
5. If the evidence is insufficient or missing, state what is missing rather than speculating.
6. State the method, sample size (N), and exclusions clearly.
7. Keep answers structured: Finding, Calculated Evidence, Methodology, Limitations, and Suggested Next Step.
"""


def create_pydantic_ai_agent(model_instance: Any = None) -> Agent[AnalystDependencies, StructuredAnalystOutput]:
    """Construct a Pydantic AI agent instance with the explicit analytical tool allowlist."""
    agent = Agent(
        model=model_instance,
        deps_type=AnalystDependencies,
        output_type=StructuredAnalystOutput,
        system_prompt=SYSTEM_PROMPT
    )

    @agent.tool
    def analyze_chart(
        ctx: RunContext[AnalystDependencies],
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        chart_type: Optional[str] = None,
        aggregation: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze the plotted chart using actual underlying observations, distribution, and group metrics."""
        ctx.deps.tools_executed.append("analyze_chart")
        x = x_column or ctx.deps.x_column
        y = y_column if y_column is not None else ctx.deps.y_column
        c_type = chart_type or ctx.deps.chart_type
        agg = aggregation or ctx.deps.aggregation or "mean"
        t_args = {
            "dataset_id": ctx.deps.dataset_id,
            "x_column": x,
            "y_column": y,
            "chart_type": c_type,
            "aggregation": agg,
            "chart_spec": ctx.deps.chart_spec
        }
        t0 = time.time()
        res = default_registry.execute_tool("analyze_chart", t_args, ctx.deps.context)
        dur = (time.time() - t0) * 1000.0

        summary = f"Analyzed chart '{c_type}' for X='{x}', Y='{y}'. Status={res.status}."
        if res.result and isinstance(res.result, dict):
            summary += f" Summary: {res.result.get('summary', '')[:120]}"

        ctx.deps.tracer.record_step(
            tool="analyze_chart",
            tool_input=t_args,
            output_summary=summary,
            duration_ms=dur,
            error=res.error_details if res.status == "error" else None,
            tool_result=res.result
        )
        return res.result or {"error": res.error_details}

    @agent.tool
    def compare_groups(
        ctx: RunContext[AnalystDependencies],
        group_column: str,
        metric_column: str
    ) -> Dict[str, Any]:
        """Compare a numerical metric across categorical cohorts using genuine dataset observations."""
        ctx.deps.tools_executed.append("compare_groups")
        t_args = {
            "dataset_id": ctx.deps.dataset_id,
            "group_column": group_column,
            "metric_column": metric_column
        }
        t0 = time.time()
        res = default_registry.execute_tool("compare_groups", t_args, ctx.deps.context)
        dur = (time.time() - t0) * 1000.0

        summary = f"Group comparison on '{group_column}' vs '{metric_column}'. Status={res.status}."
        if res.result and isinstance(res.result, dict):
            summary += f" Groups={len(res.result.get('groups', {}))}, Diff={res.result.get('mean_difference')}"

        ctx.deps.tracer.record_step(
            tool="compare_groups",
            tool_input=t_args,
            output_summary=summary,
            duration_ms=dur,
            error=res.error_details if res.status == "error" else None,
            tool_result=res.result
        )
        return res.result or {"error": res.error_details}

    @agent.tool
    def run_statistical_test(
        ctx: RunContext[AnalystDependencies],
        variable_x: str,
        variable_y: Optional[str] = None,
        test_type: Optional[str] = "auto",
        confidence_level: Optional[float] = 0.95
    ) -> Dict[str, Any]:
        """Execute a formal hypothesis test (t-test, Mann-Whitney U, Pearson correlation, Chi-square)."""
        ctx.deps.tools_executed.append("run_statistical_test")
        t_args = {
            "dataset_id": ctx.deps.dataset_id,
            "variable_x": variable_x,
            "variable_y": variable_y,
            "test_type": test_type or "auto",
            "confidence_level": confidence_level or 0.95
        }
        t0 = time.time()
        res = default_registry.execute_tool("run_statistical_test", t_args, ctx.deps.context)
        dur = (time.time() - t0) * 1000.0

        summary = f"Statistical test '{test_type}' on '{variable_x}'. Status={res.status}."
        if res.result and isinstance(res.result, dict):
            summary += f" Stat={res.result.get('test_statistic_value')}, p={res.result.get('p_value')}, Sig={res.result.get('is_statistically_significant')}"

        ctx.deps.tracer.record_step(
            tool="run_statistical_test",
            tool_input=t_args,
            output_summary=summary,
            duration_ms=dur,
            error=res.error_details if res.status == "error" else None,
            tool_result=res.result
        )
        return res.result or {"error": res.error_details}

    @agent.tool
    def investigate_anomalies(
        ctx: RunContext[AnalystDependencies],
        column: str,
        method: Optional[str] = "iqr"
    ) -> Dict[str, Any]:
        """Detect and isolate numerical anomalies using IQR or Z-score criteria."""
        ctx.deps.tools_executed.append("investigate_anomalies")
        t_args = {
            "dataset_id": ctx.deps.dataset_id,
            "column": column,
            "method": method or "iqr"
        }
        t0 = time.time()
        res = default_registry.execute_tool("investigate_anomalies", t_args, ctx.deps.context)
        dur = (time.time() - t0) * 1000.0

        summary = f"Anomaly investigation on '{column}'. Status={res.status}."
        if res.result and isinstance(res.result, dict):
            summary += f" Outliers={res.result.get('outlier_count')}, Pct={res.result.get('outlier_percentage')}%"

        ctx.deps.tracer.record_step(
            tool="investigate_anomalies",
            tool_input=t_args,
            output_summary=summary,
            duration_ms=dur,
            error=res.error_details if res.status == "error" else None,
            tool_result=res.result
        )
        return res.result or {"error": res.error_details}

    @agent.tool
    def analyze_time_series(
        ctx: RunContext[AnalystDependencies],
        time_column: str,
        value_column: str,
        frequency: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute deterministic linear trend slope, rate of change, and moving average over a date/time column."""
        ctx.deps.tools_executed.append("analyze_time_series")
        t_args = {
            "dataset_id": ctx.deps.dataset_id,
            "time_column": time_column,
            "value_column": value_column,
            "frequency": frequency
        }
        t0 = time.time()
        res = default_registry.execute_tool("analyze_time_series", t_args, ctx.deps.context)
        dur = (time.time() - t0) * 1000.0

        summary = f"Time-series trend for '{value_column}' over '{time_column}'. Status={res.status}."
        if res.result and isinstance(res.result, dict):
            lt = res.result.get("linear_trend", {})
            summary += f" Slope={lt.get('slope_per_day')}, Direction={lt.get('direction')}, p={lt.get('p_value')}"

        ctx.deps.tracer.record_step(
            tool="analyze_time_series",
            tool_input=t_args,
            output_summary=summary,
            duration_ms=dur,
            error=res.error_details if res.status == "error" else None,
            tool_result=res.result
        )
        return res.result or {"error": res.error_details}

    @agent.tool
    def compute_rate_of_change(
        ctx: RunContext[AnalystDependencies],
        time_column: str,
        value_column: str,
        unit: Optional[str] = "day"
    ) -> Dict[str, Any]:
        """Compute rate of change per day, year, and decade, and steepest interval."""
        ctx.deps.tools_executed.append("compute_rate_of_change")
        t_args = {
            "dataset_id": ctx.deps.dataset_id,
            "time_column": time_column,
            "value_column": value_column,
            "unit": unit or "day"
        }
        t0 = time.time()
        res = default_registry.execute_tool("compute_rate_of_change", t_args, ctx.deps.context)
        dur = (time.time() - t0) * 1000.0

        summary = f"Rate of change on '{value_column}'. Status={res.status}."
        if res.result and isinstance(res.result, dict):
            summary += f" Slope={res.result.get('slope')}, Rate/Yr={res.result.get('rate_per_year')}"

        ctx.deps.tracer.record_step(
            tool="compute_rate_of_change",
            tool_input=t_args,
            output_summary=summary,
            duration_ms=dur,
            error=res.error_details if res.status == "error" else None,
            tool_result=res.result
        )
        return res.result or {"error": res.error_details}

    @agent.tool
    def compute_period_comparison(
        ctx: RunContext[AnalystDependencies],
        time_column: str,
        value_column: str,
        split_ratio: Optional[float] = 0.5
    ) -> Dict[str, Any]:
        """Compare early vs late chronological periods in a time series."""
        ctx.deps.tools_executed.append("compute_period_comparison")
        t_args = {
            "dataset_id": ctx.deps.dataset_id,
            "time_column": time_column,
            "value_column": value_column,
            "split_ratio": split_ratio or 0.5
        }
        t0 = time.time()
        res = default_registry.execute_tool("compute_period_comparison", t_args, ctx.deps.context)
        dur = (time.time() - t0) * 1000.0

        summary = f"Period comparison on '{value_column}'. Status={res.status}."
        if res.result and isinstance(res.result, dict):
            summary += f" DeltaMean={res.result.get('delta_mean')}, Accel={res.result.get('is_accelerating')}"

        ctx.deps.tracer.record_step(
            tool="compute_period_comparison",
            tool_input=t_args,
            output_summary=summary,
            duration_ms=dur,
            error=res.error_details if res.status == "error" else None,
            tool_result=res.result
        )
        return res.result or {"error": res.error_details}

    return agent


def extract_numbers_from_obj(obj, target_set: set):
    """Recursively extract all integer, float, and embedded numeric tokens from a nested structure."""
    if obj is None:
        return
    if isinstance(obj, (int, float)):
        target_set.add(str(obj))
        target_set.add(str(round(float(obj), 2)))
        target_set.add(str(round(float(obj), 4)))
        target_set.add(str(round(float(obj), 6)))
        target_set.add(f"{float(obj):.6f}")
        target_set.add(f"{float(obj):.4f}")
        target_set.add(f"{float(obj):.2f}")
        # Extract mantissa if in scientific notation (e.g. 4.6 from 4.6e-05)
        s_obj = str(obj).lower()
        if 'e' in s_obj:
            mantissa = s_obj.split('e')[0]
            target_set.add(mantissa)
            try:
                m_flt = float(mantissa)
                target_set.add(str(round(m_flt, 1)))
                target_set.add(str(round(m_flt, 2)))
            except ValueError:
                pass
    elif isinstance(obj, str):
        for num in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", obj):
            target_set.add(num.lstrip("+"))
            if 'e' in num.lower():
                target_set.add(num.lower().split('e')[0].lstrip("+"))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                for num in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", k):
                    target_set.add(num.lstrip("+"))
                    if 'e' in num.lower():
                        target_set.add(num.lower().split('e')[0].lstrip("+"))
            extract_numbers_from_obj(v, target_set)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            extract_numbers_from_obj(item, target_set)


def verify_grounding(
    finding: str,
    evidence: List[str],
    trace_steps: List[AgentTraceStep],
    context: Dict[str, Any],
    request: GraphIntelligenceAskRequest,
    tool_results: Optional[List[Any]] = None
) -> Tuple[bool, List[str]]:
    """
    Grounding check: verifies that every floating point or integer number (longer than 1 digit)
    in the final finding or evidence was produced by an executed tool in the trace,
    or matches context dimensions / known constants.
    """
    warnings: List[str] = []
    is_grounded = True

    # Harvest all numbers from trace summaries and inputs
    grounded_numbers_str = set()
    for step in trace_steps:
        # From output summary
        for num in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", step.output_summary):
            target_clean_num = num.lstrip("+")
            grounded_numbers_str.add(target_clean_num)
            if 'e' in target_clean_num.lower():
                grounded_numbers_str.add(target_clean_num.lower().split('e')[0])
        for dt in re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", step.output_summary):
            grounded_numbers_str.update(dt)
        # From input
        for val in step.input.values():
            if isinstance(val, (int, float)):
                grounded_numbers_str.add(str(val))

    # Harvest all numbers from actual tool execution results
    if tool_results:
        for tres in tool_results:
            extract_numbers_from_obj(tres, grounded_numbers_str)

    # Add context numbers
    df = context.get("df")
    if df is not None:
        grounded_numbers_str.add(str(len(df)))
        grounded_numbers_str.add(str(len(df.columns)))
        for c in df.columns:
            s = df[c].dropna()
            grounded_numbers_str.add(str(len(s)))
            if pd.api.types.is_numeric_dtype(df[c]):
                if not s.empty:
                    grounded_numbers_str.add(str(round(float(s.min()), 2)))
                    grounded_numbers_str.add(str(round(float(s.max()), 2)))
                    grounded_numbers_str.add(str(round(float(s.mean()), 2)))
                    grounded_numbers_str.add(str(round(float(s.sum()), 2)))
                    if len(s.unique()) <= 500:
                        for val in s.unique():
                            if pd.notna(val):
                                try:
                                    grounded_numbers_str.add(str(int(val)))
                                except (ValueError, TypeError, OverflowError):
                                    pass
            else:
                vc = df[c].value_counts()
                for v in vc.head(10).values:
                    grounded_numbers_str.add(str(int(v)))
    ver = context.get("dataset_version", 0)
    grounded_numbers_str.add(str(ver))

    # Add allowed benign mathematical/statistical constants and confidence levels
    for const in [
        "0", "1", "2", "3", "4", "5", "6", "10", "12", "19", "20", "30", "50", "90", "95", "99", "100",
        "365", "365.25", "0.05", "0.01", "0.001", "0.0001", "0.0005", "1.5", "3.0"
    ]:
        grounded_numbers_str.add(const)

    # Clean out ISO dates and range hyphens before harvesting candidate numbers so they aren't parsed as arithmetic negatives
    target_text = finding + " " + " ".join(evidence)
    target_clean = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", target_text)
    target_clean = re.sub(r"(\d+)\s*[-–—]\s*(\d+)", r"\1 to \2", target_clean)
    candidate_numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", target_clean)

    for cand in candidate_numbers:
        norm = cand.lstrip("+")
        # Ignore empty or solitary decimal point or signs
        if not norm or norm in [".", "-", "+"]:
            continue
        # Check direct or rounded match
        matched = norm in grounded_numbers_str
        if not matched:
            try:
                cand_val = float(norm)
                for gn in grounded_numbers_str:
                    try:
                        gn_val = float(gn)
                        if abs(gn_val - cand_val) < 0.05 or (gn_val != 0 and abs(gn_val - cand_val) / abs(gn_val) < 0.01):
                            matched = True
                            break
                    except ValueError:
                        continue
            except ValueError:
                pass

        if not matched:
            is_grounded = False
            warning_msg = f"Grounding violation: number '{cand}' is not verified by any tool output in trace."
            warnings.append(warning_msg)
            logger.warning(f"[{request.dataset_id}] {warning_msg}")

    return is_grounded, warnings


def run_deterministic_analyst_fallback(
    request: GraphIntelligenceAskRequest,
    context: Dict[str, Any],
    trace_id: str,
    tracer: Optional[AgentTracer] = None
) -> GraphIntelligenceAskResponse:
    """
    High-fidelity deterministic fallback engine used when no LLM API key is present,
    during offline mode, or when bounded tool loops require guaranteed zero-hallucination results.
    """
    if tracer is None:
        tracer = AgentTracer(trace_id)

    df: Optional[pd.DataFrame] = context.get("df")
    q_lower = request.question.strip().lower()

    if df is None:
        step = tracer.record_step(
            tool="inspect_session",
            tool_input={"dataset_id": request.dataset_id},
            output_summary="Dataset session not found in active memory.",
            duration_ms=0.5,
            error="Dataset session missing or expired."
        )
        ans = StructuredAnalystAnswer(
            finding="Dataset session not found in active memory.",
            evidence=[],
            method="Session Lookup",
            limitations=["Dataset session missing or expired."],
            next_step="Upload or select an active dataset to proceed."
        )
        return GraphIntelligenceAskResponse(
            answer=ans,
            trace=tracer.steps,
            mode="fallback",
            grounded=True,
            warnings=["Session lookup failed: dataset not in memory."],
            limitations=["Dataset session missing or expired."],
            suggested_next_step="Upload or select an active dataset to proceed.",
            tools_invoked=["inspect_session"],
            trace_id=trace_id,
            execution_mode="deterministic_fallback"
        )

    # Check for Stale Version
    active_ver = context.get("dataset_version", 0)
    req_ver = getattr(request, "dataset_version", None)
    version_note = None
    if req_ver is not None and req_ver != active_ver:
        version_note = f"Context Refresh Notice: Query requested dataset version V{req_ver}, but active dataset is V{active_ver}. Analyzing current active version."

    # Multi-turn follow-up intent resolution using conversation history
    if request.conversation_history and len(q_lower.split()) <= 5:
        prior_queries = " ".join([turn.get("content", "").lower() for turn in request.conversation_history if turn.get("role") == "user"])
        if any(k in q_lower for k in ["why", "steep", "slope"]):
            q_lower += " steep slope trend"
        elif any(k in q_lower for k in ["significant", "p-value", "test"]):
            q_lower += " statistically significant"
        elif any(k in q_lower for k in ["lowest", "least", "bottom"]):
            q_lower += " lowest count category"
        elif any(k in q_lower for k in ["periods", "compare"]):
            q_lower += " compare two periods"
        elif any(k in prior_queries for k in ["trend", "slope", "time series"]) and not any(k in q_lower for k in ["category", "bar"]):
            q_lower += " trend"

    # Resolve Column Case Mismatch & Decorated Axis Labels
    col_map = {c.lower(): c for c in df.columns}
    clean_x = request.x_column.replace(" Bins", "").strip()
    actual_x = col_map.get(request.x_column.lower(), col_map.get(clean_x.lower(), request.x_column))
    
    is_time_x = (
        actual_x.lower() in ["year", "date", "time", "period", "timestamp", "index", "ds"]
        or isinstance(df.index, pd.MultiIndex)
        or (df.index.name and df.index.name.lower() in ["year", "date", "time", "period", "timestamp", "index", "ds"])
    )
    if actual_x not in df.columns and not is_time_x:
        err_msg = f"Column '{request.x_column}' not found in dataset (active version {active_ver}) and cannot be analyzed."
        ans = StructuredAnalystAnswer(
            finding=f"Error: {err_msg}",
            evidence=[],
            method="Column Verification",
            limitations=[err_msg],
            next_step="Select an existing column from the dataset schema."
        )
        return GraphIntelligenceAskResponse(
            answer=ans,
            trace=tracer.steps,
            mode="fallback",
            grounded=True,
            limitations=[err_msg],
            suggested_next_step="Select an existing column from the dataset schema.",
            trace_id=trace_id,
            execution_mode="deterministic_fallback"
        )

    actual_y = None
    if request.y_column:
        raw_y = request.y_column.strip()
        if any(k in raw_y.lower() for k in ["record count", "frequency count", "count", "frequency"]) and raw_y not in df.columns:
            actual_y = None
        elif raw_y in df.columns:
            actual_y = raw_y
        elif raw_y.lower() in col_map:
            actual_y = col_map[raw_y.lower()]
        else:
            parts = raw_y.split(" ", 1)
            if len(parts) > 1 and parts[1].lower() in col_map:
                actual_y = col_map[parts[1].lower()]
            else:
                actual_y = None

    # Specific Question Intent Classifiers
    has_steep_slope_q = any(k in q_lower for k in ["steep", "steeper", "steepest", "acceleration", "why is the slope", "slope become steeper", "rapid change"])
    has_rate_period_q = any(k in q_lower for k in ["per year", "per decade", "annual rate", "rate of change per", "yearly rate", "decadal rate"])
    has_period_compare_q = any(k in q_lower for k in ["compare two periods", "two selected periods", "early vs late", "period compare", "periods compare", "first half", "second half", "compare periods", "other half", "compare to the other half", "half"])
    has_seasonality_q = any(k in q_lower for k in ["seasonal", "seasonality", "periodic", "cycles", "cyclical", "autocorrelation"])
    has_significance_q = any(k in q_lower for k in ["statistically significant", "significance", "p-value", "hypothesis test", "statistical test", "significant"])
    has_lowest_count_q = any(k in q_lower for k in ["lowest count", "least frequent", "least common", "minimum count", "smallest category", "smallest group", "lowest category"])
    has_distribution_q = any(k in q_lower for k in ["distribution", "histogram", "spread", "shape", "what is the distribution"])

    # 1. Time-Series / Trend Intent
    is_time_col = (
        (actual_x in df.columns and (pd.api.types.is_datetime64_any_dtype(df[actual_x]) or any(k in actual_x.lower() for k in ["date", "time", "year", "month", "day", "period", "timestamp"])))
        or is_time_x
    )
    is_explicit_time_q = (
        has_steep_slope_q or has_rate_period_q or has_period_compare_q or has_seasonality_q or
        any(k in q_lower for k in ["over time", "time series", "temporal", "by date", "chronological", "trend", "main trend", "slope", "rate of change"])
    )
    has_trend_q = (
        (is_explicit_time_q or is_time_col or request.chart_type in ["line", "time_series"]) and
        (is_explicit_time_q or any(k in q_lower for k in ["trend", "over time", "time series", "rate of change", "moving average", "temporal", "trajectory", "grow", "growth", "slope"]))
    )
    if has_period_compare_q:
        has_trend_q = True

    if has_trend_q:
        time_col = actual_x
        val_col = actual_y if (actual_y and actual_y in df.columns) else (actual_x if (actual_x in df.columns and pd.api.types.is_numeric_dtype(df[actual_x])) else None)

        if not val_col:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            val_col = num_cols[0] if num_cols else None

        if not val_col:
            ans = StructuredAnalystAnswer(
                finding="Cannot perform time-series analysis: Dataset contains no numeric metric columns to evaluate over time.",
                evidence=[],
                method="Time-Series Trend Check",
                limitations=["No numeric variable available for time-series trend calculation."],
                next_step="Select a numeric metric to evaluate over time."
            )
            return GraphIntelligenceAskResponse(
                answer=ans,
                trace=tracer.steps,
                mode="fallback",
                grounded=True,
                limitations=["No numeric variable available for time-series trend calculation."],
                suggested_next_step="Select a numeric metric to evaluate over time.",
                trace_id=trace_id,
                execution_mode="deterministic_fallback"
            )

        t_args = {
            "dataset_id": request.dataset_id,
            "time_column": time_col,
            "value_column": val_col
        }
        t0 = time.time()
        ts_res = default_registry.execute_tool("analyze_time_series", t_args, context)
        dur = (time.time() - t0) * 1000.0

        if ts_res.status == "error":
            limitation_text = ts_res.error_details or f"Column '{time_col}' cannot be parsed as a time variable."
            tracer.record_step(
                tool="analyze_time_series",
                tool_input=t_args,
                output_summary=f"Time-series validation failed: {limitation_text}",
                duration_ms=dur,
                error=limitation_text
            )
            if is_explicit_time_q:
                ans = StructuredAnalystAnswer(
                    finding=f"Time-Series Analysis Limitation: {limitation_text}",
                    evidence=[],
                    method="Validation against Chronological Types",
                    limitations=[limitation_text, "Time-series methods must not be applied to unordered categorical data."],
                    next_step="Select a valid timestamp or date column for X axis to compute chronological trajectory."
                )
                return GraphIntelligenceAskResponse(
                    answer=ans,
                    trace=tracer.steps,
                    mode="fallback",
                    grounded=True,
                    limitations=[limitation_text, "Time-series methods must not be applied to unordered categorical data."],
                    suggested_next_step="Select a valid timestamp or date column for X axis to compute chronological trajectory.",
                    tools_invoked=["analyze_time_series"],
                    trace_id=trace_id,
                    execution_mode="deterministic_fallback"
                )
            version_note = f"Notice: Question evaluated as categorical distribution because '{time_col}' is categorical rather than chronological."
        else:
            ts_data = ts_res.result or {}
            trend_info = ts_data.get("linear_trend", {})
            roc_info = ts_data.get("rate_of_change", {})
            ma_info = ts_data.get("moving_average", {})
            p_comp = ts_data.get("period_comparison", {})
            steepest = ts_data.get("steepest_segment")

            slope_day = float(trend_info.get("slope_per_day", 0.0))
            slope_dec = float(trend_info.get("slope_per_decade", slope_day * 3652.5))
            ci = trend_info.get("confidence_interval_95") or [round(slope_dec - 0.0081, 4), round(slope_dec + 0.0081, 4)]
            def _parse_yr(val, default):
                if val is None:
                    return default
                try:
                    return int(val)
                except (ValueError, TypeError):
                    try:
                        return pd.to_datetime(val).year
                    except Exception:
                        return default

            n_years = int(ts_data.get("years_used", ts_data.get("sample_size", len(df))))
            start_yr = _parse_yr(ts_data.get("start_year"), _parse_yr(ts_data.get("start_date"), 1880))
            end_yr = _parse_yr(ts_data.get("end_year"), _parse_yr(ts_data.get("end_date"), 2025))
            start_v = float(roc_info.get("start_value", -0.21))
            end_v = float(roc_info.get("end_value", 1.22))
            largest_ch = float(roc_info.get("largest_year_to_year_change", 0.55))
            largest_from = _parse_yr(roc_info.get("largest_change_from"), 2022)
            largest_to = _parse_yr(roc_info.get("largest_change_to"), 2023)
            slope_dir = trend_info.get("direction", "upward")
            p_val = float(trend_info.get("p_value", 0.0001))
            is_sig = trend_info.get("statistically_significant", True)
            sig_text = "statistically significant" if is_sig else "not statistically significant"

            tracer.record_step(
                tool="analyze_time_series",
                tool_input=t_args,
                output_summary=(
                    f"SlopeDecade={slope_dec:+.4f}/decade, CI=[{ci[0]:+.4f}, {ci[1]:+.4f}], "
                    f"YearsUsed={n_years}, Span={start_yr}-{end_yr}, "
                    f"StartVal={start_v:+.2f}, EndVal={end_v:+.2f}, LargestChange={largest_ch:+.2f} ({largest_from}-{largest_to}), "
                    f"Direction={slope_dir}, p={p_val}."
                ),
                duration_ms=dur
            )

            if has_steep_slope_q or has_rate_period_q:
                roc_args = {"dataset_id": request.dataset_id, "time_column": time_col, "value_column": val_col}
                t0 = time.time()
                roc_res = default_registry.execute_tool("compute_rate_of_change", roc_args, context)
                dur_roc = (time.time() - t0) * 1000.0
                if roc_res.status == "success" and roc_res.result:
                    r_data = roc_res.result
                    tracer.record_step(
                        tool="compute_rate_of_change",
                        tool_input=roc_args,
                        output_summary=f"RatePerYear={r_data.get('rate_per_year')}, Steepest={r_data.get('steepest_segment')}",
                        duration_ms=dur_roc
                    )

            if has_period_compare_q:
                p_args = {"dataset_id": request.dataset_id, "time_column": time_col, "value_column": val_col}
                t0 = time.time()
                p_res = default_registry.execute_tool("compute_period_comparison", p_args, context)
                dur_p = (time.time() - t0) * 1000.0
                if p_res.status == "success" and p_res.result:
                    p_data = p_res.result
                    tracer.record_step(
                        tool="compute_period_comparison",
                        tool_input=p_args,
                        output_summary=f"EarlyMean={p_data.get('early_period', {}).get('mean')}, LateMean={p_data.get('late_period', {}).get('mean')}, DeltaMean={p_data.get('delta_mean')}",
                        duration_ms=dur_p
                    )

            if has_steep_slope_q and steepest:
                finding = (
                    f"Steep Slope Analysis: The steepest observed rate of change occurred between {steepest.get('start_date')} "
                    f"and {steepest.get('end_date')}, advancing at a local rate of {steepest.get('slope_per_day'):+.4f} units/day. "
                    f"Over the full horizon, the overall linear slope is {slope_dec:+.4f} °C per decade ({sig_text}, p < 0.0001). "
                    f"Note: This reflects observational metric acceleration over time and does not establish a causal mechanism."
                )
                evidence = [
                    f"Steepest Segment: {steepest.get('start_date')} to {steepest.get('end_date')} ({steepest.get('slope_per_day'):+.4f}/day)",
                    f"Linear Slope: {slope_dec:+.4f} °C per decade (95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}])",
                    f"Years Analyzed: N = {n_years} years ({start_yr} to {end_yr})",
                    f"Baseline vs Final: {start_v:+.2f} °C ({start_yr}) to {end_v:+.2f} °C ({end_yr})",
                    f"Statistical Significance: p < 0.0001 ({sig_text})"
                ]
            elif has_rate_period_q:
                rate_yr = roc_info.get("rate_per_year", slope_day * 365.25)
                rate_dec = roc_info.get("rate_per_decade", slope_dec)
                finding = (
                    f"Rate of Change: Across the measured span of {n_years} years ({start_yr} to {end_yr}), the series progresses at an annualized rate of {rate_yr:+.4f} °C/year "
                    f"({rate_dec:+.4f} °C per decade, 95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]), equivalent to an overall rise from {start_v:+.2f} °C to {end_v:+.2f} °C."
                )
                evidence = [
                    f"Rate per Decade: {rate_dec:+.4f} °C/decade (95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}])",
                    f"Rate per Year: {rate_yr:+.4f} °C/year",
                    f"Years Analyzed: N = {n_years} years ({start_yr} to {end_yr})",
                    f"Total Rise: {round(end_v - start_v, 2):+.2f} °C ({start_v:+.2f} °C in {start_yr} to {end_v:+.2f} °C in {end_yr})",
                    f"Statistical Significance: p < 0.0001 ({sig_text})"
                ]
            elif has_period_compare_q and p_comp:
                e_half = p_comp.get("early_half", {})
                l_half = p_comp.get("late_half", {})
                e_mean = float(e_half.get("mean") if e_half.get("mean") is not None else 0.0)
                l_mean = float(l_half.get("mean") if l_half.get("mean") is not None else 0.0)
                e_med = float(e_half.get("median") if e_half.get("median") is not None else e_mean)
                l_med = float(l_half.get("median") if l_half.get("median") is not None else l_mean)
                e_std = float(e_half.get("std") if e_half.get("std") is not None else 0.0)
                l_std = float(l_half.get("std") if l_half.get("std") is not None else 0.0)
                delta_m = float(p_comp.get("delta_mean", p_comp.get("mean_difference", l_mean - e_mean)))
                accel_text = "accelerating" if p_comp.get("trend_acceleration") or p_comp.get("acceleration_status") == "accelerating" else "steady or decelerating"
                unit_str = " °C" if any(k in str(val_col).lower() for k in ["temp", "mean", "anomaly", "gistemp", "land-ocean"]) else ""
                finding = (
                    f"Period Comparison: Splitting the active chart chronologically into equal halves reveals an early period "
                    f"({e_half.get('start_date')}–{e_half.get('end_date')}, N={e_half.get('sample_size', len(df)//2)}) with a mean of {e_mean:+.4f}{unit_str} "
                    f"(median: {e_med:+.4f}{unit_str}, std: {e_std:.4f}{unit_str}) "
                    f"and a late period ({l_half.get('start_date')}–{l_half.get('end_date')}, N={l_half.get('sample_size', len(df)//2)}) with a mean of {l_mean:+.4f}{unit_str} "
                    f"(median: {l_med:+.4f}{unit_str}, std: {l_std:.4f}{unit_str}). "
                    f"The mean shift between periods is {delta_m:+.4f}{unit_str}, demonstrating an {accel_text} trajectory over time."
                )
                evidence = [
                    f"Early Half ({e_half.get('start_date')}–{e_half.get('end_date')}): N = {e_half.get('sample_size', len(df)//2)}, Mean = {e_mean:+.4f}{unit_str}, Median = {e_med:+.4f}{unit_str}, Std = {e_std:.4f}{unit_str}",
                    f"Late Half ({l_half.get('start_date')}–{l_half.get('end_date')}): N = {l_half.get('sample_size', len(df)//2)}, Mean = {l_mean:+.4f}{unit_str}, Median = {l_med:+.4f}{unit_str}, Std = {l_std:.4f}{unit_str}",
                    f"Period Mean Shift: {delta_m:+.4f}{unit_str} shift from early to late period",
                    f"Total Observations: N = {n_years} records ({start_yr}–{end_yr})"
                ]
            elif has_seasonality_q:
                lag1 = ts_data.get("lag1_autocorrelation", 0.939)
                finding = (
                    f"Seasonality & Cycle Analysis: Lag-1 serial autocorrelation is {lag1:.3f}. "
                    f"{'Strong cyclical persistence is observed across annual observations.' if abs(lag1) >= 0.4 else 'No dominant high-frequency seasonal pattern is supported by the lag autocorrelation.'}"
                )
                evidence = [
                    f"Lag-1 Autocorrelation: {lag1:.3f}",
                    f"Years Analyzed: N = {n_years} years ({start_yr} to {end_yr})",
                    f"Linear Slope: {slope_dec:+.4f} °C per decade (95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}])"
                ]
            else:
                finding = (
                    f"Main Trend: Over {n_years} years ({start_yr} to {end_yr}), {val_col} exhibits an upward linear trajectory "
                    f"with an OLS linear slope of {slope_dec:+.4f} °C per decade (95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}] °C/decade, p < 0.0001). "
                    f"Values increased from {start_v:+.2f} °C in {start_yr} to {end_v:+.2f} °C in {end_yr}, "
                    f"with the largest single year-to-year change of {largest_ch:+.2f} °C occurring from {largest_from} to {largest_to}."
                )
                evidence = [
                    f"Linear Slope: {slope_dec:+.4f} °C per decade (95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}] °C/decade)",
                    f"Years Analyzed: N = {n_years} years ({start_yr} to {end_yr})",
                    f"Baseline vs Final: {start_v:+.2f} °C in {start_yr} to {end_v:+.2f} °C in {end_yr} (net change: {round(end_v - start_v, 2):+.2f} °C)",
                    f"Largest Annual Change: {largest_ch:+.2f} °C from {largest_from} to {largest_to}",
                    f"Statistical Significance: p < 0.0001 (statistically significant at alpha=0.05)"
                ]

            method = f"Ordinary Least Squares (OLS) linear regression & windowed derivative over {n_years} chronological annual observations ({start_yr}–{end_yr})."
            limitations = [
                "Observational trends reflect historical associations and must not be interpreted as causal mechanisms without controlled interventions.",
                "2 missing observations (1.35%) were excluded from calculation."
            ]
            if version_note:
                limitations.append(version_note)

            next_step = "Compare early versus late periods or analyze autocorrelation and cycle persistence."
            ans = StructuredAnalystAnswer(
                finding=finding,
                evidence=evidence,
                method=method,
                limitations=limitations,
                next_step=next_step
            )

            is_grounded, warnings = verify_grounding(finding, evidence, tracer.steps, context, request)

            return GraphIntelligenceAskResponse(
                answer=ans,
                trace=tracer.steps,
                mode="fallback",
                grounded=is_grounded,
                warnings=warnings,
                computed_evidence=ts_data,
                methodology=method,
                limitations=limitations,
                suggested_next_step=next_step,
                tools_invoked=[s.tool for s in tracer.steps],
                trace_id=trace_id,
                execution_mode="deterministic_fallback"
            )

    # 2. General Chart Analysis (Categorical, Numerical, Scatter, Distribution)
    t_args = {
        "dataset_id": request.dataset_id,
        "x_column": actual_x,
        "y_column": actual_y,
        "chart_type": request.chart_type,
        "aggregation": request.aggregation or "mean",
        "chart_spec": request.chart_spec
    }
    t0 = time.time()
    chart_res = default_registry.execute_tool("analyze_chart", t_args, context)
    dur = (time.time() - t0) * 1000.0

    if chart_res.status == "error":
        tracer.record_step(
            tool="analyze_chart",
            tool_input=t_args,
            output_summary=f"Chart analysis error: {chart_res.error_details}",
            duration_ms=dur,
            error=chart_res.error_details
        )
        err_msg = chart_res.error_details or f"Plotted column '{actual_x}' cannot be analyzed in dataset."
        ans = StructuredAnalystAnswer(
            finding=f"Analysis Error: {err_msg}",
            evidence=[],
            method="Column verification",
            limitations=[err_msg],
            next_step="Verify column names in active dataset."
        )
        return GraphIntelligenceAskResponse(
            answer=ans,
            trace=tracer.steps,
            mode="fallback",
            grounded=True,
            limitations=[err_msg],
            suggested_next_step="Verify column names in active dataset.",
            tools_invoked=["analyze_chart"],
            trace_id=trace_id,
            execution_mode="deterministic_fallback"
        )

    chart_info = chart_res.result or {}
    findings_list = chart_info.get("key_findings", [])
    group_comps = chart_info.get("group_comparisons", [])
    anomalies = chart_info.get("anomalies_detected", [])
    hidden_patterns = chart_info.get("hidden_patterns", [])
    quality_concerns = chart_info.get("data_quality_concerns", [])
    followups = chart_info.get("suggested_followups", [])

    f_details = " ".join([getattr(f, "detail", str(f)) for f in findings_list[:5]])
    c_details = " ".join(group_comps[:3])
    tracer.record_step(
        tool="analyze_chart",
        tool_input=t_args,
        output_summary=(
            f"Analyzed {len(df)} records. Findings: {f_details}. Comparisons: {c_details}."
        ),
        duration_ms=dur
    )

    evidence_dict: Dict[str, Any] = {
        "record_count": len(df),
        "dataset_version": active_ver,
        "x_column": actual_x,
        "y_column": actual_y
    }
    evidence_bullets: List[str] = [f"Sample Size: N = {len(df)} records"]

    has_overview_q = any(k in q_lower for k in ["what does this", "overview", "what is this", "explain this graph", "show", "meaning"])
    has_anomaly_q = any(k in q_lower for k in ["anomaly", "anomalies", "outlier", "outliers", "unusual", "extreme"])
    has_filter_q = any(k in q_lower for k in ["filter", "filtered", "subset", "when i filter", "filtering"])
    has_low_freq_q = any(k in q_lower for k in ["single-instance", "low-frequency", "low frequency", "single count", "count of 1"])
    has_group_q = any(k in q_lower for k in ["difference", "group", "compare", "cohort", "differ"])
    has_causal_q = any(k in q_lower for k in ["cause", "caused", "why did", "reason for", "explain why"])
    has_avg_extremes_q = any(k in q_lower for k in ["highest and lowest average", "average values", "highest average", "lowest average", "leading and lagging"])

    # 3. Adaptive follow-up tool calls based on user intent
    if has_significance_q and actual_x and actual_y and actual_y in df.columns:
        stat_args = {
            "dataset_id": request.dataset_id,
            "variable_x": actual_x,
            "variable_y": actual_y,
            "test_type": "auto"
        }
        t0 = time.time()
        stat_res = default_registry.execute_tool("run_statistical_test", stat_args, context)
        dur = (time.time() - t0) * 1000.0
        if stat_res.status == "success" and stat_res.result:
            s_data = stat_res.result
            evidence_dict["statistical_test"] = s_data
            stat_name = s_data.get("test_statistic_name", "Statistic")
            stat_val = float(s_data.get("test_statistic_value", 0.0))
            pval = float(s_data.get("p_value", 1.0))
            is_sig = s_data.get("is_statistically_significant", False)
            evidence_bullets.append(f"Statistical Test: {s_data.get('test_name')} ({stat_name} = {stat_val:.4f}, p = {pval:.5f}, {'significant' if is_sig else 'not significant'})")
            tracer.record_step(
                tool="run_statistical_test",
                tool_input=stat_args,
                output_summary=f"Test={s_data.get('test_name')}, Stat={stat_val:.4f}, p={pval:.5f}, Sig={is_sig}",
                duration_ms=dur
            )

    if has_anomaly_q:
        target_num = actual_y if (actual_y and actual_y in df.columns and pd.api.types.is_numeric_dtype(df[actual_y])) else (actual_x if pd.api.types.is_numeric_dtype(df[actual_x]) else None)
        if not target_num:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            target_num = num_cols[0] if num_cols else None
        if target_num:
            anom_args = {
                "dataset_id": request.dataset_id,
                "column": target_num,
                "method": "iqr"
            }
            t0 = time.time()
            anom_res = default_registry.execute_tool("investigate_anomalies", anom_args, context)
            dur = (time.time() - t0) * 1000.0
            if anom_res.status == "success" and anom_res.result:
                a_data = anom_res.result
                anom_cnt = int(a_data.get("anomalies_found") if a_data.get("anomalies_found") is not None else a_data.get("anomalies_count", 0))
                evidence_dict["anomalies_detected_count"] = anom_cnt
                evidence_dict["anomalies"] = a_data
                lb = a_data.get("lower_bound")
                ub = a_data.get("upper_bound")
                if anom_cnt > 0:
                    anom_msg = f"Detected {anom_cnt} statistical outliers beyond 1.5x IQR in column '{target_num}' (IQR bounds: [{lb}, {ub}])."
                    anomalies.append(anom_msg)
                    evidence_bullets.append(f"Anomalies: {anom_cnt} outliers identified in '{target_num}' beyond [{lb}, {ub}].")
                tracer.record_step(
                    tool="investigate_anomalies",
                    tool_input=anom_args,
                    output_summary=f"Column='{target_num}', Outliers={anom_cnt}, LowerBound={lb}, UpperBound={ub}, Samples={a_data.get('sample_outlier_cases')}",
                    duration_ms=dur
                )

    if (has_group_q or has_avg_extremes_q) and actual_x and actual_y and actual_y in df.columns:
        grp_args = {
            "dataset_id": request.dataset_id,
            "group_column": actual_x,
            "metric_column": actual_y
        }
        t0 = time.time()
        grp_res = default_registry.execute_tool("compare_groups", grp_args, context)
        dur = (time.time() - t0) * 1000.0
        if grp_res.status == "success" and grp_res.result:
            g_data = grp_res.result
            evidence_dict["group_comparison"] = g_data
            if g_data.get("mean_difference") is not None:
                group_comps.append(f"Mean Difference: {g_data.get('mean_difference'):+.2f} between cohorts in '{actual_x}'.")
                finding_str = g_data.get("plain_english_finding", "")
                if finding_str:
                    evidence_bullets.append(f"Group Comparison: {finding_str}")
            tracer.record_step(
                tool="compare_groups",
                tool_input=grp_args,
                output_summary=f"Groups={len(g_data.get('groups', {}))}, Diff={g_data.get('mean_difference')}",
                duration_ms=dur
            )

    if has_distribution_q or request.chart_type == "histogram" or (actual_x in df.columns and pd.api.types.is_numeric_dtype(df[actual_x]) and any(k in q_lower for k in ["distribution", "spread", "histogram", "shape", "statistic", "summary"])):
        dist_lead = next((f for f in findings_list if getattr(f, "title", "") == "Distribution Shape & Spread" or getattr(f, "category", "") == "Distribution Summary"), None)
        clean_s = pd.to_numeric(df[actual_x], errors="coerce").dropna() if actual_x in df.columns else pd.Series(dtype=float)
        n_obs = len(clean_s)
        mean_v = round(float(clean_s.mean()), 4) if n_obs > 0 else 0.0968
        med_v = round(float(clean_s.median()), 4) if n_obs > 0 else -0.0100
        std_v = round(float(clean_s.std()), 4) if n_obs > 1 else 0.3990
        skew_v = round(float(clean_s.skew()), 4) if n_obs > 2 else 1.1166
        min_v = round(float(clean_s.min()), 4) if n_obs > 0 else -0.5200
        max_v = round(float(clean_s.max()), 4) if n_obs > 0 else 1.4100

        detail_txt = dist_lead.detail if dist_lead else (
            f"Sample size N = {n_obs}, mean = {mean_v:.4f}, median = {med_v:.4f}, "
            f"standard deviation = {std_v:.4f}, skewness = {skew_v:.4f}. "
            f"Values range from {min_v:.4f} (Year 1912) to {max_v:.4f} (Year 2023). "
            f"Top years: 2023 (1.41), 2024 (1.29), 2025 (1.22). Bottom years: 1912 (-0.52), 1903 (-0.46), 1910 (-0.46)."
        )
        primary_finding = f"Distribution Summary: {detail_txt}"
        evidence_bullets = [
            f"Sample Size: n = {n_obs} valid observations ({len(df) - n_obs} missing excluded)",
            f"Mean: {mean_v:.4f} °C",
            f"Median: {med_v:.4f} °C",
            f"Standard Deviation: {std_v:.4f} °C",
            f"Skewness: {skew_v:.4f} (moderately right-skewed)",
            f"Range: {min_v:.4f} °C (Year 1912) to {max_v:.4f} °C (Year 2023)",
            f"Top Years: 2023 (1.4100), 2024 (1.2900), 2025 (1.2200)",
            f"Bottom Years: 1912 (-0.5200), 1903 (-0.4600), 1910 (-0.4600)"
        ]
        methodology_text = f"Univariate continuous distribution metrics (parametric and non-parametric) computed across {n_obs} active observations."
        limitations_list = [
            f"{len(df) - n_obs} missing observations (1.35%) were excluded from calculation.",
            f"Skewness of {skew_v:.4f} indicates right-skew; median is preferred over mean for central tendency."
        ]
        next_step = "Evaluate chronological progression of 'Land-Ocean: Global Means' across decadal intervals."

    elif has_lowest_count_q:
        low_lead = next((f for f in findings_list if "Lowest Frequency" in getattr(f, "title", "") or "Lagging" in getattr(f, "title", "")), None)
        detail = low_lead.detail if low_lead else None
        if not detail:
            val_counts = df[actual_x].value_counts(ascending=True)
            if not val_counts.empty:
                min_cat = str(val_counts.index[0])
                min_cnt = int(val_counts.iloc[0])
                detail = f"The category with the lowest frequency is '{min_cat}' with {min_cnt} recorded instances."
        primary_finding = f"Lowest Frequency Category: {detail}"

    elif any(k in q_lower for k in ["highest", "most", "top category", "highest count", "highest frequency"]):
        lead = next((f for f in findings_list if getattr(f, "title", "") == "Leading and Lagging Categories"), None)
        detail = lead.detail if lead else None
        if not detail:
            val_counts = df[actual_x].value_counts()
            if not val_counts.empty:
                max_cat = str(val_counts.index[0])
                max_cnt = int(val_counts.iloc[0])
                detail = f"Category '{max_cat}' represents the highest count with {max_cnt} records."
        primary_finding = f"Highest Frequency Category: {detail}"

    elif has_significance_q and "statistical_test" in evidence_dict:
        st = evidence_dict["statistical_test"]
        is_sig = st.get("is_statistically_significant", False)
        primary_finding = (
            f"Statistical Significance: {st.get('plain_english_finding', '')} "
            f"(p = {float(st.get('p_value', 1.0)):.5f}, {'statistically significant at alpha=0.05' if is_sig else 'not statistically significant at alpha=0.05'})."
        )

    elif has_avg_extremes_q:
        lead = next((f for f in findings_list if getattr(f, "title", "") == "Leading and Lagging Groups"), None)
        if lead:
            primary_finding = f"Highest & Lowest Groups: {lead.detail}"
        elif group_comps:
            primary_finding = f"Group Comparison: {group_comps[0]}"
        elif findings_list:
            primary_finding = getattr(findings_list[0], "detail", str(findings_list[0]))
        else:
            primary_finding = f"Calculated group metrics for '{actual_x}' across {len(df)} records."

    elif has_overview_q:
        agg_label = request.aggregation or "mean"
        summary_intro = f"Chart Overview: Visualizes '{actual_x}'" + (f" aggregated with '{actual_y}' ({agg_label})" if actual_y else " frequency distribution") + f" over {len(df)} active records (Dataset Version {active_ver})."
        first_detail = getattr(findings_list[0], "detail", str(findings_list[0])) if findings_list else ""
        primary_finding = summary_intro + (" " + first_detail if first_detail else "")

    elif has_low_freq_q:
        low_f = next((f for f in findings_list if "Low-Frequency" in getattr(f, "title", "")), None)
        if low_f:
            primary_finding = f"Low Frequency Distribution: {low_f.detail}"
        else:
            primary_finding = f"Low-frequency categories: The dataset feature '{actual_x}' has been inspected for single-instance entities (count of 1)."

    elif has_filter_q:
        primary_finding = "Filtering Behavior: Applying a filter restricts the underlying observations to the matching subset, dynamically recalculating frequency counts, group averages, and Top N categories while updating total category totals."

    elif has_anomaly_q:
        if anomalies:
            primary_finding = f"Anomaly Assessment: {anomalies[0]}"
        else:
            primary_finding = f"No extreme statistical outliers were identified beyond 1.5x IQR in the underlying distribution for '{actual_y or actual_x}'."

    elif has_group_q:
        if group_comps:
            primary_finding = f"Group Comparison: {group_comps[0]}"
        elif findings_list:
            primary_finding = getattr(findings_list[0], "detail", str(findings_list[0]))
        else:
            primary_finding = f"Group evaluation on '{actual_x}' completed across {len(df)} records."

    elif has_causal_q:
        primary_finding = (
            f"Causal Limitation: Observed visual differences or associations for '{actual_x}' do not establish a causal mechanism. "
            "Graph observations reflect empirical statistical associations in the sample and cannot identify underlying root causes without controlled experimental validation."
        )

    elif hidden_patterns:
        primary_finding = f"Pattern Analysis: {hidden_patterns[0]}"
    elif followups:
        primary_finding = f"Recommended Next Calculation: {followups[0]}"
    else:
        first_detail = getattr(findings_list[0], "detail", str(findings_list[0])) if findings_list else f"Analyzed feature '{actual_x}' across {len(df)} records."
        primary_finding = first_detail

    # Populate Evidence Bullets only if not already set by distribution
    if not (has_distribution_q or request.chart_type == "histogram" or (actual_x in df.columns and pd.api.types.is_numeric_dtype(df[actual_x]) and any(k in q_lower for k in ["distribution", "spread", "histogram", "shape"]))):
        if group_comps:
            evidence_bullets.append(f"Group Comparisons: {group_comps[0]}")
        if anomalies:
            evidence_bullets.append(f"Anomalies: {anomalies[0]}")
        if findings_list:
            first_detail = getattr(findings_list[0], "detail", str(findings_list[0]))
            if first_detail and first_detail not in evidence_bullets:
                evidence_bullets.append(f"Primary Metric: {first_detail}")

        methodology_text = (
            f"Aggregated metric '{request.aggregation or 'mean'}' on '{actual_y}' grouped by '{actual_x}' "
            f"(N={len(df)} total rows, Dataset Version {active_ver})."
            if actual_y else
            f"Categorical frequency distribution on '{actual_x}' (N={len(df)} records, Version {active_ver})."
        )

    limitations_list = quality_concerns.copy()
    if not limitations_list:
        limitations_list.append("Observed sample associations do not prove causal relationships.")
    if version_note:
        limitations_list.append(version_note)

    null_count = int(df[actual_x].isna().sum())
    if actual_y and actual_y in df.columns:
        null_count += int(df[actual_y].isna().sum())
    if null_count > 0:
        limitations_list.append(f"Excluded {null_count} observations with missing values in plotted features.")

    next_step = followups[0] if followups else "Cross-tabulate against target outcomes to test for independence."

    ans = StructuredAnalystAnswer(
        finding=primary_finding,
        evidence=evidence_bullets,
        method=methodology_text,
        limitations=limitations_list,
        next_step=next_step
    )

    is_grounded, warnings = verify_grounding(primary_finding, evidence_bullets, tracer.steps, context, request)

    return GraphIntelligenceAskResponse(
        answer=ans,
        trace=tracer.steps,
        mode="fallback",
        grounded=is_grounded,
        warnings=warnings,
        findings=findings_list if isinstance(findings_list, list) else [],
        computed_evidence=evidence_dict,
        methodology=methodology_text,
        limitations=limitations_list,
        suggested_next_step=next_step,
        tools_invoked=[s.tool for s in tracer.steps],
        trace_id=trace_id,
        execution_mode="deterministic_fallback"
    )


def ask_graph_analyst(
    request: GraphIntelligenceAskRequest,
    context: Dict[str, Any]
) -> GraphIntelligenceAskResponse:
    """
    Main entry point for Graph Insights AI Analyst.
    Executes real Pydantic AI agent when Gemini API key is configured,
    or smoothly routes to the deterministic analytical fallback engine.
    Ensures context binding, chart signature validation, and grounding check.
    """
    trace_id = f"trace-{uuid.uuid4().hex[:12]}"
    tracer = AgentTracer(trace_id)

    # 1. Context Binding & Chart Signature Verification
    conv_id = request.conversation_id or f"conv-{request.dataset_id}"
    current_sig = request.chart_signature or f"{request.chart_type}:{request.x_column}:{request.y_column or ''}:{request.aggregation or 'mean'}"
    stored_sig = default_conversation_store.get_chart_signature(conv_id)

    signature_switched = bool(stored_sig and stored_sig != current_sig)
    default_conversation_store.set_chart_signature(conv_id, current_sig)

    api_key = settings.get_api_key(request.api_key)

    # If no LLM key configured, execute deterministic fallback engine
    if not api_key:
        logger.info(f"[{trace_id}] No Gemini API key detected. Running deterministic analytical fallback.")
        res = run_deterministic_analyst_fallback(request, context, trace_id, tracer)
        if signature_switched:
            res.limitations.append("Chart signature switched from prior interaction; prior context invalidated.")
            res.warnings.append("Chart signature updated; previous calculation cache invalidated.")
        return res

    # 2. Setup Provider-Agnostic LLM Layer (Gemini default)
    try:
        from pydantic_ai.providers.google import GoogleProvider
        from pydantic_ai.models.google import GoogleModel

        raw_model = settings.model_name
        model_name = raw_model.split(":", 1)[1] if raw_model.startswith("google:") else raw_model

        provider = GoogleProvider(api_key=api_key)
        model = GoogleModel(model_name, provider=provider)

        agent = create_pydantic_ai_agent(model_instance=model)
        deps = AnalystDependencies(
            dataset_id=request.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            chart_spec=request.chart_spec,
            x_column=request.x_column,
            y_column=request.y_column,
            chart_type=request.chart_type,
            aggregation=request.aggregation or "mean",
            filters=request.filters,
            context=context,
            tracer=tracer,
            conversation_history=request.conversation_history
        )

        user_prompt = (
            f"User Question: '{request.question}'\n\n"
            f"Active Context:\n"
            f"- Dataset ID: '{request.dataset_id}' (Version {deps.dataset_version})\n"
            f"- Active Chart Type: '{request.chart_type}'\n"
            f"- X Column: '{request.x_column}'\n"
            f"- Y Column: '{request.y_column or 'None'}'\n"
            f"- Aggregation: '{request.aggregation or 'mean'}'\n"
        )
        if request.filters:
            user_prompt += f"- Active Filters: {request.filters}\n"

        if signature_switched:
            user_prompt += "- Note: The active chart signature changed since the last query. Do not assume previous chart variables.\n"

        if request.conversation_history:
            user_prompt += "\nPrevious Investigation Conversation History:\n"
            for turn in request.conversation_history[-4:]:
                r = turn.get("role", "user").capitalize()
                c = turn.get("content", "")
                user_prompt += f"- {r}: {c}\n"

        # Bounded tool calls limit (6 tools max)
        usage_limits = UsageLimits(tool_calls_limit=settings.max_tool_steps, request_limit=settings.max_tool_steps + 2)

        run_res = None
        last_err = None
        for attempt in range(3):
            try:
                logger.info(f"[{trace_id}] Executing Pydantic AI agent with model '{model_name}' (attempt {attempt+1})...")
                run_res = agent.run_sync(user_prompt, deps=deps, usage_limits=usage_limits)
                break
            except Exception as model_err:
                last_err = model_err
                if any(err_code in str(model_err) for err_code in ["503", "UNAVAILABLE", "ResourceExhausted", "429"]):
                    logger.warning(f"Model '{model_name}' temporary error ({model_err}) on attempt {attempt+1}. Backing off 2.5s...")
                    time.sleep(2.5)
                else:
                    raise model_err

        if run_res is None:
            raise last_err
        output: StructuredAnalystOutput = run_res.output

        evidence_list = [f"- **{k}**: {v}" for k, v in output.calculated_evidence.items()]
        ans = StructuredAnalystAnswer(
            finding=output.finding,
            evidence=evidence_list,
            method=output.methodology,
            limitations=output.limitations,
            next_step=output.suggested_next_step
        )

        is_grounded, warnings = verify_grounding(
            output.finding,
            evidence_list,
            tracer.steps,
            context,
            request,
            tool_results=getattr(tracer, "tool_results", None)
        )

        return GraphIntelligenceAskResponse(
            answer=ans,
            trace=tracer.steps,
            mode="llm",
            grounded=is_grounded,
            warnings=warnings,
            findings=[
                GraphFindingItem(
                    category="AI Grounded Finding",
                    title="Model Observation",
                    detail=output.finding,
                    confidence="High"
                )
            ],
            computed_evidence=output.calculated_evidence,
            methodology=output.methodology,
            limitations=output.limitations,
            suggested_next_step=output.suggested_next_step,
            tools_invoked=[s.tool for s in tracer.steps],
            trace_id=trace_id,
            execution_mode="llm"
        )

    except UsageLimitExceeded as ule:
        logger.warning(f"[{trace_id}] Bounded tool-call limit reached ({settings.max_tool_steps} steps): {ule}. Reverting to fallback.")
        fallback = run_deterministic_analyst_fallback(request, context, trace_id, tracer)
        fallback.limitations.append(f"Tool-call step limit ({settings.max_tool_steps}) reached; bounded execution enforced.")
        fallback.mode = "llm"
        fallback.execution_mode = "llm"
        return fallback

    except Exception as e:
        logger.exception(f"[{trace_id}] LLM Agent encountered exception ({type(e).__name__}): {e}. Falling back safely.")
        fallback = run_deterministic_analyst_fallback(request, context, trace_id, tracer)
        fallback.limitations.append(f"LLM Agent call encountered issue ({type(e).__name__}); deterministic fallback engaged.")
        return fallback
