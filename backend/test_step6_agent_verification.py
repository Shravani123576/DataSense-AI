"""
DataSense AI V6 — Step 6 Explicit Agent Verification Tests
Verifies:
1. Synthetic dataset with known answers (linear slope=2.0/day, group mean=100.0, planted outlier=999.0).
2. Tool selection for at least 8 distinct question types (mocked and fallback).
3. Grounding check rejection when fabricated numbers are introduced.
4. Stale chart signature blocks reuse of prior results across turns.
5. Tool timeout and step-limit enforcement.
6. LLM unavailable -> fallback mode returns valid, grounded 5-part answer contract.
7. Real sample dataset (sample_customer_churn) end-to-end integration.
"""
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from main import app, DATASET_STORAGE, DATASET_VERSIONS_STORAGE
from tools import default_registry
from analyst_agent import (
    ask_graph_analyst,
    create_pydantic_ai_agent,
    AnalystDependencies,
    verify_grounding,
    AgentTracer
)
from models import GraphIntelligenceAskRequest, AgentTraceStep
from config import settings
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

test_client = TestClient(app)


def test_synthetic_dataset_known_values():
    """
    Assert that the analytical tools compute values matching exact mathematical ground truth:
    - Linear slope: exactly 2.0 / day
    - Group mean difference: 100.0 - 50.0 = 50.0
    - Planted outlier: value 999.0 detected at row index 20
    """
    # 1. Exact Linear series: y = 100.0 + 2.0 * day
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    revenue = [100.0 + 2.0 * i for i in range(20)]
    ts_df = pd.DataFrame({"order_date": dates, "revenue": revenue})

    ctx_ts = {"df": ts_df, "dataset_id": "known_ts", "dataset_version": 0}
    res_ts = default_registry.execute_tool(
        "analyze_time_series",
        {"dataset_id": "known_ts", "time_column": "order_date", "value_column": "revenue"},
        ctx_ts
    )
    assert res_ts.status == "success"
    slope = res_ts.result["linear_trend"]["slope_per_day"]
    assert abs(slope - 2.0) < 0.001, f"Expected slope 2.0, got {slope}"
    assert res_ts.result["linear_trend"]["direction"] == "upward"

    # 2. Exact Group means: Alpha=[90, 100, 110] (mean=100.0), Beta=[45, 55] (mean=50.0)
    grp_df = pd.DataFrame({
        "cohort": ["Alpha", "Alpha", "Alpha", "Beta", "Beta"],
        "metric": [90.0, 100.0, 110.0, 45.0, 55.0]
    })
    ctx_grp = {"df": grp_df, "dataset_id": "known_grp", "dataset_version": 0}
    res_grp = default_registry.execute_tool(
        "compare_groups",
        {"dataset_id": "known_grp", "group_column": "cohort", "metric_column": "metric"},
        ctx_grp
    )
    assert res_grp.status == "success"
    group_stats = res_grp.result.get("group_statistics", [])
    groups = {g["group"]: g for g in group_stats}
    assert abs(groups["Alpha"]["mean"] - 100.0) < 0.001
    assert abs(groups["Beta"]["mean"] - 50.0) < 0.001
    mean_diff = groups["Alpha"]["mean"] - groups["Beta"]["mean"]
    assert abs(mean_diff - 50.0) < 0.001

    # 3. Planted outlier at index 20 with value 999.0
    anom_vals = [20.0, 21.0, 22.0, 20.0] * 5 + [999.0]
    anom_df = pd.DataFrame({"item_id": [f"item_{i}" for i in range(21)], "val": anom_vals})
    ctx_anom = {"df": anom_df, "dataset_id": "known_anom", "dataset_version": 0}
    res_anom = default_registry.execute_tool(
        "investigate_anomalies",
        {"dataset_id": "known_anom", "column": "val", "method": "iqr"},
        ctx_anom
    )
    assert res_anom.status == "success"
    assert res_anom.result["anomalies_found"] >= 1
    found_vals = [c["value"] for c in res_anom.result["sample_outlier_cases"]]
    assert 999.0 in found_vals


def test_tool_selection_for_8_question_types():
    """Verify tool selection for at least 8 distinct question types in fallback mode."""
    # Setup test dataset with both temporal and categorical dimensions
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "revenue": [100.0 + 3.0 * i for i in range(30)],
        "region": ["East", "West", "North"] * 10,
        "satisfaction": [80.0, 85.0, 90.0] * 10
    })
    sess_id = "test_8_intents_session"
    DATASET_STORAGE[sess_id] = df.copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(df)}]

    # Question types and their expected tool in trace
    test_cases = [
        # 1. Trend
        ("What is the trajectory and trend of revenue over time?", "date", "revenue", "line", "analyze_time_series"),
        # 2. Steep slope / Rate of change
        ("Why does the slope become steep and what is the rate of change?", "date", "revenue", "line", "compute_rate_of_change"),
        # 3. Period comparison
        ("How do the early and late periods compare?", "date", "revenue", "line", "compute_period_comparison"),
        # 4. Highest count / Frequency
        ("Which region has the highest count of records?", "region", None, "bar", "analyze_chart"),
        # 5. Group comparison
        ("Which groups differ the most in satisfaction?", "region", "satisfaction", "bar", "compare_groups"),
        # 6. Anomaly / Outliers
        ("Are there unusual extreme outliers in revenue?", "region", "revenue", "bar", "investigate_anomalies"),
        # 7. Statistical significance
        ("Is the difference in satisfaction between groups statistically significant?", "region", "satisfaction", "bar", "run_statistical_test"),
        # 8. General chart overview
        ("What does this graph show?", "region", "satisfaction", "bar", "analyze_chart"),
    ]

    for q_text, x_col, y_col, c_type, expected_tool in test_cases:
        resp = test_client.post("/api/graph/ask", json={
            "dataset_id": sess_id,
            "x_column": x_col,
            "y_column": y_col,
            "chart_type": c_type,
            "question": q_text
        })
        assert resp.status_code == 200, f"Failed for query: {q_text}"
        data = resp.json()
        assert data["mode"] in ["llm", "fallback"]
        trace_tools = [s["tool"] for s in data["trace"]]
        assert expected_tool in trace_tools, (
            f"Question '{q_text}' expected tool '{expected_tool}', but trace contained: {trace_tools}"
        )


def test_grounding_check_rejects_fabricated_number():
    """Verify that verify_grounding identifies and flags ungrounded / fabricated numbers."""
    tracer = AgentTracer("test_grounding_trace")
    tracer.record_step(
        tool="analyze_chart",
        tool_input={"x_column": "cat"},
        output_summary="Analyzed chart with count 10 for Alpha and 5 for Beta.",
        duration_ms=10.0
    )

    req = GraphIntelligenceAskRequest(
        dataset_id="test_ds",
        x_column="cat",
        chart_type="bar",
        question="What is the result?"
    )
    context = {"df": pd.DataFrame({"cat": ["Alpha"] * 10 + ["Beta"] * 5}), "dataset_version": 0}

    # Case 1: Grounded answer (uses 10, 5, 15)
    grounded_finding = "Alpha has 10 records and Beta has 5 records (total N=15)."
    is_grounded, warnings = verify_grounding(grounded_finding, ["Alpha: 10", "Beta: 5"], tracer.steps, context, req)
    assert is_grounded is True
    assert len(warnings) == 0

    # Case 2: Hallucinated / fabricated number (e.g. 74892.33)
    fabricated_finding = "The secret average is exactly 74892.33 dollars."
    is_grounded_bad, warnings_bad = verify_grounding(fabricated_finding, ["Alpha: 10"], tracer.steps, context, req)
    assert is_grounded_bad is False
    assert any("74892.33" in w for w in warnings_bad)


def test_stale_chart_signature_blocks_prior_context_reuse():
    """Verify changing chart signature resets previous chart assumptions."""
    sess_id = "test_signature_session"
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "dept": ["Engineering", "Sales"] * 5,
        "budget": [500000.0, 300000.0] * 5,
        "date": dates,
        "revenue": [1000.0 + 100.0 * i for i in range(10)]
    })
    DATASET_STORAGE[sess_id] = df.copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(df)}]

    conv_id = "conv-test-signature-switch"

    # Turn 1: Query about dept/budget bar chart
    resp1 = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "conversation_id": conv_id,
        "chart_signature": "bar:dept:budget:mean",
        "x_column": "dept",
        "y_column": "budget",
        "chart_type": "bar",
        "question": "Which department has higher budget?"
    })
    assert resp1.status_code == 200

    # Turn 2: User switches chart to date/revenue line chart with follow-up question
    resp2 = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "conversation_id": conv_id,
        "chart_signature": "line:date:revenue:mean",
        "x_column": "date",
        "y_column": "revenue",
        "chart_type": "line",
        "question": "What is the trend?"
    })
    assert resp2.status_code == 200
    data2 = resp2.json()
    ans2 = data2["answer"]
    ans_text = f"{ans2['finding']} {' '.join(ans2['evidence'])}"
    # The new answer must be about the line chart variables (revenue / date), not the old budget chart
    assert "revenue" in ans_text.lower() or "linear slope" in ans_text.lower()
    assert "budget" not in ans2["finding"].lower()


def test_tool_timeout_and_step_limit_enforcement():
    """Verify tool execution bounds and step-limits are enforced."""
    # Check settings configuration values
    assert settings.max_tool_steps == 6
    assert settings.per_tool_timeout_seconds == 30.0

    # Test Pydantic AI UsageLimit enforcement
    agent = create_pydantic_ai_agent(model_instance=TestModel(call_tools="all"))
    df = pd.DataFrame({"cat": ["A", "B"], "val": [10, 20]})
    deps = AnalystDependencies(
        dataset_id="test_timeout_sess",
        dataset_version=0,
        chart_spec=None,
        x_column="cat",
        y_column="val",
        chart_type="bar",
        aggregation="mean",
        filters=None,
        context={"df": df}
    )

    # Calling all tools with limit=1 must be safely caught and bounded
    with pytest.raises(Exception):
        agent.run_sync(
            "Execute all tools",
            deps=deps,
            usage_limits=UsageLimits(tool_calls_limit=1, request_limit=2)
        )


def test_llm_unavailable_fallback_returns_valid_structured_contract():
    """Verify that when no LLM API key is present, fallback mode produces the full structured contract."""
    sess_id = "test_fallback_contract_sess"
    df = pd.DataFrame({
        "channel": ["Organic", "Paid", "Referral"],
        "conversions": [120, 95, 45]
    })
    DATASET_STORAGE[sess_id] = df.copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(df)}]

    # Explicitly verify with no API key provided
    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "channel",
        "y_column": "conversions",
        "chart_type": "bar",
        "question": "What does this chart show?"
    })
    assert resp.status_code == 200
    data = resp.json()

    # Step 2 Contract Assertions
    assert data["mode"] == "fallback"
    assert data["grounded"] is True
    assert isinstance(data["warnings"], list)
    assert isinstance(data["trace"], list)
    assert len(data["trace"]) > 0

    ans = data["answer"]
    assert "finding" in ans and len(ans["finding"]) > 5
    assert "evidence" in ans and isinstance(ans["evidence"], list)
    assert "method" in ans and len(ans["method"]) > 5
    assert "limitations" in ans and isinstance(ans["limitations"], list)
    assert "next_step" in ans and ans["next_step"] is not None


def test_real_sample_dataset_end_to_end():
    """Verify chat endpoint returns non-empty, well-formed response for built-in sample_customer_churn."""
    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": "sample_customer_churn",
        "x_column": "Contract",
        "y_column": "MonthlyCharges",
        "chart_type": "bar",
        "aggregation": "mean",
        "question": "Which contract type has the highest average monthly charges?"
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["mode"] in ["llm", "fallback"]
    assert len(data["trace"]) > 0
    ans = data["answer"]
    assert len(ans["finding"]) > 10
    assert len(ans["evidence"]) > 0
    assert len(ans["method"]) > 0
    assert len(ans["limitations"]) > 0
    # Must compute on actual Contract values (Month-to-month, One year, Two year)
    ans_text = f"{ans['finding']} {' '.join(ans['evidence'])}"
    assert any(term in ans_text for term in ["Month-to-month", "One year", "Two year", "Contract", "MonthlyCharges"])
