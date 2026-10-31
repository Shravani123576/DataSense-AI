"""
DataSense AI V6 — Dedicated Graph AI Analyst Test Suite
Verifies all 16 required analytical scenarios for the contextual Graph AI Analyst
against the Step 2 Structured Contract:
{answer: {finding, evidence[], method, limitations[], next_step}, trace: [...], mode: "llm" | "fallback", grounded: bool, warnings: []}
"""
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from main import app, DATASET_STORAGE, DATASET_VERSIONS_STORAGE
from analyst_agent import (
    create_pydantic_ai_agent,
    AnalystDependencies,
    ask_graph_analyst
)
from models import GraphIntelligenceAskRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

test_client = TestClient(app)


def check_answer_contract(data: dict) -> str:
    """Verifies that data complies with the structured response contract and returns the flattened answer string for text assertions."""
    assert "answer" in data, "Contract violation: missing 'answer'"
    assert "trace" in data, "Contract violation: missing 'trace'"
    assert "mode" in data, "Contract violation: missing 'mode'"
    assert data["mode"] in ["llm", "fallback"], f"Unexpected mode: {data['mode']}"
    assert "grounded" in data, "Contract violation: missing 'grounded'"
    assert "warnings" in data, "Contract violation: missing 'warnings'"

    ans = data["answer"]
    assert isinstance(ans, dict), "Contract violation: answer must be a structured dict"
    for k in ["finding", "evidence", "method", "limitations", "next_step"]:
        assert k in ans, f"Contract violation: answer missing '{k}'"

    text = f"{ans['finding']} {' '.join(ans['evidence'])} {ans['method']} {' '.join(ans['limitations'])} {ans['next_step'] or ''}"
    return text


@pytest.fixture
def test_datasets():
    """Create deterministic synthetic datasets with independently verifiable ground truth."""
    # 1. Categorical dataset
    # Alpha: 10, Beta: 6, Gamma: 2, Delta: 1 (Lowest: Delta, Highest: Alpha)
    cat_df = pd.DataFrame({
        "department": ["Alpha"] * 10 + ["Beta"] * 6 + ["Gamma"] * 2 + ["Delta"] * 1,
        "performance": [92.0] * 10 + [75.0] * 6 + [55.0] * 2 + [40.0] * 1
    })

    # 2. Time-series dataset with steep acceleration in second half
    # 20 days: Days 0-9 increase by 1/day; Days 10-19 increase by 5/day (steep slope!)
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    sales = [100.0 + 1.0 * i for i in range(10)] + [110.0 + 5.0 * (i - 10) for i in range(10, 20)]
    ts_df = pd.DataFrame({
        "date": dates,
        "revenue": sales
    })

    # 3. Anomaly dataset: 20 regular points + 1 massive outlier (value=999)
    anomaly_vals = [20.0 + (i % 3) for i in range(20)] + [999.0]
    anomaly_df = pd.DataFrame({
        "item_id": [f"item_{i}" for i in range(21)],
        "metric": anomaly_vals
    })

    # 4. Missing values dataset: 10 rows with 3 NaN values in metric
    missing_vals = [10.0, 20.0, None, 40.0, None, 60.0, 70.0, None, 90.0, 100.0]
    missing_df = pd.DataFrame({
        "region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
        "revenue": missing_vals
    })

    # 5. Statistical comparison dataset: Group A (high mean ~100) vs Group B (low mean ~50)
    stat_df = pd.DataFrame({
        "cohort": ["Treatment"] * 15 + ["Control"] * 15,
        "outcome": [100.0 + (i % 4) for i in range(15)] + [50.0 + (i % 4) for i in range(15)]
    })

    return {
        "cat_df": cat_df,
        "ts_df": ts_df,
        "anomaly_df": anomaly_df,
        "missing_df": missing_df,
        "stat_df": stat_df
    }


# Test 1: Simple question about the meaning of a bar chart
def test_1_meaning_of_bar_chart(test_datasets):
    sess_id = "test_sess_meaning"
    DATASET_STORAGE[sess_id] = test_datasets["cat_df"].copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(test_datasets["cat_df"])}]

    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "department",
        "y_column": "performance",
        "chart_type": "bar",
        "question": "What does this graph show?"
    })
    assert resp.status_code == 200
    data = resp.json()
    ans_text = check_answer_contract(data)
    assert "department" in ans_text
    assert "performance" in ans_text
    assert "19" in ans_text


# Test 2: Correct identification of category with highest frequency
def test_2_highest_frequency_category(test_datasets):
    sess_id = "test_sess_highest_cat"
    DATASET_STORAGE[sess_id] = test_datasets["cat_df"].copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(test_datasets["cat_df"])}]

    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "department",
        "y_column": None,
        "chart_type": "bar",
        "question": "Which category has the highest count?"
    })
    assert resp.status_code == 200
    data = resp.json()
    ans_text = check_answer_contract(data)
    assert "Alpha" in ans_text
    assert "10" in ans_text


# Test 3: Time-series question about increasing or decreasing trends
def test_3_time_series_trend_direction(test_datasets):
    sess_id = "test_sess_ts_trend"
    DATASET_STORAGE[sess_id] = test_datasets["ts_df"].copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(test_datasets["ts_df"])}]

    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "date",
        "y_column": "revenue",
        "chart_type": "line",
        "question": "What is the overall trend over time?"
    })
    assert resp.status_code == 200
    data = resp.json()
    ans_text = check_answer_contract(data)
    assert "upward" in ans_text.lower()
    assert "Linear Slope" in ans_text or "linear slope" in ans_text.lower()
    assert data["computed_evidence"]["linear_trend"]["direction"] == "upward"


# Test 4: Steep-slope question that requires an actual calculated rate of change
def test_4_steep_slope_rate_of_change(test_datasets):
    sess_id = "test_sess_steep_slope"
    DATASET_STORAGE[sess_id] = test_datasets["ts_df"].copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(test_datasets["ts_df"])}]

    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "date",
        "y_column": "revenue",
        "chart_type": "line",
        "question": "Why does the slope become steeper during this period?"
    })
    assert resp.status_code == 200
    data = resp.json()
    ans_text = check_answer_contract(data)
    assert "slope" in ans_text.lower()
    assert "rate" in ans_text.lower() or "acceleration" in ans_text.lower()
    assert "observational" in ans_text.lower() or "causal" in ans_text.lower()


# Test 5: Comparison of two time periods
def test_5_period_comparison(test_datasets):
    sess_id = "test_sess_periods"
    DATASET_STORAGE[sess_id] = test_datasets["ts_df"].copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(test_datasets["ts_df"])}]

    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "date",
        "y_column": "revenue",
        "chart_type": "line",
        "question": "How do two selected periods compare?"
    })
    assert resp.status_code == 200
    data = resp.json()
    ans_text = check_answer_contract(data)
    assert "early" in ans_text.lower()
    assert "late" in ans_text.lower() or "delta" in ans_text.lower()


# Test 6: Graph with a known anomaly
def test_6_known_anomaly_detection(test_datasets):
    sess_id = "test_sess_anomaly"
    DATASET_STORAGE[sess_id] = test_datasets["anomaly_df"].copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(test_datasets["anomaly_df"])}]

    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "item_id",
        "y_column": "metric",
        "chart_type": "bar",
        "question": "Are there unusual observations or outliers?"
    })
    assert resp.status_code == 200
    data = resp.json()
    ans_text = check_answer_contract(data)
    assert "outlier" in ans_text.lower() or "anomaly" in ans_text.lower()
    assert data["computed_evidence"]["anomalies_detected_count"] > 0


# Test 7: Statistical-significance question using an appropriate test
def test_7_statistical_significance_test(test_datasets):
    sess_id = "test_sess_stat_test"
    DATASET_STORAGE[sess_id] = test_datasets["stat_df"].copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(test_datasets["stat_df"])}]

    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "cohort",
        "y_column": "outcome",
        "chart_type": "bar",
        "question": "Is the observed difference statistically significant?"
    })
    assert resp.status_code == 200
    data = resp.json()
    ans_text = check_answer_contract(data)
    assert "statistically significant" in ans_text.lower()
    assert "p =" in ans_text or "p-value" in ans_text.lower()
    assert "run_statistical_test" in data["tools_invoked"]


# Test 8: Graph with missing values or excluded observations
def test_8_missing_values_documented(test_datasets):
    sess_id = "test_sess_missing"
    DATASET_STORAGE[sess_id] = test_datasets["missing_df"].copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(test_datasets["missing_df"])}]

    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "region",
        "y_column": "revenue",
        "chart_type": "bar",
        "question": "What does this chart show?"
    })
    assert resp.status_code == 200
    data = resp.json()
    check_answer_contract(data)
    assert any("missing values" in lim.lower() for lim in data["answer"]["limitations"])


# Test 9: Graph with filters applied
def test_9_graph_with_filters(test_datasets):
    sess_id = "test_sess_filtered"
    DATASET_STORAGE[sess_id] = test_datasets["cat_df"].copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(test_datasets["cat_df"])}]

    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "department",
        "y_column": "performance",
        "chart_type": "bar",
        "filters": {"department": ["Alpha", "Beta"]},
        "question": "How do changes when I filter appear?"
    })
    assert resp.status_code == 200
    data = resp.json()
    ans_text = check_answer_contract(data)
    assert "filter" in ans_text.lower()


# Test 10: Follow-up question that depends on prior context
def test_10_followup_question_with_conversation_history(test_datasets):
    sess_id = "test_sess_followup"
    DATASET_STORAGE[sess_id] = test_datasets["ts_df"].copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(test_datasets["ts_df"])}]

    history = [
        {"role": "user", "content": "What is the revenue trend over time?"},
        {"role": "assistant", "content": "Revenue exhibits an upward trend with slope +3.0 units/day."}
    ]
    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "date",
        "y_column": "revenue",
        "chart_type": "line",
        "conversation_history": history,
        "question": "Why is it steep?"
    })
    assert resp.status_code == 200
    data = resp.json()
    ans_text = check_answer_contract(data)
    assert "slope" in ans_text.lower() or "rate" in ans_text.lower()
    assert "analyze_time_series" in data["tools_invoked"]


# Test 11: Dataset change that must invalidate stale context
def test_11_dataset_version_stale_invalidation(test_datasets):
    sess_id = "test_sess_stale_version"
    DATASET_STORAGE[sess_id] = test_datasets["cat_df"].copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [
        {"version_id": 0, "rows": 19},
        {"version_id": 1, "rows": 19},
        {"version_id": 2, "rows": 19}
    ]

    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "dataset_version": 0,
        "x_column": "department",
        "y_column": None,
        "chart_type": "bar",
        "question": "Which category has the highest count?"
    })
    assert resp.status_code == 200
    data = resp.json()
    check_answer_contract(data)
    assert any("version" in lim.lower() for lim in data["answer"]["limitations"])


# Test 12: Inappropriate analytical method (trend query on unordered categorical)
def test_12_inappropriate_method_rejection(test_datasets):
    sess_id = "test_sess_inappropriate"
    DATASET_STORAGE[sess_id] = test_datasets["cat_df"].copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(test_datasets["cat_df"])}]

    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "department",
        "y_column": "performance",
        "chart_type": "bar",
        "question": "What is the trend over time and rate of change per year?"
    })
    assert resp.status_code == 200
    data = resp.json()
    ans_text = check_answer_contract(data)
    assert "limitation" in ans_text.lower() or "cannot be parsed" in ans_text.lower()
    assert "unordered categorical data" in str(data["answer"]["limitations"])


# Test 13: Missing or inaccessible dataset returns 404
def test_13_missing_dataset_returns_404():
    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": "non_existent_dataset_abc_123",
        "x_column": "col1",
        "chart_type": "bar",
        "question": "What does this graph show?"
    })
    assert resp.status_code == 404
    assert "Dataset session not found" in resp.json()["detail"]


# Test 14: Invalid tool arguments and graceful error recovery
def test_14_invalid_arguments_graceful_recovery(test_datasets):
    sess_id = "test_sess_invalid_args"
    DATASET_STORAGE[sess_id] = test_datasets["cat_df"].copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(test_datasets["cat_df"])}]

    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "completely_fictional_column_xyz",
        "chart_type": "bar",
        "question": "What does this show?"
    })
    assert resp.status_code == 200
    data = resp.json()
    ans_text = check_answer_contract(data)
    assert "error" in ans_text.lower() or "not found" in ans_text.lower() or "cannot be analyzed" in ans_text.lower()


# Test 15: Tool-call limit and bounded execution safety
def test_15_tool_call_limit_bounded_safety(test_datasets):
    agent = create_pydantic_ai_agent(model_instance=TestModel(call_tools="all"))
    deps = AnalystDependencies(
        dataset_id="test_sess_limit",
        dataset_version=0,
        chart_spec=None,
        x_column="department",
        y_column="performance",
        chart_type="bar",
        aggregation="mean",
        filters=None,
        context={"df": test_datasets["cat_df"], "dataset_id": "test_sess_limit", "dataset_version": 0}
    )

    with pytest.raises(Exception):
        agent.run_sync(
            "Execute all tools",
            deps=deps,
            usage_limits=UsageLimits(tool_calls_limit=1, request_limit=3)
        )


# Test 16: Data cannot support requested conclusion (causality check)
def test_16_causality_caveat_unsupported_conclusion(test_datasets):
    sess_id = "test_sess_causal"
    DATASET_STORAGE[sess_id] = test_datasets["cat_df"].copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(test_datasets["cat_df"])}]

    resp = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "department",
        "y_column": "performance",
        "chart_type": "bar",
        "question": "Does being in department Alpha cause higher performance?"
    })
    assert resp.status_code == 200
    data = resp.json()
    ans_text = check_answer_contract(data)
    assert "causal" in ans_text.lower()
    assert "not establish a causal mechanism" in ans_text.lower() or "observational" in ans_text.lower()
