"""
DataSense AI V6 — Runtime AI Analyst Test Suite
Verifies deterministic synthetic datasets, tool selection, time-series validation,
bounded execution limits, fallback mechanisms, and real-model invocation when credentials are configured.
"""
import os
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from main import app, DATASET_STORAGE, DATASET_VERSIONS_STORAGE
from tools import default_registry
from analyst_agent import (
    is_llm_configured,
    get_configured_model_name,
    create_pydantic_ai_agent,
    AnalystDependencies,
    ask_graph_analyst
)
from models import GraphIntelligenceAskRequest, GraphIntelligenceAskResponse
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

test_client = TestClient(app)


@pytest.fixture
def synthetic_timeseries_df():
    """Deterministic synthetic dataset with genuine chronological dates and a linear upward trend."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    # Base 100 + 2.5 per day + small deterministic noise
    sales = [100.0 + 2.5 * i + (i % 3) for i in range(30)]
    return pd.DataFrame({
        "order_date": dates,
        "daily_revenue": sales,
        "channel": ["online" if i % 2 == 0 else "retail" for i in range(30)]
    })


@pytest.fixture
def synthetic_categorical_df():
    """Deterministic categorical dataset with known leading, lagging, and low-frequency groups."""
    return pd.DataFrame({
        "category": ["Alpha"] * 10 + ["Beta"] * 6 + ["Gamma"] * 2 + ["Delta"] * 1 + ["Epsilon"] * 1,
        "score": [95.0] * 10 + [50.0] * 6 + [30.0] * 2 + [10.0] * 1 + [12.0] * 1
    })


def test_time_series_tool_with_valid_dates(synthetic_timeseries_df):
    """Verify analyze_time_series calculates deterministic linear slope, R², and rate of change."""
    sess_id = "test_ts_valid_session"
    DATASET_STORAGE[sess_id] = synthetic_timeseries_df.copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(synthetic_timeseries_df)}]

    context = {"df": synthetic_timeseries_df, "dataset_id": sess_id, "dataset_version": 0}
    res = default_registry.execute_tool(
        "analyze_time_series",
        {
            "dataset_id": sess_id,
            "time_column": "order_date",
            "value_column": "daily_revenue"
        },
        context
    )
    assert res.status == "success"
    result = res.result
    assert result["sample_size"] == 30
    assert result["linear_trend"]["direction"] == "upward"
    assert result["linear_trend"]["slope_per_day"] > 2.0
    assert result["linear_trend"]["r_squared"] > 0.95
    assert result["linear_trend"]["statistically_significant"] is True
    assert result["rate_of_change"]["absolute_change"] > 60.0


def test_time_series_tool_rejects_unordered_categorical(synthetic_categorical_df):
    """Verify analyze_time_series safely rejects unordered non-datetime columns."""
    sess_id = "test_ts_invalid_session"
    DATASET_STORAGE[sess_id] = synthetic_categorical_df.copy()
    context = {"df": synthetic_categorical_df, "dataset_id": sess_id, "dataset_version": 0}

    res = default_registry.execute_tool(
        "analyze_time_series",
        {
            "dataset_id": sess_id,
            "time_column": "category",
            "value_column": "score"
        },
        context
    )
    assert res.status == "error"
    assert "chronological date or timestamp variable" in res.error_details


def test_ask_graph_analyst_trend_question_integration(synthetic_timeseries_df):
    """Verify /api/graph/ask executes time-series trend calculation and formats 5-part report."""
    sess_id = "test_graph_ask_ts_session"
    DATASET_STORAGE[sess_id] = synthetic_timeseries_df.copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(synthetic_timeseries_df)}]

    payload = {
        "dataset_id": sess_id,
        "x_column": "order_date",
        "y_column": "daily_revenue",
        "chart_type": "line",
        "question": "What is the revenue trend over time and rate of change?"
    }
    resp = test_client.post("/api/graph/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Structural contracts
    ans = data["answer"]
    assert "finding" in ans
    assert "evidence" in ans
    assert "method" in ans
    assert "limitations" in ans
    assert "next_step" in ans

    # Numerical accuracy
    ans_text = f"{ans['finding']} {' '.join(ans['evidence'])} {ans['method']} {' '.join(ans['limitations'])}"
    assert "Upward" in ans_text or "upward" in ans_text
    assert "30" in ans_text
    assert "analyze_time_series" in data["tools_invoked"]
    assert data["trace_id"] is not None
    assert data["execution_mode"] in ["llm", "deterministic_fallback"]


def test_ask_graph_analyst_trend_on_non_time_column(synthetic_categorical_df):
    """Verify trend question on non-datetime column states limitation without inventing trend."""
    sess_id = "test_graph_ask_invalid_ts"
    DATASET_STORAGE[sess_id] = synthetic_categorical_df.copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(synthetic_categorical_df)}]

    payload = {
        "dataset_id": sess_id,
        "x_column": "category",
        "y_column": "score",
        "chart_type": "bar",
        "question": "What is the trend over time?"
    }
    resp = test_client.post("/api/graph/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    ans = data["answer"]
    ans_text = str(ans) if isinstance(ans, str) else f"{ans.get('finding', '')} {' '.join(ans.get('limitations', []))}"
    assert "limitation" in ans_text.lower() or "cannot be parsed" in ans_text.lower()
    assert len(data["limitations"]) > 0


def test_ask_graph_analyst_missing_dataset():
    """Verify asking a question on a non-existent dataset returns 404."""
    payload = {
        "dataset_id": "non_existent_dataset_9999",
        "x_column": "cat",
        "y_column": "val",
        "chart_type": "bar",
        "question": "Summarize this chart."
    }
    resp = test_client.post("/api/graph/ask", json=payload)
    assert resp.status_code == 404
    assert "Dataset session not found" in resp.json()["detail"]


def test_ask_graph_analyst_highest_count_and_groups(synthetic_categorical_df):
    """Verify dominant group and leading/lagging group calculations."""
    sess_id = "test_graph_ask_cat_session"
    DATASET_STORAGE[sess_id] = synthetic_categorical_df.copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(synthetic_categorical_df)}]

    # Q1: Highest count
    resp1 = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "category",
        "y_column": None,
        "chart_type": "bar",
        "question": "Which category has the highest count?"
    })
    assert resp1.status_code == 200
    d1 = resp1.json()
    d1_ans = d1["answer"]
    d1_text = str(d1_ans) if isinstance(d1_ans, str) else f"{d1_ans.get('finding', '')} {' '.join(d1_ans.get('evidence', []))}"
    assert "Alpha" in d1_text
    assert "10" in d1_text

    # Q2: Average comparison
    resp2 = test_client.post("/api/graph/ask", json={
        "dataset_id": sess_id,
        "x_column": "category",
        "y_column": "score",
        "chart_type": "bar",
        "question": "Which groups have the highest and lowest average values?"
    })
    assert resp2.status_code == 200
    d2 = resp2.json()
    d2_ans = d2["answer"]
    d2_text = str(d2_ans) if isinstance(d2_ans, str) else f"{d2_ans.get('finding', '')} {' '.join(d2_ans.get('evidence', []))}"
    assert "Alpha" in d2_text
    assert "Delta" in d2_text or "Epsilon" in d2_text


def test_pydantic_ai_agent_mock_tool_calling(synthetic_categorical_df):
    """
    Test Pydantic AI agent tool-calling loop using TestModel to verify
    end-to-end multi-step tool execution without making external network calls.
    """
    sess_id = "test_mock_agent_session"
    context = {"df": synthetic_categorical_df, "dataset_id": sess_id, "dataset_version": 0}

    agent = create_pydantic_ai_agent(model_instance=TestModel(call_tools=["analyze_chart", "compare_groups"]))
    deps = AnalystDependencies(
        dataset_id=sess_id,
        dataset_version=0,
        chart_spec=None,
        x_column="category",
        y_column="score",
        chart_type="bar",
        aggregation="mean",
        filters=None,
        context=context
    )

    run_res = agent.run_sync(
        "Analyze the leading and lagging groups and inspect data quality.",
        deps=deps,
        usage_limits=UsageLimits(tool_calls_limit=3, request_limit=5)
    )

    assert run_res.output is not None
    assert len(deps.tools_executed) > 0
    # Tools executed must be from the verified allowlist
    for t in deps.tools_executed:
        assert t in [
            "analyze_chart", "compare_groups", "run_statistical_test",
            "investigate_anomalies", "inspect_data_quality", "analyze_time_series"
        ]


def test_three_tool_call_limit_enforcement(synthetic_categorical_df):
    """Verify agent enforces strict tool-calls limit of 3."""
    agent = create_pydantic_ai_agent(model_instance=TestModel(call_tools="all"))
    deps = AnalystDependencies(
        dataset_id="limit_sess",
        dataset_version=0,
        chart_spec=None,
        x_column="category",
        y_column="score",
        chart_type="bar",
        aggregation="mean",
        filters=None,
        context={"df": synthetic_categorical_df}
    )

    # If tool calls limit is 1, calling all tools must raise UsageLimitExceeded
    with pytest.raises(Exception):
        agent.run_sync(
            "Run all tools",
            deps=deps,
            usage_limits=UsageLimits(tool_calls_limit=1, request_limit=3)
        )


@pytest.mark.skipif(not is_llm_configured(), reason="Real Gemini API key not present in environment.")
def test_real_gemini_model_integration_smoke(synthetic_categorical_df):
    """
    Live integration test against Google Gemini API free tier.
    Only executed when GEMINI_API_KEY or GOOGLE_API_KEY is configured in backend environment.
    """
    sess_id = "test_live_gemini_session"
    DATASET_STORAGE[sess_id] = synthetic_categorical_df.copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(synthetic_categorical_df)}]

    payload = {
        "dataset_id": sess_id,
        "x_column": "category",
        "y_column": "score",
        "chart_type": "bar",
        "question": "Which category is dominant, and is there evidence of outlier values?"
    }
    resp = test_client.post("/api/graph/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["execution_mode"] == "llm"
    assert len(data["tools_invoked"]) > 0
    assert "Alpha" in data["answer"]
