"""
DataSense AI — Phase 3 Test Suite: Data Science Tool Layer Verification
Validates the central tool registry, core tools, statistical testing with known mathematical outcomes,
grounded chart analysis, ML isolation safeguards, cleaning approval workflows, and MCP adapter.
"""
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import (
    app,
    DATASET_STORAGE,
    DATASET_VERSIONS_STORAGE,
    PROPOSED_TRANSFORMATIONS_STORAGE,
    AUDIT_LOG_STORAGE,
    PROFILE_STORAGE,
    METADATA_STORAGE,
    LAST_ML_STORAGE,
    build_tool_context
)
from tools import default_registry, ToolExecutionResult, ToolExecutionStatus
from models import HypothesisTestRequest, CleaningProposal
import mcp_server


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def sample_session():
    """Seed an active dataset session with deterministic test data."""
    session_id = "test_phase3_session"
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "group": np.random.choice(["Alpha", "Beta", "Gamma"], size=n, p=[0.4, 0.4, 0.2]),
        "metric_a": np.random.normal(loc=50.0, scale=10.0, size=n),
        "metric_b": np.random.exponential(scale=5.0, size=n),
        "category_status": np.random.choice(["Active", "Inactive"], size=n, p=[0.7, 0.3]),
        "target_class": np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
        "target_reg": np.random.normal(loc=100.0, scale=15.0, size=n),
    })
    # Inject intentional outlier in metric_a
    df.loc[0, "metric_a"] = 150.0

    DATASET_STORAGE[session_id] = df.copy()
    DATASET_VERSIONS_STORAGE[session_id] = [{
        "version_id": 1,
        "parent_version_id": 0,
        "timestamp": "2026-10-26T00:00:00Z",
        "operation": "initial_load",
        "description": "Initial test dataset load",
        "affected_rows": 0,
        "affected_cols": len(df.columns),
        "total_rows": len(df),
        "total_cols": len(df.columns)
    }]
    PROPOSED_TRANSFORMATIONS_STORAGE[session_id] = {}
    AUDIT_LOG_STORAGE[session_id] = []
    METADATA_STORAGE[session_id] = {"name": "Phase 3 Test Data", "filename": "test_phase3.csv"}

    return session_id, df


# =====================================================================
# 1. TOOL REGISTRY CONTRACTS & DISCOVERY
# =====================================================================

def test_tool_registry_manifest_completeness():
    """Verify registry has registered all required core data science tools with valid schemas."""
    tools = default_registry.list_tools()
    tool_names = [t["name"] for t in tools]

    expected_tools = [
        # Dataset
        "profile_dataset", "inspect_data_quality", "get_dataset_schema", "get_dataset_version_history",
        # Cleaning
        "preview_cleaning_changes", "apply_approved_cleaning", "reject_cleaning_proposal",
        # Analysis
        "run_exploratory_analysis", "generate_chart", "analyze_chart", "compare_groups", "investigate_anomalies",
        # Statistical
        "run_statistical_test",
        # ML
        "prepare_modeling_task", "train_and_compare_models", "evaluate_final_model", "analyze_model_errors", "explain_feature_importance",
        # Reporting
        "generate_analysis_report", "export_analysis_artifact"
    ]

    for expected in expected_tools:
        assert expected in tool_names, f"Expected tool '{expected}' missing from registry."

    # Verify every tool has valid description and JSON schema
    for t in tools:
        assert len(t["description"]) > 10
        assert "properties" in t["input_schema"]
        assert t["category"] in ["dataset", "cleaning", "analysis", "statistical", "ml", "reporting"]
        assert t["permission"] in ["read_only", "mutates_data", "requires_approval"]


def test_tool_execution_unknown_tool():
    """Executing an unregistered tool must return a safe, structured error without crashing."""
    ctx = build_tool_context(None)
    res = default_registry.execute_tool("non_existent_tool_xyz", {}, ctx)
    assert res.status == ToolExecutionStatus.ERROR.value
    assert "Unknown tool" in res.error_details
    assert not res.mutated_data


def test_tool_execution_invalid_arguments(sample_session):
    """Passing malformed or missing required arguments must produce structured validation error."""
    session_id, _ = sample_session
    ctx = build_tool_context(session_id)
    # compare_groups requires group_column and metric_column
    res = default_registry.execute_tool("compare_groups", {"dataset_id": session_id}, ctx)
    assert res.status == ToolExecutionStatus.ERROR.value
    assert "validation error" in res.error_details.lower()


# =====================================================================
# 2. DATASET TOOLS
# =====================================================================

def test_dataset_tools_execution(sample_session):
    """Verify profile_dataset, inspect_data_quality, schema, and version history tools."""
    session_id, df = sample_session
    ctx = build_tool_context(session_id)

    # 1. profile_dataset
    res_prof = default_registry.execute_tool("profile_dataset", {"dataset_id": session_id}, ctx)
    assert res_prof.status == ToolExecutionStatus.SUCCESS.value
    assert res_prof.result["metadata"]["rows"] == len(df)
    assert "health" in res_prof.result

    # 2. inspect_data_quality
    res_qual = default_registry.execute_tool("inspect_data_quality", {"dataset_id": session_id}, ctx)
    assert res_qual.status == ToolExecutionStatus.SUCCESS.value
    assert res_qual.result["total_rows"] == len(df)
    assert "overall_health_score" in res_qual.result

    # 3. get_dataset_schema
    res_sch = default_registry.execute_tool("get_dataset_schema", {"dataset_id": session_id}, ctx)
    assert res_sch.status == ToolExecutionStatus.SUCCESS.value
    assert len(res_sch.result["columns"]) == len(df.columns)

    # 4. get_dataset_version_history
    res_ver = default_registry.execute_tool("get_dataset_version_history", {"dataset_id": session_id}, ctx)
    assert res_ver.status == ToolExecutionStatus.SUCCESS.value
    assert res_ver.result["current_version_id"] == 1
    assert len(res_ver.result["versions"]) == 1


# =====================================================================
# 3. ANALYSIS & CHART TOOLS (GROUNDED IN UNDERLYING DATA)
# =====================================================================

def test_analysis_tools_unknown_columns(sample_session):
    """Analysis tools must gracefully reject unknown columns with clean error messages."""
    session_id, _ = sample_session
    ctx = build_tool_context(session_id)

    # Unknown column in compare_groups
    res = default_registry.execute_tool("compare_groups", {
        "dataset_id": session_id,
        "group_column": "non_existent_column",
        "metric_column": "metric_a"
    }, ctx)
    assert res.status == ToolExecutionStatus.ERROR.value
    assert "not found" in res.error_details

    # Unknown column in investigate_anomalies
    res_anom = default_registry.execute_tool("investigate_anomalies", {
        "dataset_id": session_id,
        "column": "invalid_column"
    }, ctx)
    assert res_anom.status == ToolExecutionStatus.ERROR.value
    assert "does not exist" in res_anom.error_details


def test_investigate_anomalies_detects_outlier(sample_session):
    """Verify investigate_anomalies identifies the injected 150.0 outlier in metric_a."""
    session_id, df = sample_session
    ctx = build_tool_context(session_id)

    res = default_registry.execute_tool("investigate_anomalies", {
        "dataset_id": session_id,
        "column": "metric_a",
        "method": "iqr"
    }, ctx)
    assert res.status == ToolExecutionStatus.SUCCESS.value
    assert res.result["anomalies_found"] >= 1
    # Check that row 0 (the injected outlier) is detected
    detected_indices = [case["row_index"] for case in res.result["sample_outlier_cases"]]
    assert 0 in detected_indices


def test_compare_groups_independent_calculation(sample_session):
    """Verify compare_groups accurately matches independent Pandas groupby computations."""
    session_id, df = sample_session
    ctx = build_tool_context(session_id)

    res = default_registry.execute_tool("compare_groups", {
        "dataset_id": session_id,
        "group_column": "group",
        "metric_column": "metric_a"
    }, ctx)
    assert res.status == ToolExecutionStatus.SUCCESS.value

    # Verify against manual Pandas groupby
    expected_means = df.groupby("group")["metric_a"].mean().round(3).to_dict()
    for g_stat in res.result["group_statistics"]:
        grp_name = g_stat["group"]
        assert round(g_stat["mean"], 2) == round(expected_means[grp_name], 2)


def test_analyze_chart_uses_underlying_data(sample_session):
    """analyze_chart must compute group differences, dispersion, and caveats from raw data."""
    session_id, df = sample_session
    ctx = build_tool_context(session_id)

    res = default_registry.execute_tool("analyze_chart", {
        "dataset_id": session_id,
        "x_column": "group",
        "y_column": "metric_a",
        "chart_type": "bar"
    }, ctx)

    assert res.status == ToolExecutionStatus.SUCCESS.value
    findings = res.result["key_findings"]
    assert len(findings) > 0
    assert "causation_disclaimer" in res.result
    assert "not be interpreted as causal" in res.result["causation_disclaimer"]


# =====================================================================
# 4. STATISTICAL HYPOTHESIS TESTING (KNOWN MATHEMATICAL OUTCOMES)
# =====================================================================

def test_statistical_two_sample_t_known_significant():
    """Verify two-sample t-test detects statistically significant difference on known distinct distributions."""
    session_id = "test_ttest_session"
    np.random.seed(123)
    # Group A: N(10, 1), Group B: N(20, 1) -> Large true effect, p < 0.001
    s_a = np.random.normal(loc=10.0, scale=1.0, size=30)
    s_b = np.random.normal(loc=20.0, scale=1.0, size=30)
    df = pd.DataFrame({
        "treatment": ["A"] * 30 + ["B"] * 30,
        "outcome": np.concatenate([s_a, s_b])
    })

    DATASET_STORAGE[session_id] = df
    DATASET_VERSIONS_STORAGE[session_id] = [{"version_id": 1}]
    ctx = build_tool_context(session_id)

    res = default_registry.execute_tool("run_statistical_test", {
        "dataset_id": session_id,
        "test_type": "two_sample_t",
        "variable_x": "treatment",
        "variable_y": "outcome"
    }, ctx)

    assert res.status == ToolExecutionStatus.SUCCESS.value
    result = res.result
    assert result["is_statistically_significant"] is True
    assert result["p_value"] < 0.001
    assert result["effect_size_magnitude"] == "large"
    assert "Statistical significance does not prove real-world importance" in result["caveat_disclaimer"]
    assert len(result["assumptions"]) >= 2


def test_statistical_two_sample_t_known_non_significant():
    """Verify two-sample t-test correctly fails to reject null hypothesis for identical distributions."""
    session_id = "test_ttest_null"
    np.random.seed(999)
    # Both groups drawn from identical distribution N(50, 10)
    s_a = np.random.normal(loc=50.0, scale=10.0, size=30)
    s_b = np.random.normal(loc=50.0, scale=10.0, size=30)
    df = pd.DataFrame({
        "group": ["G1"] * 30 + ["G2"] * 30,
        "score": np.concatenate([s_a, s_b])
    })

    DATASET_STORAGE[session_id] = df
    DATASET_VERSIONS_STORAGE[session_id] = [{"version_id": 1}]
    ctx = build_tool_context(session_id)

    res = default_registry.execute_tool("run_statistical_test", {
        "dataset_id": session_id,
        "test_type": "two_sample_t",
        "variable_x": "group",
        "variable_y": "score"
    }, ctx)

    assert res.status == ToolExecutionStatus.SUCCESS.value
    result = res.result
    assert result["is_statistically_significant"] is False
    assert result["p_value"] >= 0.05


def test_statistical_chi_square_known_independence():
    """Verify Chi-Square test of independence on perfectly independent features."""
    session_id = "test_chi2_session"
    # Create perfectly balanced independent cross-tabulation
    df = pd.DataFrame({
        "gender": ["M"] * 50 + ["F"] * 50,
        "preference": (["Tea"] * 25 + ["Coffee"] * 25) * 2
    })

    DATASET_STORAGE[session_id] = df
    DATASET_VERSIONS_STORAGE[session_id] = [{"version_id": 1}]
    ctx = build_tool_context(session_id)

    res = default_registry.execute_tool("run_statistical_test", {
        "dataset_id": session_id,
        "test_type": "chi_square",
        "variable_x": "gender",
        "variable_y": "preference"
    }, ctx)

    assert res.status == ToolExecutionStatus.SUCCESS.value
    result = res.result
    assert result["is_statistically_significant"] is False
    assert result["test_statistic_value"] == 0.0
    assert result["p_value"] == 1.0


def test_statistical_pearson_correlation_known_linear():
    """Verify Pearson correlation test on deterministic linear relationship y = 3x."""
    session_id = "test_corr_session"
    x = np.linspace(1, 20, 20)
    y = 3.0 * x
    df = pd.DataFrame({"feature_x": x, "feature_y": y})

    DATASET_STORAGE[session_id] = df
    DATASET_VERSIONS_STORAGE[session_id] = [{"version_id": 1}]
    ctx = build_tool_context(session_id)

    res = default_registry.execute_tool("run_statistical_test", {
        "dataset_id": session_id,
        "test_type": "pearson_r",
        "variable_x": "feature_x",
        "variable_y": "feature_y"
    }, ctx)

    assert res.status == ToolExecutionStatus.SUCCESS.value
    result = res.result
    assert result["is_statistically_significant"] is True
    assert result["test_statistic_value"] == 1.0
    assert result["effect_size_magnitude"] == "large"


def test_statistical_insufficient_sample_size():
    """Verify statistical test rejects insufficient observations (< 5 rows) with clean error."""
    session_id = "test_small_session"
    df = pd.DataFrame({"a": [1, 2], "b": [10, 20]})

    DATASET_STORAGE[session_id] = df
    DATASET_VERSIONS_STORAGE[session_id] = [{"version_id": 1}]
    ctx = build_tool_context(session_id)

    res = default_registry.execute_tool("run_statistical_test", {
        "dataset_id": session_id,
        "test_type": "auto",
        "variable_x": "a",
        "variable_y": "b"
    }, ctx)

    assert res.status == ToolExecutionStatus.ERROR.value
    assert "Insufficient non-null observations" in res.error_details


# =====================================================================
# 5. CLEANING PROPOSAL & APPROVAL WORKFLOW
# =====================================================================

def test_cleaning_proposal_staging_and_approval_workflow(sample_session):
    """Verify proposal staging does not mutate data, and approval advances version with audit record."""
    session_id, df = sample_session
    ctx = build_tool_context(session_id)

    # 1. Preview changes (stages proposal)
    res_preview = default_registry.execute_tool("preview_cleaning_changes", {
        "dataset_id": session_id,
        "rule": "cap_outliers",
        "column": "metric_a",
        "strategy": "cap_iqr"
    }, ctx)

    assert res_preview.status in [ToolExecutionStatus.SUCCESS.value, ToolExecutionStatus.PENDING_APPROVAL.value]
    assert res_preview.mutated_data is False
    proposal_id = res_preview.result["proposal_id"]
    assert proposal_id in PROPOSED_TRANSFORMATIONS_STORAGE[session_id]

    # Verify active dataset has NOT mutated yet
    assert DATASET_STORAGE[session_id].loc[0, "metric_a"] == 150.0

    # 2. Commit approval
    res_apply = default_registry.execute_tool("apply_approved_cleaning", {
        "proposal_id": proposal_id
    }, ctx)

    assert res_apply.status == ToolExecutionStatus.SUCCESS.value
    assert res_apply.mutated_data is True
    assert res_apply.dataset_version == 2

    # Verify outlier was capped in active working dataset
    assert DATASET_STORAGE[session_id].loc[0, "metric_a"] < 150.0

    # Verify version advanced in storage
    assert len(DATASET_VERSIONS_STORAGE[session_id]) == 2
    assert DATASET_VERSIONS_STORAGE[session_id][-1]["version_id"] == 2


def test_cleaning_proposal_stale_version_rejected(sample_session):
    """A proposal created against version V1 must be rejected if the dataset advanced to V2 in the meantime."""
    session_id, _ = sample_session
    ctx = build_tool_context(session_id)

    # Stage proposal against Version 1
    res_preview = default_registry.execute_tool("preview_cleaning_changes", {
        "dataset_id": session_id,
        "rule": "normalize_categories",
        "column": "category_status",
        "strategy": "strip_and_titlecase"
    }, ctx)
    proposal_id = res_preview.result["proposal_id"]

    # Artificially advance active version to Version 2
    DATASET_VERSIONS_STORAGE[session_id].append({
        "version_id": 2,
        "parent_version_id": 1,
        "timestamp": "2026-10-26T01:00:00Z",
        "operation": "intermediate_change"
    })
    ctx["dataset_version"] = 2

    # Attempting to apply stale proposal must fail
    res_apply = default_registry.execute_tool("apply_approved_cleaning", {
        "proposal_id": proposal_id
    }, ctx)

    assert res_apply.status == ToolExecutionStatus.ERROR.value
    assert "Proposal is stale" in res_apply.error_details


def test_cleaning_proposal_rejection(sample_session):
    """Rejecting a proposal must mark it as rejected without mutating dataset."""
    session_id, _ = sample_session
    ctx = build_tool_context(session_id)

    res_preview = default_registry.execute_tool("preview_cleaning_changes", {
        "dataset_id": session_id,
        "rule": "drop_rows",
        "column": "category_status",
        "strategy": "drop_rows"
    }, ctx)
    proposal_id = res_preview.result["proposal_id"]

    res_reject = default_registry.execute_tool("reject_cleaning_proposal", {
        "proposal_id": proposal_id
    }, ctx)

    assert res_reject.status == ToolExecutionStatus.SUCCESS.value
    assert PROPOSED_TRANSFORMATIONS_STORAGE[session_id][proposal_id].status == "rejected"
    assert res_reject.mutated_data is False


# =====================================================================
# 6. MACHINE LEARNING SAFEGUARDS & ISOLATION
# =====================================================================

def test_ml_prepare_modeling_safeguards(sample_session):
    """prepare_modeling_task must detect task type and reject single-class targets."""
    session_id, df = sample_session
    ctx = build_tool_context(session_id)

    # 1. Valid task preparation
    res = default_registry.execute_tool("prepare_modeling_task", {
        "dataset_id": session_id,
        "target_column": "target_class"
    }, ctx)
    assert res.status == ToolExecutionStatus.SUCCESS.value
    assert res.result["inferred_task_type"] == "classification"

    # 2. Reject single class
    df_single = df.copy()
    df_single["constant_col"] = "OnlyClass"
    DATASET_STORAGE[session_id] = df_single
    ctx_single = build_tool_context(session_id)
    res_err = default_registry.execute_tool("prepare_modeling_task", {
        "dataset_id": session_id,
        "target_column": "constant_col"
    }, ctx_single)
    assert res_err.status == ToolExecutionStatus.ERROR.value
    assert "only 1 unique class" in res_err.error_details


def test_ml_train_evaluate_and_explain(sample_session):
    """Run AutoML tournament through tool layer and verify final test isolation and error analysis."""
    session_id, df = sample_session
    ctx = build_tool_context(session_id)

    # 1. Train and compare models
    res_train = default_registry.execute_tool("train_and_compare_models", {
        "dataset_id": session_id,
        "target_column": "target_class",
        "task_type": "classification"
    }, ctx)
    assert res_train.status == ToolExecutionStatus.SUCCESS.value
    assert "best_model_name" in res_train.result

    # 2. Evaluate final model
    res_eval = default_registry.execute_tool("evaluate_final_model", {
        "dataset_id": session_id
    }, ctx)
    assert res_eval.status == ToolExecutionStatus.SUCCESS.value
    assert "final_independent_test_score" in res_eval.result
    assert "Strict final test-set isolation enforced" in res_eval.result["isolation_guarantee"]

    # 3. Analyze model errors
    res_errors = default_registry.execute_tool("analyze_model_errors", {
        "dataset_id": session_id
    }, ctx)
    assert res_errors.status == ToolExecutionStatus.SUCCESS.value
    assert "worst_prediction_cases" in res_errors.result

    # 4. Explain feature importance
    res_feat = default_registry.execute_tool("explain_feature_importance", {
        "dataset_id": session_id
    }, ctx)
    assert res_feat.status == ToolExecutionStatus.SUCCESS.value
    assert len(res_feat.result["features_ranked"]) > 0


# =====================================================================
# 7. REPORTING TOOLS
# =====================================================================

def test_reporting_tools(sample_session):
    """Verify generate_analysis_report and export_analysis_artifact produce verified content."""
    session_id, _ = sample_session
    ctx = build_tool_context(session_id)

    # 1. generate_analysis_report
    res_rep = default_registry.execute_tool("generate_analysis_report", {
        "dataset_id": session_id,
        "objective": "Customer segmentation"
    }, ctx)
    assert res_rep.status == ToolExecutionStatus.SUCCESS.value
    assert len(res_rep.result["executive_summary"]) > 0

    # 2. export_analysis_artifact (Markdown & HTML)
    res_md = default_registry.execute_tool("export_analysis_artifact", {
        "dataset_id": session_id,
        "artifact_format": "markdown"
    }, ctx)
    assert res_md.status == ToolExecutionStatus.SUCCESS.value
    assert res_md.result["mime_type"] == "text/markdown"

    res_html = default_registry.execute_tool("export_analysis_artifact", {
        "dataset_id": session_id,
        "artifact_format": "html"
    }, ctx)
    assert res_html.status == ToolExecutionStatus.SUCCESS.value
    assert res_html.result["mime_type"] == "text/html"


# =====================================================================
# 8. FASTAPI ENDPOINTS E2E VERIFICATION
# =====================================================================

def test_fastapi_tool_endpoints(test_client, sample_session):
    """Verify HTTP endpoints /api/tools/list, /api/tools/execute, /api/statistical/test, /api/graph/ask."""
    session_id, _ = sample_session

    # 1. GET /api/tools/list
    resp_list = test_client.get("/api/tools/list")
    assert resp_list.status_code == 200
    tools = resp_list.json()
    assert len(tools) >= 20

    # 2. POST /api/tools/execute
    resp_exec = test_client.post("/api/tools/execute", json={
        "tool_name": "get_dataset_schema",
        "arguments": {"dataset_id": session_id},
        "dataset_id": session_id
    })
    assert resp_exec.status_code == 200
    assert resp_exec.json()["status"] == "success"

    # 3. POST /api/statistical/test
    resp_stat = test_client.post("/api/statistical/test", json={
        "dataset_id": session_id,
        "test_type": "auto",
        "variable_x": "group",
        "variable_y": "metric_a"
    })
    assert resp_stat.status_code == 200
    stat_data = resp_stat.json()
    assert "test_name" in stat_data
    assert "caveat_disclaimer" in stat_data

    # 4. POST /api/graph/analyze
    resp_analyze = test_client.post("/api/graph/analyze", json={
        "dataset_id": session_id,
        "x_column": "group",
        "y_column": "metric_a",
        "chart_type": "bar",
        "chart_spec": {}
    })
    assert resp_analyze.status_code == 200
    assert len(resp_analyze.json()["key_findings"]) > 0

    # 5. POST /api/graph/ask
    resp_ask = test_client.post("/api/graph/ask", json={
        "dataset_id": session_id,
        "x_column": "group",
        "y_column": "metric_a",
        "chart_type": "bar",
        "chart_spec": {},
        "question": "Which groups differ the most and are there any unusual points?"
    })
    assert resp_ask.status_code == 200
    ask_data = resp_ask.json()
    ans = ask_data["answer"]
    ans_text = str(ans) if isinstance(ans, str) else f"{ans.get('finding', '')} {' '.join(ans.get('evidence', []))}"
    assert len(ans_text) > 10
    assert "Group Comparison" in ans_text or "differ" in ans_text.lower() or "group" in ans_text.lower()
    assert "disclaimer" in ask_data


# =====================================================================
# 9. MCP PROTOCOL-LEVEL VERIFICATION
# =====================================================================

def test_mcp_protocol_level_integration(sample_session):
    """Verify MCP protocol responses for initialize, tools/list, and tools/call using underlying tools."""
    session_id, _ = sample_session

    # 1. MCP tool manifest
    manifest = mcp_server.get_mcp_tool_manifest()
    assert len(manifest) >= 20
    tool_names = [t["name"] for t in manifest]
    assert "run_statistical_test" in tool_names
    assert "profile_dataset" in tool_names

    # 2. FastMCP server instantiation
    fastmcp_app = mcp_server.create_fastmcp_server()
    assert fastmcp_app.name == "datasense-ai"

    # 3. Direct execution routing via registry context
    ctx = build_tool_context(session_id)
    exec_res = default_registry.execute_tool("profile_dataset", {"dataset_id": session_id}, ctx)
    assert exec_res.status == "success"
    assert exec_res.result["metadata"]["rows"] == 100
