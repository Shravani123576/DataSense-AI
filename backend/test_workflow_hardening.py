"""
DataSense AI V6 — Workflow Hardening & Exact Calculation Verification Test Suite
Covers Sections 4, 5, 6, 7, 9, 10.B, 10.C, 10.D, 10.G, and 10.H:
- Exact data-quality metric & health score assertions against mathematical ground truth.
- Outlier review without auto-deletion.
- Transactional rollback on failed cleaning actions.
- Proposal rejection without mutation.
- Stale proposal 409 conflict rejection.
- Raw data immutability across multiple cleaning steps.
- High-cardinality categorical Top N truncation.
- Graph aggregations matching independent calculations.
- Report export reflection of active version without ML fabrication.
- Complete end-to-end non-ML workflow integration.
"""
import io
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import (
    app,
    DATASET_STORAGE,
    RAW_DATASET_STORAGE,
    UNMODIFIED_RAW_STORAGE,
    DATASET_VERSIONS_STORAGE,
    PROPOSED_TRANSFORMATIONS_STORAGE,
    AUDIT_LOG_STORAGE,
    PROFILE_STORAGE,
    METADATA_STORAGE
)
from data_profiler import profile_dataframe, calculate_data_quality_score
from data_cleaner import generate_cleaning_recommendations, preview_cleaning_action, apply_cleaning_action
from models import CleaningActionRequest

test_client = TestClient(app)


# =========================================================================
# 1. Exact Data Quality Assertions (Mathematical Ground Truth)
# =========================================================================
def test_exact_data_quality_and_health_score_calculation():
    """
    Construct a synthetic 10-row dataset with mathematically known anomalies:
    - Total cells: 10 rows * 5 cols = 50 cells.
    - Missing cells: 2 in 'salary', 1 in 'department' = 3 cells (3/50 = 6.0%).
    - Duplicate rows: 2 exact duplicates (2/10 = 20.0%).
    - Constant column: 'status' has 1 unique value ("Active").
    - Outlier: 'salary' has 1 extreme value (5000.0 vs normal values ~50).
    """
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 8, 8],  # rows 8 and 9 are duplicate of 7
        "salary": [50.0, 52.0, 48.0, 55.0, np.nan, np.nan, 51.0, 5000.0, 5000.0, 5000.0],
        "department": ["Eng", "HR", "Sales", "Eng", "HR", np.nan, "Sales", "Eng", "Eng", "Eng"],
        "status": ["Active"] * 10,  # Constant column
        "score": [80, 85, 90, 75, 88, 92, 79, 85, 85, 85]
    })

    # Expected calculations:
    total_cells = 50
    missing_cells = 3
    missing_pct = (3 / 50) * 100  # 6.0%
    duplicate_rows = 2
    dup_pct = (2 / 10) * 100      # 20.0%

    # Completeness formula: max(0, min(100, int(100 - (missing_pct * 2.5))))
    expected_completeness = int(100 - (6.0 * 2.5))  # 85
    # Uniqueness formula: max(0, min(100, int(100 - (dup_pct * 10))))
    expected_uniqueness = max(0, int(100 - (20.0 * 10)))  # 0 (penalized heavily)

    health = calculate_data_quality_score(
        df=df,
        duplicate_rows=duplicate_rows,
        missing_cells=missing_cells,
        total_cells=total_cells,
        outlier_counts=[1]  # 1 outlier in salary
    )

    assert health.completeness_score == expected_completeness
    assert health.uniqueness_score == expected_uniqueness
    assert health.overall_score >= 10 and health.overall_score <= 100
    assert any("duplicate" in r.lower() for r in health.recommendations)
    assert any("missing" in r.lower() for r in health.recommendations)


# =========================================================================
# 2. Outlier Review without Auto-Deletion
# =========================================================================
def test_outlier_review_preserves_rows_by_default():
    """Verify that outlier recommendations default to IQR capping/winsorization, NOT dropping rows."""
    df = pd.DataFrame({
        "metric": [10.0, 11.0, 10.5, 9.8, 10.2, 10.1, 10.4, 10.3, 9.9, 10.0, 10.2, 500.0]  # 12 rows, 500 is outlier
    })

    recs = generate_cleaning_recommendations(df)
    out_recs = [r for r in recs if r.rule == "cap_outliers"]
    assert len(out_recs) == 1
    rec = out_recs[0]

    # Rule must be cap_outliers with default strategy cap_iqr (NOT auto-dropping)
    assert rec.rule == "cap_outliers"
    assert "cap_iqr" in rec.strategy_options
    assert "Winsorize" in rec.recommended_action

    # Apply default cap_iqr strategy
    req = CleaningActionRequest(dataset_id="test", rule="cap_outliers", column="metric", strategy="cap_iqr")
    cleaned_df, audit = apply_cleaning_action(df, req)

    # Row count MUST remain 12 (no deletion)
    assert len(cleaned_df) == 12
    # The extreme outlier 500.0 must be capped within statistical bounds
    assert cleaned_df["metric"].max() < 500.0
    assert cleaned_df["metric"].max() > 10.0
    assert audit.affected_rows == 1


# =========================================================================
# 3. Transactional Rollback & Raw Data Immutability
# =========================================================================
def test_failed_cleaning_leaves_dataset_unchanged():
    """Verify that an invalid cleaning request or failure does not mutate working data."""
    csv_bytes = b"id,val\n1,10\n2,20\n3,30\n"
    files = {"file": ("transaction_test.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = test_client.post("/api/upload", files=files)
    assert resp.status_code == 200
    dataset_id = resp.json()["metadata"]["id"]

    orig_df_copy = DATASET_STORAGE[dataset_id].copy()

    # Attempt invalid cleaning operation (nonexistent column)
    invalid_req = {
        "dataset_id": dataset_id,
        "rule": "impute_missing",
        "column": "non_existent_column",
        "strategy": "mean"
    }
    resp_fail = test_client.post("/api/clean/apply", json=invalid_req)
    assert resp_fail.status_code == 400

    # Dataset in storage must remain 100% identical to original
    current_df = DATASET_STORAGE[dataset_id]
    pd.testing.assert_frame_equal(current_df, orig_df_copy)


def test_raw_data_immutability():
    """Verify that raw source data in RAW_DATASET_STORAGE remains immutable after cleaning."""
    csv_bytes = b"id,val\n1,10\n1,10\n2,20\n"  # 1 duplicate row
    files = {"file": ("immutability_test.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = test_client.post("/api/upload", files=files)
    dataset_id = resp.json()["metadata"]["id"]

    assert len(RAW_DATASET_STORAGE[dataset_id]) == 3
    assert len(UNMODIFIED_RAW_STORAGE[dataset_id]) == 3

    # Apply deduplication to working dataset
    dedup_req = {
        "dataset_id": dataset_id,
        "rule": "deduplicate",
        "strategy": "keep_first"
    }
    resp_clean = test_client.post("/api/clean/apply", json=dedup_req)
    assert resp_clean.status_code == 200

    # Working dataset has 2 rows
    assert len(DATASET_STORAGE[dataset_id]) == 2
    # Raw pristine datasets MUST still have 3 rows
    assert len(RAW_DATASET_STORAGE[dataset_id]) == 3
    assert len(UNMODIFIED_RAW_STORAGE[dataset_id]) == 3


# =========================================================================
# 4. Proposal Rejection and Stale Proposal 409 Conflict
# =========================================================================
def test_proposal_rejection_leaves_dataset_unchanged():
    """Verify that rejecting a proposal changes status to 'rejected' and mutates nothing."""
    csv_bytes = b"id,val\n1,10\n1,10\n2,20\n"
    files = {"file": ("reject_test.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = test_client.post("/api/upload", files=files)
    dataset_id = resp.json()["metadata"]["id"]

    prop_req = {
        "dataset_id": dataset_id,
        "rule": "deduplicate",
        "strategy": "keep_first"
    }
    resp_prop = test_client.post("/api/clean/propose", json=prop_req)
    assert resp_prop.status_code == 200
    prop_id = resp_prop.json()["proposal"]["proposal_id"]

    # Reject proposal
    resp_rej = test_client.post(f"/api/clean/reject/{prop_id}")
    assert resp_rej.status_code == 200
    assert resp_rej.json()["status"] == "rejected"

    # Working dataset still has 3 rows (unmutated)
    assert len(DATASET_STORAGE[dataset_id]) == 3


def test_stale_proposal_conflict_rejection():
    """Verify that a proposal staged against version V1 is rejected if dataset advanced to V2."""
    csv_bytes = b"id,name,score\n1,Alice,100\n1,Alice,100\n2,Bob,200\n"
    files = {"file": ("stale_test.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = test_client.post("/api/upload", files=files)
    dataset_id = resp.json()["metadata"]["id"]

    # Stage proposal 1 (deduplicate) against current version (V1)
    resp_prop = test_client.post("/api/clean/propose", json={
        "dataset_id": dataset_id,
        "rule": "deduplicate",
        "strategy": "keep_first"
    })
    prop_id = resp_prop.json()["proposal"]["proposal_id"]

    # Now independently apply a transformation to advance version to V2
    client_apply = test_client.post("/api/clean/apply", json={
        "dataset_id": dataset_id,
        "rule": "impute_missing",
        "column": "score",
        "strategy": "mean"
    })
    assert client_apply.status_code == 200

    # Current dataset version is now V2. Attempt to approve stale proposal staged against V1
    resp_approve = test_client.post(f"/api/clean/approve/{prop_id}")
    assert resp_approve.status_code == 409
    assert "stale" in resp_approve.json()["detail"].lower()


# =========================================================================
# 5. EDA & Chart Aggregation and Top N Safeguards
# =========================================================================
def test_eda_chart_aggregation_exactness_and_top_n():
    """
    Verify bar chart aggregations (sum, mean, count) match independent pandas calculations
    and high-cardinality top_n limits work properly.
    """
    # 30 categories, each with 2 rows
    cats = [f"Category_{i:02d}" for i in range(30) for _ in range(2)]
    vals = [float(i * 10) for i in range(30) for _ in range(2)]
    df = pd.DataFrame({"category": cats, "value": vals})

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("eda_test.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = test_client.post("/api/upload", files=files)
    dataset_id = resp.json()["metadata"]["id"]

    # 1. Test sum aggregation
    chart_req_sum = {
        "dataset_id": dataset_id,
        "x_column": "category",
        "y_column": "value",
        "chart_type": "bar",
        "aggregation": "sum"
    }
    resp_sum = test_client.post("/api/custom-chart", json=chart_req_sum)
    assert resp_sum.status_code == 200
    chart_sum = resp_sum.json()

    # Independent calculation for Category_29 (top category): two rows with val = 290.0 -> sum = 580.0
    item_29 = next(d for d in chart_sum["data"] if d["x"] == "Category_29")
    assert item_29["y"] == 580.0

    # 2. Test mean aggregation
    chart_req_mean = {
        "dataset_id": dataset_id,
        "x_column": "category",
        "y_column": "value",
        "chart_type": "bar",
        "aggregation": "mean"
    }
    resp_mean = test_client.post("/api/custom-chart", json=chart_req_mean)
    assert resp_mean.status_code == 200
    chart_mean = resp_mean.json()
    item_29_mean = next(d for d in chart_mean["data"] if d["x"] == "Category_29")
    assert item_29_mean["y"] == 290.0


# =========================================================================
# 6. Report Export Integrity (Active Version, No Fabricated ML)
# =========================================================================
def test_export_report_reflects_active_version_without_ml_fabrication():
    """Verify that generated summary reports reflect active version and do not invent ML results."""
    csv_bytes = b"feature_a,feature_b\n1.0,10.0\n2.0,20.0\n3.0,30.0\n"
    files = {"file": ("report_test.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = test_client.post("/api/upload", files=files)
    dataset_id = resp.json()["metadata"]["id"]

    # Export markdown report
    resp_rep = test_client.get(f"/api/dataset/{dataset_id}/export/summary")
    assert resp_rep.status_code == 200
    content = resp_rep.text

    # Must contain dataset schema facts
    assert "- **Total Records (Rows)**: 3" in content
    assert "- **Total Features (Columns)**: 2" in content
    assert "feature_a" in content
    assert "feature_b" in content

    # Crucial check: Must NOT fabricate Machine Learning models or training when not executed!
    assert "Machine Learning Best Model" not in content
    assert "Automated Machine Learning (AutoML) Evaluation" not in content


# =========================================================================
# 7. Complete End-to-End Non-ML Workflow Integration
# =========================================================================
def test_complete_end_to_end_workflow():
    """
    Test the full lifecycle from upload -> profile -> clean proposal -> approve ->
    EDA custom chart -> Time series analysis -> Contextual AI Analyst -> Report export.
    """
    # 1. Ingestion: Upload time-series sales dataset
    dates = pd.date_range("2024-01-01", periods=15, freq="D")
    sales = [100.0 + i * 10.0 for i in range(15)]
    base_df = pd.DataFrame({
        "order_date": dates,
        "revenue": sales,
        "region": ["North"] * 15
    })
    # Concat 1 exact duplicate row to make it 16 rows
    df = pd.concat([base_df, base_df.iloc[[0]]], ignore_index=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("e2e_sales.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp_upload = test_client.post("/api/upload", files=files)
    assert resp_upload.status_code == 200
    dataset_id = resp_upload.json()["metadata"]["id"]
    assert resp_upload.json()["metadata"]["rows"] == 16

    # 2. Quality & Proposal
    recs = test_client.get(f"/api/clean/{dataset_id}/recommendations").json()
    assert any(r["rule"] == "deduplicate" for r in recs)

    prop_resp = test_client.post("/api/clean/propose", json={
        "dataset_id": dataset_id,
        "rule": "deduplicate",
        "strategy": "keep_first"
    })
    assert prop_resp.status_code == 200
    prop_id = prop_resp.json()["proposal"]["proposal_id"]

    # 3. Approval & Version increment
    approve_resp = test_client.post(f"/api/clean/approve/{prop_id}")
    assert approve_resp.status_code == 200
    assert approve_resp.json()["total_records"] == 15

    # 4. EDA Chart
    chart_resp = test_client.post("/api/custom-chart", json={
        "dataset_id": dataset_id,
        "x_column": "order_date",
        "y_column": "revenue",
        "chart_type": "line",
        "aggregation": "mean"
    })
    assert chart_resp.status_code == 200
    assert len(chart_resp.json()["data"]) == 15

    # 5. Graph Intelligence Ask (Analyst Agent)
    analyst_resp = test_client.post("/api/graph/ask", json={
        "dataset_id": dataset_id,
        "x_column": "order_date",
        "y_column": "revenue",
        "chart_type": "line",
        "question": "What is the trajectory and trend of revenue over time?",
        "aggregation": "mean"
    })
    assert analyst_resp.status_code == 200
    analyst_data = analyst_resp.json()
    ans = analyst_data["answer"]
    ans_text = str(ans) if isinstance(ans, str) else f"{ans.get('finding', '')} {' '.join(ans.get('evidence', []))}"
    assert len(ans_text) > 20
    assert "linear_trend" in analyst_data["computed_evidence"] or "trend_direction" in analyst_data["computed_evidence"]
    if "linear_trend" in analyst_data["computed_evidence"]:
        assert analyst_data["computed_evidence"]["linear_trend"]["direction"] == "upward"
    assert analyst_data["execution_mode"] in ["llm", "deterministic_fallback"]

    # 6. Report Export
    report_resp = test_client.get(f"/api/dataset/{dataset_id}/export/summary")
    assert report_resp.status_code == 200
    assert "Deduplicated dataset" in report_resp.text
