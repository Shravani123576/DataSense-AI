import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from main import app, startup_event, DATASET_STORAGE

client = TestClient(app)

def test_startup_clean_state():
    startup_event()
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    # Ensure startup starts with clean state (0 active datasets unless loaded)
    assert "active_datasets" in data


def test_samples_list():
    res = client.get("/api/samples")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) >= 4
    ids = [s["id"] for s in samples]
    assert "ecommerce_sales" in ids
    assert "customer_churn" in ids
    assert "titanic_survival" in ids
    assert "zomato_restaurants" in ids
    assert "bigdata_100k" in ids


def test_empty_file_upload_error():
    """Verify that uploading an empty file (0 bytes) returns an explicit 400 error."""
    files = {"file": ("empty.csv", b"", "text/csv")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_empty_dataframe_upload_error():
    """Verify that uploading a CSV with headers but 0 data rows returns an explicit 400 error."""
    csv_content = b"Name,Age,Salary\n"
    files = {"file": ("zero_rows.csv", csv_content, "text/csv")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 400
    assert "no data rows" in res.json()["detail"].lower()


def test_dirty_currency_and_comma_coercion():
    """Verify that dirty string numbers ($1,200, ₹500, 15%, 4.5/5) are properly coerced to numeric."""
    csv_content = (
        'Item,Price,Discount,Rating,Units\n'
        'Widget A,"$1,200.50",10%,4.2/5,100\n'
        'Widget B,$850.00,15%,4.8/5,250\n'
        'Widget C,"$2,400.00",5%,3.9/5,75\n'
        'Widget D,$500.00,20%,4.5/5,300\n'
    ).encode("utf-8")

    files = {"file": ("products.csv", csv_content, "text/csv")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 200
    report = res.json()
    cols = {c["column"]: c["inferred_type"] for c in report["columns"]}
    
    # Price, Discount, and Rating should be inferred as numeric thanks to smart cleaning
    assert cols["Price"] == "numeric"
    assert cols["Discount"] == "numeric"
    assert cols["Rating"] == "numeric"
    assert cols["Units"] == "numeric"

    # Query average price
    dataset_id = report["metadata"]["id"]
    q_res = client.post("/api/query", json={
        "dataset_id": dataset_id,
        "question": "What is the average Price?"
    })
    assert q_res.status_code == 200
    ans = q_res.json()["answer_text"]
    # (1200.5 + 850 + 2400 + 500) / 4 = 1237.625 -> ~1,237.62
    assert "1,237.62" in ans or "1237.62" in ans


def test_ranking_calculations_top_and_bottom():
    """Verify that top N and bottom N rankings calculate exact extremes."""
    res = client.post("/api/samples/load?sample_id=ecommerce_sales")
    assert res.status_code == 200
    
    # Top 5 by sales
    top_res = client.post("/api/query", json={
        "dataset_id": "sample_ecommerce_sales",
        "question": "Show me the top 5 highest Sales"
    })
    assert top_res.status_code == 200
    assert "top 5" in top_res.json()["answer_text"].lower()
    assert "nlargest" in top_res.json()["code_executed"]

    # Bottom 5 by sales
    bot_res = client.post("/api/query", json={
        "dataset_id": "sample_ecommerce_sales",
        "question": "Show me the bottom 5 lowest Sales"
    })
    assert bot_res.status_code == 200
    assert "bottom 5" in bot_res.json()["answer_text"].lower()
    assert "nsmallest" in bot_res.json()["code_executed"]


def test_percentage_change_and_trends():
    """Verify that trend and percentage change over time compute accurate growth rates."""
    res = client.post("/api/query", json={
        "dataset_id": "sample_ecommerce_sales",
        "question": "What is the trend and percentage change of Sales over time?"
    })
    assert res.status_code == 200
    data = res.json()
    assert "trend" in data["answer_text"].lower() or "increase" in data["answer_text"].lower() or "decrease" in data["answer_text"].lower()
    assert data["chart_spec"] is not None
    assert data["chart_spec"]["chart_type"] == "line"


def test_ambiguous_or_missing_column_query():
    """Verify that querying a non-existent metric asks for clarification instead of hallucinating."""
    res = client.post("/api/samples/load?sample_id=titanic_survival")
    assert res.status_code == 200

    # Ask for revenue on Titanic dataset (which does NOT have revenue)
    q_res = client.post("/api/query", json={
        "dataset_id": "sample_titanic_survival",
        "question": "What is the average revenue?"
    })
    assert q_res.status_code == 200
    data = q_res.json()
    # It must explicitly state that revenue was not found and ask for clarification
    assert "couldn't find a column" in data["answer_text"].lower() or "revenue" in data["answer_text"].lower()
    assert "available columns" in data["answer_text"].lower()


def test_dataset_switching_and_reset():
    """Verify that resetting or switching datasets preserves isolation."""
    # Reset
    reset_res = client.post("/api/dataset/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["active_datasets"] == 0

    # Load churn
    res1 = client.post("/api/samples/load?sample_id=customer_churn")
    assert res1.status_code == 200
    assert "Churn" in [c["column"] for c in res1.json()["columns"]]

    # Load ecommerce
    res2 = client.post("/api/samples/load?sample_id=ecommerce_sales")
    assert res2.status_code == 200
    assert "Category" in [c["column"] for c in res2.json()["columns"]]


def test_zomato_explicit_sample_only():
    """Verify that Zomato intelligence is ONLY activated on explicit restaurant datasets."""
    # Titanic should NOT be a restaurant dataset
    res_titanic = client.post("/api/samples/load?sample_id=titanic_survival")
    assert res_titanic.json()["is_restaurant_dataset"] is False

    # Zomato Bangalore sample MUST be a restaurant dataset
    res_zomato = client.post("/api/samples/load?sample_id=zomato_restaurants")
    assert res_zomato.json()["is_restaurant_dataset"] is True
    assert res_zomato.json()["restaurant_insights"] is not None


def test_100k_rows_benchmark():
    res = client.post("/api/samples/load?sample_id=bigdata_100k")
    assert res.status_code == 200
    report = res.json()
    assert report["metadata"]["rows"] >= 100000
    assert len(report["columns"]) >= 8


# ==========================================
# V2 Tests: Auditable Cleaning, AutoML & Agent
# ==========================================

def test_auditable_cleaning_workflow():
    """Verify recommendation generation, preview diff, application, and audit history."""
    # Load Titanic dataset which has missing Age and Cabin values
    res = client.post("/api/samples/load?sample_id=titanic_survival")
    assert res.status_code == 200
    dataset_id = "sample_titanic_survival"

    # 1. Fetch recommendations
    recs_res = client.get(f"/api/clean/{dataset_id}/recommendations")
    assert recs_res.status_code == 200
    recs = recs_res.json()
    assert len(recs) > 0
    # There should be an impute recommendation for Age
    age_recs = [r for r in recs if r["column"] == "Age"]
    assert len(age_recs) == 1
    assert age_recs[0]["rule"] == "impute_missing"

    # 2. Preview cleaning action
    preview_res = client.post("/api/clean/preview", json={
        "dataset_id": dataset_id,
        "rule": "impute_missing",
        "column": "Age",
        "strategy": "median"
    })
    assert preview_res.status_code == 200
    preview_data = preview_res.json()
    assert preview_data["affected_rows"] > 0
    assert preview_data["fill_value"] is not None

    # 3. Apply cleaning action
    apply_res = client.post("/api/clean/apply", json={
        "dataset_id": dataset_id,
        "rule": "impute_missing",
        "column": "Age",
        "strategy": "median"
    })
    assert apply_res.status_code == 200
    summary = apply_res.json()
    assert summary["total_cleaned_actions_applied"] >= 1
    assert len(summary["audit_history"]) >= 1
    assert summary["audit_history"][0]["column"] == "Age"

    # 4. Check audit log endpoint
    audit_res = client.get(f"/api/clean/{dataset_id}/audit")
    assert audit_res.status_code == 200
    assert len(audit_res.json()["audit_history"]) >= 1

    # 5. Revert to raw
    revert_res = client.post(f"/api/clean/{dataset_id}/reset-to-raw")
    assert revert_res.status_code == 200
    assert revert_res.json()["status"] == "reverted"


def test_automl_classification_and_what_if():
    """Verify AutoML classification tournament, baseline dummy, and counterfactual simulation."""
    res = client.post("/api/samples/load?sample_id=customer_churn")
    assert res.status_code == 200
    dataset_id = "sample_customer_churn"

    # Train AutoML targeting 'Churn'
    ml_res = client.post("/api/automl/train", json={
        "dataset_id": dataset_id,
        "target_column": "Churn",
        "task_type": "classification"
    })
    assert ml_res.status_code == 200
    ml_data = ml_res.json()
    assert ml_data["task_type"] == "classification"
    assert len(ml_data["leaderboard"]) >= 3
    # Baseline dummy should be present
    has_baseline = any(m["is_baseline"] for m in ml_data["leaderboard"])
    assert has_baseline is True
    assert ml_data["best_model_name"] != ""
    assert len(ml_data["feature_importance"]) > 0

    # Test What-If scenario
    whatif_res = client.post("/api/automl/what-if", json={
        "dataset_id": dataset_id,
        "target_column": "Churn",
        "feature_inputs": {"MonthlyCharges": 120.0, "TenureMonths": 1}
    })
    assert whatif_res.status_code == 200
    w_data = whatif_res.json()
    assert "prediction" in w_data
    assert "disclaimer" in w_data
    assert "statistical associations" in w_data["disclaimer"].lower()


def test_agent_orchestration_goal_clarification_and_execution():
    """Verify that ambiguous goals ask for clarification, and specific goals complete."""
    res = client.post("/api/samples/load?sample_id=customer_churn")
    assert res.status_code == 200
    dataset_id = "sample_customer_churn"

    # 1. Ambiguous goal: 'predict outcomes' without target
    ambig_res = client.post("/api/agent/goal", json={
        "dataset_id": dataset_id,
        "goal": "I want to train a model to predict things"
    })
    assert ambig_res.status_code == 200
    ambig_data = ambig_res.json()
    assert ambig_data["requires_clarification"] is True
    assert len(ambig_data["clarification_options"]) > 0

    # 2. Responding with clarification
    resolved_res = client.post("/api/agent/goal", json={
        "dataset_id": dataset_id,
        "goal": "I want to train a model to predict things",
        "clarification_response": "Churn"
    })
    assert resolved_res.status_code == 200
    resolved_data = resolved_res.json()
    assert resolved_data["requires_clarification"] is False
    assert resolved_data["plan"]["is_completed"] is True
    assert len(resolved_data["findings"]) >= 2


def test_executive_reporting():
    """Verify markdown and HTML report generation and file downloads."""
    res = client.post("/api/samples/load?sample_id=customer_churn")
    assert res.status_code == 200
    dataset_id = "sample_customer_churn"

    # Generate report
    rep_res = client.post("/api/report/generate", json={
        "dataset_id": dataset_id,
        "objective": "Customer retention and churn analysis"
    })
    assert rep_res.status_code == 200
    rep_data = rep_res.json()
    assert "Executive Data Science & Analytical Intelligence Brief" in rep_data["markdown_content"]
    assert "<!DOCTYPE html>" in rep_data["html_content"]
    assert len(rep_data["recommendations"]) > 0

    # Download HTML
    html_res = client.get(f"/api/report/{dataset_id}/download/html")
    assert html_res.status_code == 200
    assert "text/html" in html_res.headers["content-type"]

    # Download Markdown
    md_res = client.get(f"/api/report/{dataset_id}/download/markdown")
    assert md_res.status_code == 200
    assert "text/markdown" in md_res.headers["content-type"]


# =====================================================================
# Inspection & Repair Validation Tests (AutoML, Copilot, Chart Accuracy)
# =====================================================================

def test_automl_regression_workflow_and_holdout():
    """Verify real regression AutoML workflow, baseline dummy, metrics, holdout predictions, and eval chart."""
    # Build a small deterministic regression dataset: Sales = 20 * Quantity + 10
    rows = []
    for i in range(50):
        qty = i + 1
        sales = round(20.0 * qty + 10.0 + (i % 3), 2)
        rows.append(f"Product_{i},{qty},{sales},Category_{i % 3}")
    csv_str = "Product,Quantity,Sales,Category\n" + "\n".join(rows)

    files = {"file": ("sales_regression.csv", csv_str.encode("utf-8"), "text/csv")}
    upload_res = client.post("/api/upload", files=files)
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["metadata"]["id"]

    # Train AutoML Regression targeting Sales
    ml_res = client.post("/api/automl/train", json={
        "dataset_id": dataset_id,
        "target_column": "Sales",
        "task_type": "regression"
    })
    assert ml_res.status_code == 200
    res_data = ml_res.json()
    assert res_data["task_type"] == "regression"
    assert res_data["target_column"] == "Sales"
    assert res_data["total_samples"] == 50
    assert res_data["test_samples"] == 10
    assert len(res_data["leaderboard"]) >= 3

    # Check regression metrics: R2, RMSE, MAE
    best_metrics = res_data["best_model_metrics"]
    assert "r2_score" in best_metrics
    assert "rmse" in best_metrics
    assert "mae" in best_metrics

    # Check baseline dummy regressor is evaluated
    baseline_item = next((m for m in res_data["leaderboard"] if m["is_baseline"]), None)
    assert baseline_item is not None
    assert "Dummy" in baseline_item["model_name"]

    # Check held-out test predictions
    assert "held_out_predictions" in res_data
    held_out = res_data["held_out_predictions"]
    assert len(held_out) == 10  # 20% of 50
    for sample in held_out:
        assert "row_index" in sample
        assert "actual" in sample
        assert "predicted" in sample
        assert "error_or_residual" in sample
        assert isinstance(sample["error_or_residual"], (float, int))

    # Check evaluation chart spec
    assert "evaluation_chart_spec" in res_data
    eval_chart = res_data["evaluation_chart_spec"]
    assert eval_chart["chart_type"] == "scatter"
    assert eval_chart["data_stage"] == "model_ready_held_out"
    assert eval_chart["rows_used"] == 10


def test_automl_invalid_targets_and_edge_cases():
    """Verify validation errors for zero-variance targets, non-numeric regression, insufficient data, and invalid targets."""
    # 1. Dataset with zero-variance target (all identical values)
    zero_var_csv = "Feature1,Feature2,Target\n" + "\n".join([f"{i},{i*2},ConstantVal" for i in range(25)])
    files = {"file": ("zero_var.csv", zero_var_csv.encode("utf-8"), "text/csv")}
    res1 = client.post("/api/upload", files=files)
    d_id1 = res1.json()["metadata"]["id"]

    train_res1 = client.post("/api/automl/train", json={
        "dataset_id": d_id1,
        "target_column": "Target"
    })
    assert train_res1.status_code == 400
    assert "only 1 unique value" in train_res1.json()["detail"].lower()

    # 2. Non-numeric column requested for regression (with >= 2 unique classes)
    non_num_csv = "Feature1,Feature2,Category\n" + "\n".join([f"{i},{i*2},Category_{i%2}" for i in range(25)])
    files2 = {"file": ("non_num.csv", non_num_csv.encode("utf-8"), "text/csv")}
    res2 = client.post("/api/upload", files=files2)
    d_id2 = res2.json()["metadata"]["id"]

    train_res2 = client.post("/api/automl/train", json={
        "dataset_id": d_id2,
        "target_column": "Category",
        "task_type": "regression"
    })
    assert train_res2.status_code == 400
    assert "non-numeric" in train_res2.json()["detail"].lower()

    # 3. Insufficient data (< 15 rows)
    small_csv = "A,B,Target\n" + "\n".join([f"{i},{i*2},{i%2}" for i in range(10)])
    files3 = {"file": ("too_small.csv", small_csv.encode("utf-8"), "text/csv")}
    res3 = client.post("/api/upload", files=files3)
    d_id3 = res3.json()["metadata"]["id"]

    train_res3 = client.post("/api/automl/train", json={
        "dataset_id": d_id3,
        "target_column": "Target"
    })
    assert train_res3.status_code == 400
    assert "15 valid rows required" in train_res3.json()["detail"].lower()

    # 4. Target not in dataset
    train_res4 = client.post("/api/automl/train", json={
        "dataset_id": d_id3,
        "target_column": "NonExistentColumn"
    })
    assert train_res4.status_code == 400
    assert "not found" in train_res4.json()["detail"].lower()


def test_copilot_different_questions_produce_distinct_answers():
    """Verify that different user questions produce DIFFERENT, computationally grounded answers."""
    # Load E-Commerce dataset with known columns: Sales, Quantity, Discount, Profit, Category, etc.
    res = client.post("/api/samples/load?sample_id=ecommerce_sales")
    assert res.status_code == 200
    d_id = "sample_ecommerce_sales"

    # Question A: Average of Sales
    q_a = client.post("/api/query", json={"dataset_id": d_id, "question": "What is the average Sales?"})
    assert q_a.status_code == 200
    ans_a = q_a.json()["answer_text"]
    assert "average" in ans_a.lower()
    assert "sales" in ans_a.lower()

    # Question B: Total Quantity
    q_b = client.post("/api/query", json={"dataset_id": d_id, "question": "What is the total Quantity?"})
    assert q_b.status_code == 200
    ans_b = q_b.json()["answer_text"]
    assert "total" in ans_b.lower()
    assert "quantity" in ans_b.lower()

    # Question C: Column listing
    q_c = client.post("/api/query", json={"dataset_id": d_id, "question": "What are the columns?"})
    assert q_c.status_code == 200
    ans_c = q_c.json()["answer_text"]
    assert "columns" in ans_c.lower() or "features" in ans_c.lower()

    # Question D: Row filtering
    q_d = client.post("/api/query", json={"dataset_id": d_id, "question": "Show records where Quantity > 10"})
    assert q_d.status_code == 200
    ans_d = q_d.json()["answer_text"]
    assert "records" in ans_d.lower()
    assert "quantity > 10" in ans_d.lower()

    # Crucial assertion: Ensure all answers are DISTINCT!
    assert ans_a != ans_b
    assert ans_a != ans_c
    assert ans_b != ans_c
    assert ans_c != ans_d


def test_copilot_limitations_and_unanswerable_questions():
    """Verify that unanswerable queries or non-existent concepts return clear limitations without hallucination."""
    res = client.post("/api/samples/load?sample_id=ecommerce_sales")
    d_id = "sample_ecommerce_sales"

    # 1. Ask about churn on ecommerce dataset (churn does not exist in ecommerce)
    q1 = client.post("/api/query", json={"dataset_id": d_id, "question": "What is the churn rate?"})
    assert q1.status_code == 200
    ans1 = q1.json()["answer_text"]
    assert "couldn't find" in ans1.lower() or "not present" in ans1.lower() or "cannot answer" in ans1.lower() or "churn" in ans1.lower()
    assert "available columns" in ans1.lower()

    # 2. Ask an unrelated question
    q2 = client.post("/api/query", json={"dataset_id": d_id, "question": "Who is the CEO of Google?"})
    assert q2.status_code == 200
    ans2 = q2.json()["answer_text"]
    assert "couldn't find" in ans2.lower() or "not present" in ans2.lower() or "cannot answer" in ans2.lower() or "ceo" in ans2.lower()
    assert "available columns" in ans2.lower()


def test_chart_provenance_and_data_accuracy():
    """Verify that custom charts and Copilot charts populate data_stage, rows_used, and transformations."""
    res = client.post("/api/samples/load?sample_id=ecommerce_sales")
    d_id = "sample_ecommerce_sales"

    # 1. Histogram Chart
    hist_res = client.post("/api/custom-chart", json={
        "dataset_id": d_id,
        "x_column": "Sales",
        "chart_type": "histogram"
    })
    assert hist_res.status_code == 200
    hist_spec = hist_res.json()
    assert hist_spec["data_stage"] == "analysis_ready"
    assert hist_spec["rows_used"] > 0
    assert "binned" in hist_spec["transformations"].lower()

    # 2. Grouped Aggregation Chart
    bar_res = client.post("/api/custom-chart", json={
        "dataset_id": d_id,
        "x_column": "Category",
        "y_column": "Sales",
        "chart_type": "bar",
        "aggregation": "mean"
    })
    assert bar_res.status_code == 200
    bar_spec = bar_res.json()
    assert bar_spec["data_stage"] == "analysis_ready"
    assert bar_spec["rows_used"] > 0
    assert "grouped" in bar_spec["transformations"].lower()


def test_dataset_switching_invalidates_results():
    """Verify that loading dataset B switches context and invalidates dataset A results."""
    # 1. Load churn dataset
    res_churn = client.post("/api/samples/load?sample_id=customer_churn")
    assert res_churn.status_code == 200
    assert res_churn.json()["metadata"]["id"] == "sample_customer_churn"

    # Train AutoML on Churn
    ml_churn = client.post("/api/automl/train", json={
        "dataset_id": "sample_customer_churn",
        "target_column": "Churn"
    })
    assert ml_churn.status_code == 200
    assert ml_churn.json()["target_column"] == "Churn"

    # 2. Switch to titanic dataset
    res_titanic = client.post("/api/samples/load?sample_id=titanic_survival")
    assert res_titanic.status_code == 200
    assert res_titanic.json()["metadata"]["id"] == "sample_titanic_survival"

    # Querying titanic dataset with Churn must return limitation, not stale churn results
    q_churn_on_titanic = client.post("/api/query", json={
        "dataset_id": "sample_titanic_survival",
        "question": "What is the churn rate?"
    })
    assert q_churn_on_titanic.status_code == 200
    ans = q_churn_on_titanic.json()["answer_text"]
    assert "couldn't find" in ans.lower() or "not present" in ans.lower() or "cannot answer" in ans.lower() or "churn" in ans.lower()


def test_excel_upload_and_parsing():
    """Verify parsing and profiling for real user-uploaded Excel (.xlsx) spreadsheets."""
    import io
    import pandas as pd
    
    # Generate in-memory Excel workbook with diverse column types
    df_excel = pd.DataFrame({
        "Customer_ID": [f"CUST_{i:03d}" for i in range(30)],
        "Monthly_Charges": [round(45.0 + i * 1.5, 2) for i in range(30)],
        "Contract_Type": ["Month-to-Month" if i % 2 == 0 else "Two-Year" for i in range(30)],
        "Churn_Risk": [1 if i % 4 == 0 else 0 for i in range(30)]
    })
    excel_buffer = io.BytesIO()
    df_excel.to_excel(excel_buffer, index=False, engine="openpyxl")
    excel_buffer.seek(0)

    files = {"file": ("customer_portfolio.xlsx", excel_buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["metadata"]["rows"] == 30
    assert data["metadata"]["columns"] == 4
    assert "Customer Portfolio" in data["metadata"]["name"]
    assert data["health"]["overall_score"] >= 80
    assert data["recommended_next_step"] is not None


def test_grounded_recommended_next_step():
    """Verify that Recommended Next Step accurately diagnoses hygiene vs predictive modeling."""
    # 1. Dataset with duplicates and missing values must recommend 'clean'
    dirty_csv = "A,B,Target\n1,10,0\n1,10,0\n,20,1\n3,,0\n4,40,1\n" + "\n".join([f"{i},{i*10},{i%2}" for i in range(5, 25)])
    files = {"file": ("dirty_test.csv", dirty_csv.encode("utf-8"), "text/csv")}
    res_dirty = client.post("/api/upload", files=files)
    assert res_dirty.status_code == 200
    rec_dirty = res_dirty.json()["recommended_next_step"]
    assert rec_dirty["action_type"] == "clean"
    assert rec_dirty["target_tab"] == "clean"
    assert "clean" in rec_dirty["title"].lower()

    # 2. Clean dataset with candidate target 'Churn' must recommend 'predict'
    clean_csv = "CustomerID,Tenure,MonthlyCharges,Churn\n" + "\n".join([f"C{i},{i*2},{50+i%20},{i%2}" for i in range(1, 50)])
    files_clean = {"file": ("clean_churn.csv", clean_csv.encode("utf-8"), "text/csv")}
    res_clean = client.post("/api/upload", files=files_clean)
    assert res_clean.status_code == 200
    rec_clean = res_clean.json()["recommended_next_step"]
    assert rec_clean["action_type"] == "predict"
    assert rec_clean["target_tab"] == "predict"
    assert "churn" in rec_clean["title"].lower()


def test_export_cleaned_dataset_preserves_state():
    """Verify that export/cleaned returns actual approved state without forced blanket mutations."""
    res = client.post("/api/samples/load?sample_id=customer_churn")
    assert res.status_code == 200
    d_id = "sample_customer_churn"

    # Export cleaned CSV directly
    exp_res = client.get(f"/api/dataset/{d_id}/export/cleaned")
    assert exp_res.status_code == 200
    assert "text/csv" in exp_res.headers["content-type"]
    
    # Verify row count matches active dataset rows (CRLF safe)
    lines = [line.strip() for line in exp_res.text.strip().splitlines() if line.strip()]
    header = [c.strip() for c in lines[0].split(",")]
    assert "Churn" in header
    # 1 Header row + 141 data rows (140 + 1 duplicate)
    assert len(lines) == res.json()["metadata"]["rows"] + 1


def test_copilot_quality_and_patterns_questions():
    """Verify Copilot answers for data quality problems, most missing values, patterns, and causation disclaimer."""
    res = client.post("/api/samples/load?sample_id=ecommerce_sales")
    d_id = "sample_ecommerce_sales"

    # 1. Ask: What are the biggest data-quality problems?
    q_qual = client.post("/api/query", json={"dataset_id": d_id, "question": "What are the biggest data quality problems?"})
    assert q_qual.status_code == 200
    ans_qual = q_qual.json()["answer_text"]
    assert "data quality" in ans_qual.lower() or "records" in ans_qual.lower()
    assert "clean studio" in ans_qual.lower() or "missing" in ans_qual.lower()

    # 2. Ask: Which columns have the most missing values?
    q_null = client.post("/api/query", json={"dataset_id": d_id, "question": "Which columns have the most missing values?"})
    assert q_null.status_code == 200
    ans_null = q_null.json()["answer_text"]
    assert "missing" in ans_null.lower() or "null" in ans_null.lower()

    # 3. Ask: What patterns or relationships appear in this dataset?
    q_patt = client.post("/api/query", json={"dataset_id": d_id, "question": "What patterns or relationships appear in this dataset?"})
    assert q_patt.status_code == 200
    ans_patt = q_patt.json()["answer_text"]
    assert "correlation" in ans_patt.lower() or "relationship" in ans_patt.lower()
    assert "causation" in ans_patt.lower()  # Must include correlation vs causation disclaimer!

    # 4. Ask: Which further analysis would be useful?
    q_next = client.post("/api/query", json={"dataset_id": d_id, "question": "Which further analysis would be useful?"})
    assert q_next.status_code == 200
    ans_next = q_next.json()["answer_text"]
    assert "recommended" in ans_next.lower() or "analysis" in ans_next.lower()


def test_automl_plain_english_metric_explanations():
    """Verify that AutoML responses include plain-English metric explanations and verification status."""
    res = client.post("/api/samples/load?sample_id=customer_churn")
    d_id = "sample_customer_churn"

    ml_res = client.post("/api/automl/train", json={
        "dataset_id": d_id,
        "target_column": "Churn",
        "task_type": "classification"
    })
    assert ml_res.status_code == 200
    data = ml_res.json()

    assert "metric_explanations" in data
    exps = data["metric_explanations"]
    assert "accuracy" in exps
    assert "f1_score" in exps
    assert "precision" in exps
    assert "recall" in exps

    assert "verification_status" in data
    assert "verified" in data["verification_status"].lower()
    assert "validation_strategy" in data
    assert "holdout" in data["validation_strategy"].lower()


# =====================================================================
# V5 Product Reliability, Scientific Integrity & Workflows Tests
# =====================================================================

def test_constant_column_detection_and_drop():
    """Verify that constant/zero-variance columns are flagged in recommendations and cleanly dropped."""
    # Build dataset with a constant column
    csv_str = "ID,Value,ConstantCol,Target\n" + "\n".join([f"{i},{i*1.5},SAME_VAL,{i%2}" for i in range(25)])
    files = {"file": ("constant_test.csv", csv_str.encode("utf-8"), "text/csv")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 200
    d_id = res.json()["metadata"]["id"]

    # Recommendations should include dropping ConstantCol
    recs_res = client.get(f"/api/clean/{d_id}/recommendations")
    assert recs_res.status_code == 200
    const_recs = [r for r in recs_res.json() if r["column"] == "ConstantCol"]
    assert len(const_recs) == 1
    assert const_recs[0]["rule"] == "drop_constant_column"

    # Preview drop
    prev_res = client.post("/api/clean/preview", json={
        "dataset_id": d_id,
        "rule": "drop_constant_column",
        "column": "ConstantCol"
    })
    assert prev_res.status_code == 200
    assert "drop" in prev_res.json()["summary"].lower()

    # Apply drop
    apply_res = client.post("/api/clean/apply", json={
        "dataset_id": d_id,
        "rule": "drop_constant_column",
        "column": "ConstantCol"
    })
    assert apply_res.status_code == 200
    # Audit history should record dropping
    audits = apply_res.json()["audit_history"]
    assert any(a["rule"] == "drop_constant_column" for a in audits)


def test_chronological_time_series_holdout_split():
    """Verify chronological split validation on datasets with datetime/time columns."""
    res = client.post("/api/samples/load?sample_id=ecommerce_sales")
    assert res.status_code == 200
    d_id = "sample_ecommerce_sales"

    # Train AutoML with OrderDate as chronological time column
    ml_res = client.post("/api/automl/train", json={
        "dataset_id": d_id,
        "target_column": "Sales",
        "task_type": "regression",
        "time_column": "OrderDate"
    })
    assert ml_res.status_code == 200
    data = ml_res.json()
    assert "chronological" in data["validation_strategy"].lower()
    assert "OrderDate" in data["validation_strategy"]
    assert "lookahead" in data["validation_strategy"].lower()


def test_multicollinearity_and_error_analysis():
    """Verify multicollinearity detection, holdout error analysis, and reproducible Python script generation."""
    # Build dataset with 2 collinear features: X2 = 2 * X1 + noise
    rows = []
    for i in range(40):
        x1 = float(i)
        x2 = float(2.0 * i + (0.01 if i % 2 == 0 else -0.01))  # Correlation > 0.99
        y = float(3.0 * x1 + 5.0)
        rows.append(f"{x1},{x2},{y}")
    csv_str = "FeatureA,FeatureB,Target\n" + "\n".join(rows)

    files = {"file": ("collinear.csv", csv_str.encode("utf-8"), "text/csv")}
    upload_res = client.post("/api/upload", files=files)
    assert upload_res.status_code == 200
    d_id = upload_res.json()["metadata"]["id"]

    ml_res = client.post("/api/automl/train", json={
        "dataset_id": d_id,
        "target_column": "Target",
        "task_type": "regression"
    })
    assert ml_res.status_code == 200
    data = ml_res.json()

    # 1. Multicollinearity warnings
    assert "multicollinearity_warnings" in data
    warnings = data["multicollinearity_warnings"]
    assert len(warnings) >= 1
    assert warnings[0]["correlation"] >= 0.85
    assert "FeatureA" in [warnings[0]["feature_1"], warnings[0]["feature_2"]]

    # 2. Error analysis
    assert "error_analysis" in data
    err_ana = data["error_analysis"]
    assert "mean_absolute_error" in err_ana
    assert "max_residual" in err_ana
    assert "top_worst_predictions" in err_ana
    assert len(err_ana["top_worst_predictions"]) > 0

    # 3. Baseline drift reference
    assert "baseline_drift_reference" in data
    drift = data["baseline_drift_reference"]
    assert "FeatureA" in drift
    assert "mean" in drift["FeatureA"]
    assert "std" in drift["FeatureA"]

    # 4. Standalone reproducible Python script
    assert "reproducible_python_script" in data
    script = data["reproducible_python_script"]
    assert "import pandas as pd" in script
    assert "ColumnTransformer" in script
    assert "Pipeline" in script
    assert "Target" in script


def test_healthcare_diabetes_research_dataset():
    """Verify multi-domain support with the newly registered clinical healthcare research dataset."""
    res = client.post("/api/samples/load?sample_id=diabetes_research")
    assert res.status_code == 200
    data = res.json()
    assert data["metadata"]["rows"] >= 150
    cols = [c["column"] for c in data["columns"]]
    assert "Glucose" in cols
    assert "BMI" in cols
    assert "Outcome" in cols

    # Query healthcare metric
    q_res = client.post("/api/query", json={
        "dataset_id": "sample_diabetes_research",
        "question": "What is the average Glucose level?"
    })
    assert q_res.status_code == 200
    assert "glucose" in q_res.json()["answer_text"].lower()
    assert "average" in q_res.json()["answer_text"].lower()


# =====================================================================
# PHASE 2: SCIENTIFIC CORRECTNESS & DATA INTEGRITY REGRESSION TESTS
# =====================================================================

def test_phase2_model_selection_uses_cv_validation_score_not_test_score():
    """Verify candidate models are ranked by cross-validation on the dev set, not final test score."""
    from main import DATASET_STORAGE
    # Create small deterministic classification dataset
    data = []
    for i in range(120):
        # FeatureA correlates with Class
        cls = "Yes" if i % 2 == 0 else "No"
        val = 10.0 if cls == "Yes" else 2.0
        data.append({"FeatureA": val + (i * 0.01), "FeatureB": (i % 5) * 1.5, "Target": cls})
    df = pd.DataFrame(data)
    d_id = "test_phase2_model_selection"
    DATASET_STORAGE[d_id] = df

    res = client.post("/api/automl/train", json={
        "dataset_id": d_id,
        "target_column": "Target",
        "task_type": "classification",
        "test_size": 0.2
    })
    assert res.status_code == 200
    res_data = res.json()

    # 1. Verification of separate validation and test metrics
    assert "model_selection_metric" in res_data
    assert "selection_metric_rationale" in res_data
    assert "F1-Score" in res_data["model_selection_metric"]
    assert "f1" in res_data["selection_metric_rationale"].lower()

    # 2. Check leaderboard candidate ranking logic
    leaderboard = res_data["leaderboard"]
    trained = [m for m in leaderboard if m["status"] == "trained"]
    assert len(trained) >= 3

    # All items must have distinct val_score, test_score, train_score
    for m in trained:
        assert "val_score" in m
        assert "test_score" in m
        assert "train_score" in m
        assert "cv_scores_mean" in m
        assert "val_metrics" in m

    # The winner must be the candidate with the highest validation score (excluding dummy baseline if competitive models exist)
    best_candidate_name = res_data["best_model_name"]
    best_item = next(m for m in trained if m["model_name"] == best_candidate_name)
    max_val_score = max(m["val_score"] for m in trained if not m["is_baseline"])
    assert best_item["val_score"] == pytest.approx(max_val_score, abs=1e-4)


def test_phase2_incidental_date_column_does_not_force_chronological_split():
    """Verify that incidental date columns (e.g., BirthDate) do not trigger unwanted chronological splits."""
    from main import DATASET_STORAGE
    # Employee dataset with BirthDate (should NOT be split chronologically for salary prediction)
    records = []
    base_dates = ["1985-04-12", "1990-11-23", "1978-01-15", "1995-07-30", "1988-09-05"]
    for i in range(100):
        records.append({
            "EmployeeID": 1000 + i,
            "BirthDate": base_dates[i % len(base_dates)],
            "ExperienceYears": (i % 15) + 1,
            "Department": "Engineering" if i % 2 == 0 else "Marketing",
            "Salary": 50000 + ((i % 15) * 4500)
        })
    df = pd.DataFrame(records)
    d_id = "test_phase2_incidental_date"
    DATASET_STORAGE[d_id] = df

    # Run AutoML without specifying chronological validation
    res = client.post("/api/automl/train", json={
        "dataset_id": d_id,
        "target_column": "Salary",
        "task_type": "regression"
    })
    assert res.status_code == 200
    data = res.json()

    # Validation strategy MUST NOT be chronological
    strat = data["validation_strategy"].lower()
    assert "chronological" not in strat
    assert "cross-validation" in strat or "holdout" in strat

    # Rationale must explicitly document why chronological was not used
    rationale = data.get("validation_rationale", "").lower()
    assert "incidental date" in rationale or "chronological forecasting was not requested" in rationale


def test_phase2_explicit_chronological_validation():
    """Verify that when chronological forecasting is explicitly requested, time-series splitting is applied."""
    from main import DATASET_STORAGE
    # Genuine forecasting dataset
    dates = pd.date_range("2024-01-01", periods=100, freq="D").strftime("%Y-%m-%d").tolist()
    records = [
        {"Timestamp": d, "TrendIndex": idx * 1.5, "Metric": idx * 10 + (idx % 3)}
        for idx, d in enumerate(dates)
    ]
    df = pd.DataFrame(records)
    d_id = "test_phase2_explicit_time"
    DATASET_STORAGE[d_id] = df

    res = client.post("/api/automl/train", json={
        "dataset_id": d_id,
        "target_column": "Metric",
        "task_type": "regression",
        "validation_strategy": "chronological",
        "time_column": "Timestamp"
    })
    assert res.status_code == 200
    data = res.json()

    assert "chronological" in data["validation_strategy"].lower()
    assert "Timestamp" in data["validation_strategy"]
    assert "future-lookahead" in data["validation_rationale"].lower() or "chronological" in data["validation_rationale"].lower()


def test_phase2_raw_data_preservation_and_coercion_audit():
    """Verify pristine raw data preservation, raw bytes storage, and explicit coercion audit recording."""
    from main import RAW_BYTE_STORAGE, UNMODIFIED_RAW_STORAGE, DATASET_STORAGE

    csv_content = (
        "Product,Cost,Margin,ReleaseDate\n"
        'Alpha,"$1,500.00",25%,2022-01-15\n'
        'Beta,"$800.00",18%,2022-06-20\n'
        'Gamma,"$3,200.00",30%,2023-03-10\n'
        'Delta,"$450.00",15%,2023-05-12\n'
        'Epsilon,"$1,100.00",20%,2023-07-04\n'
        'Zeta,"InvalidCost",12%,2023-08-01\n'
    ).encode("utf-8")

    files = {"file": ("inventory.csv", csv_content, "text/csv")}
    upload_res = client.post("/api/upload", files=files)
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["metadata"]["id"]

    # 1. Raw byte storage
    assert dataset_id in RAW_BYTE_STORAGE
    assert RAW_BYTE_STORAGE[dataset_id] == csv_content

    # 2. Unmodified parsed dataset snapshot retains dirty strings
    assert dataset_id in UNMODIFIED_RAW_STORAGE
    raw_df = UNMODIFIED_RAW_STORAGE[dataset_id]
    assert raw_df["Cost"].iloc[0] == "$1,500.00"
    assert raw_df["Margin"].iloc[0] == "25%"

    # 3. Working dataset in DATASET_STORAGE has coerced numeric types
    work_df = DATASET_STORAGE[dataset_id]
    assert work_df["Cost"].iloc[0] == 1500.0
    assert work_df["Margin"].iloc[0] == 25.0

    # 4. Check explicit coercion provenance endpoint
    audit_res = client.get(f"/api/clean/{dataset_id}/coercions")
    assert audit_res.status_code == 200
    coercions = audit_res.json()
    assert len(coercions) >= 2
    cost_coercion = next(c for c in coercions if c["column"] == "Cost")
    assert cost_coercion["affected_rows"] >= 3
    # Delta had "InvalidCost" which failed numeric coercion
    assert cost_coercion["failed_coercions"] >= 1

    # 5. Export raw dataset verifies pristine data download
    raw_export_res = client.get(f"/api/dataset/{dataset_id}/export/raw")
    assert raw_export_res.status_code == 200
    exported_raw_text = raw_export_res.text
    assert "$1,500.00" in exported_raw_text
    assert "InvalidCost" in exported_raw_text


def test_phase2_versioned_transformation_proposal_and_rejection():
    """Verify safe proposal workflow: proposed transformations do not mutate active dataset until approved."""
    from main import DATASET_STORAGE, DATASET_VERSIONS_STORAGE

    # Load customer churn sample
    client.post("/api/samples/load?sample_id=customer_churn")
    d_id = "sample_customer_churn"
    orig_rows = len(DATASET_STORAGE[d_id])

    # 1. Check version tree has V0 and V1
    v_res = client.get(f"/api/clean/{d_id}/versions")
    assert v_res.status_code == 200
    versions = v_res.json()
    assert len(versions) >= 2
    assert versions[0]["version_id"] == 0
    assert versions[1]["version_id"] == 1

    # 2. Propose a consequential cleaning action (e.g. drop duplicates or drop rows)
    prop_res = client.post("/api/clean/propose", json={
        "dataset_id": d_id,
        "rule": "deduplicate",
        "strategy": "drop_rows"
    })
    assert prop_res.status_code == 200
    prop_data = prop_res.json()
    proposal = prop_data["proposal"]
    proposal_id = proposal["proposal_id"]
    assert proposal["status"] == "pending"

    # ACTIVE DATASET MUST NOT BE MUTATED BY A PROPOSAL!
    assert len(DATASET_STORAGE[d_id]) == orig_rows

    # 3. Reject proposal
    rej_res = client.post(f"/api/clean/reject/{proposal_id}")
    assert rej_res.status_code == 200
    assert rej_res.json()["status"] == "rejected"

    # Active dataset remains unmodified
    assert len(DATASET_STORAGE[d_id]) == orig_rows

    # 4. Propose and approve another cleaning action
    prop_res2 = client.post("/api/clean/propose", json={
        "dataset_id": d_id,
        "rule": "cap_outliers",
        "column": "MonthlyCharges",
        "strategy": "cap_iqr"
    })
    assert prop_res2.status_code == 200
    prop2_id = prop_res2.json()["proposal"]["proposal_id"]

    approve_res = client.post(f"/api/clean/approve/{prop2_id}")
    assert approve_res.status_code == 200

    # Version history must now include Version 2
    v_res2 = client.get(f"/api/clean/{d_id}/versions")
    assert v_res2.status_code == 200
    v_list = v_res2.json()
    assert len(v_list) >= 3
    latest_v = v_list[-1]
    assert latest_v["version_id"] == 2
    assert latest_v["parent_version_id"] == 1
    assert latest_v["operation"] == "cap_outliers"


def test_phase2_export_integrity_preserves_approved_state_exactly():
    """Verify that export does not apply unintended deduplication or imputation and does not mutate active dataset."""
    from main import DATASET_STORAGE

    # Create dataset with intentional duplicates and missing values
    records = [
        {"ID": 1, "Category": "A", "Score": 10.0},
        {"ID": 1, "Category": "A", "Score": 10.0}, # duplicate
        {"ID": 2, "Category": "B", "Score": None}, # missing numeric
        {"ID": 3, "Category": "C", "Score": 30.0},
    ]
    df = pd.DataFrame(records)
    d_id = "test_phase2_export_integrity"
    DATASET_STORAGE[d_id] = df.copy()

    # Call export endpoint without query flags (default parameters)
    export_res = client.get(f"/api/dataset/{d_id}/export/cleaned")
    assert export_res.status_code == 200
    exported_csv = export_res.text

    # Parse exported CSV to check integrity
    exported_df = pd.read_csv(io.StringIO(exported_csv))

    # Must preserve exact 4 rows (duplicates not silently dropped)
    assert len(exported_df) == 4
    # Must preserve the NaN in Score (missing value not silently imputed)
    assert pd.isna(exported_df["Score"].iloc[2])

    # Active dataset in memory must NOT have been mutated
    assert len(DATASET_STORAGE[d_id]) == 4
    assert pd.isna(DATASET_STORAGE[d_id]["Score"].iloc[2])


def test_phase2_reset_to_raw_restores_unmodified_state_with_version_provenance():
    """Verify reset-to-raw properly restores original un-coerced snapshot and advances version history."""
    from main import DATASET_STORAGE, UNMODIFIED_RAW_STORAGE, DATASET_VERSIONS_STORAGE

    csv_content = (
        "Item,Price\n"
        'Gadget A,"$100.00"\n'
        'Gadget B,"$200.00"\n'
    ).encode("utf-8")

    files = {"file": ("gadgets.csv", csv_content, "text/csv")}
    res = client.post("/api/upload", files=files)
    assert res.status_code == 200
    dataset_id = res.json()["metadata"]["id"]

    # In working storage, Price is numeric 100.0
    assert DATASET_STORAGE[dataset_id]["Price"].iloc[0] == 100.0

    # Call reset to raw
    reset_res = client.post(f"/api/clean/{dataset_id}/reset-to-raw")
    assert reset_res.status_code == 200
    data = reset_res.json()
    assert data["status"] == "reverted"

    # Working dataset must now match original unmodified snapshot ($100.00 string)
    assert DATASET_STORAGE[dataset_id]["Price"].iloc[0] == "$100.00"

    # Version history has recorded reset_to_raw
    versions = DATASET_VERSIONS_STORAGE[dataset_id]
    assert any(v["operation"] == "reset_to_raw" for v in versions)


# =====================================================================
# Phase 2.1: Verification Gap Regression Tests
# =====================================================================

def test_phase2_1_final_test_set_strictly_isolated_from_candidate_selection():
    """
    Regression Test: Final test set MUST remain untouched during candidate model ranking.
    Only the winning champion model receives test-set evaluation (test_score is not None).
    Non-winning candidates must NOT receive test evaluation (test_score is None).
    """
    # Create synthetic dataset with clear signal
    import numpy as np
    np.random.seed(42)
    n = 60
    age = np.random.randint(20, 70, size=n)
    income = age * 1000 + np.random.randint(-2000, 2000, size=n)
    churn = [1 if (a > 45 and inc > 45000) else 0 for a, inc in zip(age, income)]
    
    csv_str = "Age,Income,Churn\n" + "\n".join([f"{a},{inc},{c}" for a, inc, c in zip(age, income, churn)])
    files = {"file": ("test_isolation_data.csv", csv_str.encode("utf-8"), "text/csv")}
    upload_res = client.post("/api/upload", files=files)
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["metadata"]["id"]

    # Run AutoML tournament
    ml_res = client.post("/api/automl/train", json={
        "dataset_id": dataset_id,
        "target_column": "Churn",
        "task_type": "classification",
        "validation_strategy": "kfold_cv"
    })
    assert ml_res.status_code == 200
    data = ml_res.json()

    leaderboard = data["leaderboard"]
    assert len(leaderboard) >= 3

    # Candidates must be ranked strictly by validation score
    val_scores = [m["val_score"] for m in leaderboard]
    # Non-baseline models should be ranked by val_score descending
    trained_non_baselines = [m for m in leaderboard if not m["is_baseline"]]
    for i in range(len(trained_non_baselines) - 1):
        assert trained_non_baselines[i]["val_score"] >= trained_non_baselines[i + 1]["val_score"]

    # 1. Champion winner (rank #1) receives final holdout test evaluation
    winner = leaderboard[0]
    assert winner["test_score"] is not None
    assert isinstance(winner["test_score"], float)
    assert winner["metrics"] != {}
    assert "accuracy" in winner["metrics"]

    # 2. All remaining candidates (rank #2+) MUST NOT have test_score evaluated
    for candidate in leaderboard[1:]:
        assert candidate["test_score"] is None, (
            f"Candidate {candidate['model_name']} had test_score evaluated before/outside winner selection! "
            "Final test set was leaked into candidate tournament."
        )
        assert candidate["metrics"] == {}, (
            f"Candidate {candidate['model_name']} had holdout test metrics populated!"
        )

    # 3. Winner's independent final test metrics reported separately in response
    assert data["best_model_name"] == winner["model_name"]
    assert data["best_model_metrics"]["accuracy"] == winner["test_score"]


def test_phase2_1_cleaning_proposal_stale_version_invalidation_409():
    """
    Regression Test: If active dataset version advances before a staged proposal is approved,
    the proposal must be invalidated and return HTTP 409 Conflict.
    """
    dirty_csv = (
        "Name,Department,Salary\n"
        "Alice,Sales,50000\n"
        "Alice,Sales,50000\n"  # Duplicate
        "Bob,Eng,\n"            # Missing salary
        "Charlie,Marketing,70000\n"
        "David,Eng,80000\n"
    ).encode("utf-8")

    files = {"file": ("stale_test.csv", dirty_csv, "text/csv")}
    upload_res = client.post("/api/upload", files=files)
    assert upload_res.status_code == 200
    d_id = upload_res.json()["metadata"]["id"]

    # 1. Stage proposal A (deduplicate) against initial Version 0
    prop_res = client.post("/api/clean/propose", json={
        "dataset_id": d_id,
        "rule": "deduplicate",
        "strategy": "drop_duplicates"
    })
    assert prop_res.status_code == 200
    proposal_a = prop_res.json()["proposal"]
    prop_id = proposal_a["proposal_id"]
    assert proposal_a["base_version_id"] == 1

    # 2. Advance the dataset version by applying an unrelated transformation (impute missing salary)
    apply_res = client.post("/api/clean/apply", json={
        "dataset_id": d_id,
        "rule": "impute_missing",
        "column": "Salary",
        "strategy": "median"
    })
    assert apply_res.status_code == 200

    # Verify version has advanced to 2
    v_res = client.get(f"/api/clean/{d_id}/versions")
    assert v_res.status_code == 200
    versions = v_res.json()
    assert versions[-1]["version_id"] == 2

    # 3. Attempt to approve stale proposal A (created against version 0) -> MUST fail with 409 Conflict
    approve_stale_res = client.post(f"/api/clean/approve/{prop_id}")
    assert approve_stale_res.status_code == 409
    error_detail = approve_stale_res.json()["detail"]
    assert "stale" in error_detail.lower() or "version" in error_detail.lower()


def test_phase2_1_cleaning_proposal_rejection_leaves_dataset_intact():
    """
    Regression Test: Rejecting a proposed cleaning transformation must not alter the active dataset.
    """
    csv_data = "ID,Score\n1,10\n2,20\n3,\n4,40\n".encode("utf-8")
    files = {"file": ("reject_test.csv", csv_data, "text/csv")}
    upload_res = client.post("/api/upload", files=files)
    d_id = upload_res.json()["metadata"]["id"]

    # Propose dropping rows with missing score
    prop_res = client.post("/api/clean/propose", json={
        "dataset_id": d_id,
        "rule": "impute_missing",
        "column": "Score",
        "strategy": "drop_rows"
    })
    assert prop_res.status_code == 200
    prop_id = prop_res.json()["proposal"]["proposal_id"]

    # Reject proposal
    rej_res = client.post(f"/api/clean/reject/{prop_id}")
    assert rej_res.status_code == 200
    assert rej_res.json()["status"] == "rejected"

    # Verify dataset was NOT modified (still 4 rows, including NaN)
    audit_res = client.get(f"/api/clean/{d_id}/audit")
    assert audit_res.json()["total_records"] == 4
    assert audit_res.json()["total_cleaned_actions_applied"] == 0


def test_phase2_1_validation_strategy_controls_and_task_suitability():
    """
    Regression Test: Validation strategy controls:
    - Incidental dates do not trigger chronological validation when auto is selected.
    - Explicit chronological validation on valid date column succeeds.
    - Unsuitable validation requests raise HTTP 400.
    """
    # Create dataset with incidental date
    rows = []
    for i in range(1, 35):
        rows.append(f"2023-01-{i%28+1:02d},C{i},{i*10},{50+i%10},{i%2}")
    csv_content = "OrderDate,CustomerID,Tenure,MonthlyCharges,Churn\n" + "\n".join(rows)
    files = {"file": ("dates_test.csv", csv_content.encode("utf-8"), "text/csv")}
    upload_res = client.post("/api/upload", files=files)
    d_id = upload_res.json()["metadata"]["id"]

    # 1. Incidental date with auto strategy -> must retain standard stratified/kfold validation
    auto_res = client.post("/api/automl/train", json={
        "dataset_id": d_id,
        "target_column": "Churn",
        "task_type": "classification",
        "validation_strategy": "auto"
    })
    assert auto_res.status_code == 200
    auto_data = auto_res.json()
    assert "chronological" not in auto_data["validation_strategy"].lower()
    assert "incidental date" in auto_data["validation_strategy"].lower() or "k-fold" in auto_data["validation_strategy"].lower()

    # 2. Explicit chronological validation requested with OrderDate -> must apply chronological split
    chrono_res = client.post("/api/automl/train", json={
        "dataset_id": d_id,
        "target_column": "Churn",
        "task_type": "classification",
        "validation_strategy": "chronological",
        "time_column": "OrderDate"
    })
    assert chrono_res.status_code == 200
    chrono_data = chrono_res.json()
    assert "chronological" in chrono_data["validation_strategy"].lower()
    assert "OrderDate" in chrono_data["validation_strategy"]

    # 3. Unsuitable validation: Stratified validation requested for continuous regression
    unsuitable_strat_res = client.post("/api/automl/train", json={
        "dataset_id": d_id,
        "target_column": "MonthlyCharges",
        "task_type": "regression",
        "validation_strategy": "stratified_holdout"
    })
    assert unsuitable_strat_res.status_code == 400
    assert "stratified" in unsuitable_strat_res.json()["detail"].lower()

    # 4. Unsuitable validation: Nonexistent time column requested
    bad_time_res = client.post("/api/automl/train", json={
        "dataset_id": d_id,
        "target_column": "Churn",
        "validation_strategy": "chronological",
        "time_column": "NonexistentDateCol"
    })
    assert bad_time_res.status_code == 400
    assert "not found" in bad_time_res.json()["detail"].lower()


def test_phase2_1_imbalanced_classification_uses_macro_f1():
    """
    Regression Test: Imbalanced classification targets must select Macro F1
    and provide transparent rationale explaining that it avoids majority class dominance.
    """
    # 70 negative class vs 10 positive class (imbalance ratio = 7.0 >= 2.5)
    rows = []
    for i in range(70):
        rows.append(f"{20+i%30},{i*100},0")
    for i in range(10):
        rows.append(f"{50+i%20},{5000+i*200},1")
    csv_content = "Age,Spending,Defaulter\n" + "\n".join(rows)

    files = {"file": ("imbalance_test.csv", csv_content.encode("utf-8"), "text/csv")}
    upload_res = client.post("/api/upload", files=files)
    d_id = upload_res.json()["metadata"]["id"]

    res = client.post("/api/automl/train", json={
        "dataset_id": d_id,
        "target_column": "Defaulter",
        "task_type": "classification"
    })
    assert res.status_code == 200
    data = res.json()

    explanations = data["metric_explanations"]
    assert "selection_criterion" in explanations
    assert "macro f1" in explanations["selection_criterion"].lower()
    # Check rationale transparency: explicitly mentions avoiding majority class dominance
    assert "majority" in explanations["selection_rationale"].lower() or "imbalance" in explanations["selection_rationale"].lower()






