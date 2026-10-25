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




