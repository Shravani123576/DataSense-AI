import io
import uuid
import logging
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()


from models import (
    ProfileReport,
    SampleDatasetInfo,
    QueryRequest,
    QueryResponse,
    CustomChartRequest,
    ChartSpec,
    CleaningRecommendation,
    CleaningActionRequest,
    CleaningAuditRecord,
    CleaningSummaryResponse,
    AutoMLRequest,
    AutoMLResponse,
    WhatIfRequest,
    WhatIfResponse,
    AgentGoalRequest,
    AgentGoalResponse,
    ExecutiveReportRequest,
    ExecutiveReportResponse
)
from sample_datasets import SAMPLE_REGISTRY
from data_profiler import profile_dataframe, sanitize_for_json
from analysis_engine import generate_custom_chart
from llm_service import process_user_query
from data_cleaner import (
    generate_cleaning_recommendations,
    preview_cleaning_action,
    apply_cleaning_action
)
from ml_engine import train_automl_pipeline, simulate_what_if
from agent_orchestrator import process_agent_goal
from report_generator import generate_executive_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datasense.main")

app = FastAPI(
    title="DataSense AI API",
    description="Beginner-friendly AI Data Science Backend with Grounded Pandas Execution",
    version="2.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Storage for Active Dataset Sessions
RAW_DATASET_STORAGE: Dict[str, pd.DataFrame] = {}       # Pristine raw uploaded dataset
DATASET_STORAGE: Dict[str, pd.DataFrame] = {}           # Active dataset reflecting auditable cleanings
PROFILE_STORAGE: Dict[str, ProfileReport] = {}
METADATA_STORAGE: Dict[str, Dict[str, Any]] = {}
AUDIT_LOG_STORAGE: Dict[str, List[CleaningAuditRecord]] = {}
LAST_ML_STORAGE: Dict[str, AutoMLResponse] = {}


@app.on_event("startup")
def startup_event():
    """Startup with clean state. Datasets are loaded on-demand via upload or sample picker."""
    logger.info("DataSense AI Core V2 initialized with clean session state.")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DataSense AI Core V2",
        "active_datasets": len(DATASET_STORAGE)
    }


@app.get("/api/samples", response_model=List[SampleDatasetInfo])
def get_sample_datasets():
    """Retrieve metadata about pre-built beginner sample datasets."""
    samples = []
    for s_id, info in SAMPLE_REGISTRY.items():
        sample_df = info["generator"]()
        samples.append(SampleDatasetInfo(
            id=s_id,
            name=info["name"],
            description=info["description"],
            category=info["category"],
            rows=len(sample_df),
            columns=len(sample_df.columns),
            tags=info["tags"]
        ))
    return samples


@app.post("/api/samples/load", response_model=ProfileReport)
def load_sample_dataset(sample_id: str = Query(..., description="Sample dataset identifier")):
    """Load one of the pre-configured sample datasets into the active session."""
    if sample_id not in SAMPLE_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Sample dataset '{sample_id}' not found.")
        
    info = SAMPLE_REGISTRY[sample_id]
    df = info["generator"]()
    # Clean disguised numeric / date columns
    from data_profiler import clean_and_prepare_dataframe
    df = clean_and_prepare_dataframe(df)

    session_id = f"sample_{sample_id}"
    RAW_DATASET_STORAGE[session_id] = df.copy()
    DATASET_STORAGE[session_id] = df.copy()
    AUDIT_LOG_STORAGE[session_id] = []
    report = profile_dataframe(
        df=df,
        dataset_id=session_id,
        name=info["name"],
        filename=f"{sample_id}.csv",
        preset_questions=info["sample_questions"]
    )
    PROFILE_STORAGE[session_id] = report
    METADATA_STORAGE[session_id] = {
        "name": info["name"],
        "filename": f"{sample_id}.csv"
    }
    
    return report


@app.post("/api/upload", response_model=ProfileReport)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload user CSV or Excel file (.csv, .xlsx, .xls) and perform instant profiling."""
    filename = file.filename or "uploaded_dataset.csv"
    content = await file.read()
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    try:
        if filename.endswith(".csv") or filename.endswith(".txt"):
            try:
                df = pd.read_csv(io.BytesIO(content))
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), encoding="latin-1")
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload a CSV (.csv) or Excel (.xlsx, .xls) file."
            )
    except Exception as e:
        logger.exception(f"Error parsing uploaded file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Dataset contains no data rows (empty table).")

    # Clean and infer disguised numeric & date columns
    from data_profiler import clean_and_prepare_dataframe
    df = clean_and_prepare_dataframe(df)

    dataset_id = str(uuid.uuid4())[:8]
    clean_name = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    
    RAW_DATASET_STORAGE[dataset_id] = df.copy()
    DATASET_STORAGE[dataset_id] = df.copy()
    AUDIT_LOG_STORAGE[dataset_id] = []
    report = profile_dataframe(
        df=df,
        dataset_id=dataset_id,
        name=clean_name,
        filename=filename
    )
    PROFILE_STORAGE[dataset_id] = report
    METADATA_STORAGE[dataset_id] = {
        "name": clean_name,
        "filename": filename
    }
    
    return report


@app.post("/api/dataset/reset")
def reset_active_datasets():
    """Clear in-memory datasets and reset state."""
    RAW_DATASET_STORAGE.clear()
    DATASET_STORAGE.clear()
    PROFILE_STORAGE.clear()
    METADATA_STORAGE.clear()
    AUDIT_LOG_STORAGE.clear()
    LAST_ML_STORAGE.clear()
    return {"status": "reset", "active_datasets": 0}


@app.get("/api/dataset/{dataset_id}/preview")
def get_dataset_preview(
    dataset_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None)
):
    """Fetch paginated dataset rows with optional filtering."""
    if dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset not found or session expired.")

    df = DATASET_STORAGE[dataset_id]
    
    # Optional search across text/object columns
    filtered_df = df
    if search:
        s_lower = search.lower()
        mask = df.astype(str).apply(lambda row: row.str.lower().str.contains(s_lower, na=False).any(), axis=1)
        filtered_df = df[mask]

    total_records = len(filtered_df)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_df = filtered_df.iloc[start_idx:end_idx]

    rows = []
    for idx, row in page_df.iterrows():
        row_dict = {"_row_id": int(idx)}
        for k, v in row.items():
            row_dict[str(k)] = sanitize_for_json(v)
        rows.append(row_dict)

    return {
        "dataset_id": dataset_id,
        "total_rows": total_records,
        "total_columns": len(df.columns),
        "page": page,
        "page_size": page_size,
        "columns": [str(c) for c in df.columns],
        "rows": rows
    }


@app.get("/api/dataset/{dataset_id}/profile", response_model=ProfileReport)
def get_dataset_profile(dataset_id: str):
    """Retrieve full data profiling report and health scores."""
    if dataset_id not in PROFILE_STORAGE:
        # If dataset is present, recalculate
        if dataset_id in DATASET_STORAGE:
            df = DATASET_STORAGE[dataset_id]
            meta = METADATA_STORAGE.get(dataset_id, {"name": "Dataset", "filename": "data.csv"})
            report = profile_dataframe(df, dataset_id, meta["name"], meta["filename"])
            PROFILE_STORAGE[dataset_id] = report
            return report
        raise HTTPException(status_code=404, detail="Dataset profile not found.")
        
    return PROFILE_STORAGE[dataset_id]


@app.post("/api/query", response_model=QueryResponse)
def execute_natural_language_query(request: QueryRequest):
    """Process natural language question with strictly grounded pandas execution."""
    if request.dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload or load a dataset first.")

    df = DATASET_STORAGE[request.dataset_id]
    response = process_user_query(
        df=df,
        question=request.question,
        api_key=request.api_key
    )
    return response


@app.post("/api/custom-chart", response_model=ChartSpec)
def build_custom_chart(request: CustomChartRequest):
    """Generate custom interactive chart data from dataset columns."""
    if request.dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = DATASET_STORAGE[request.dataset_id]
    try:
        spec = generate_custom_chart(df, request)
        return spec
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Error building chart: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate chart: {str(e)}")


@app.get("/api/dataset/{dataset_id}/export/summary")
def export_summary_report(dataset_id: str):
    """Export comprehensive analytical data audit in Markdown format."""
    if dataset_id not in PROFILE_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    profile = PROFILE_STORAGE[dataset_id]
    meta = profile.metadata
    health = profile.health

    lines = [
        f"# Executive Data Health & Analytics Audit",
        f"**Dataset Name**: {meta.name}",
        f"**Source Filename**: `{meta.filename}`",
        f"**Audit Timestamp**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary",
        f"- **Total Records (Rows)**: {meta.rows:,}",
        f"- **Total Features (Columns)**: {meta.columns}",
        f"- **Data Completeness**: {100 - meta.missing_percentage:.1f}% ({meta.missing_cells} missing cells)",
        f"- **Duplicate Rows**: {meta.duplicate_rows} duplicate record(s)",
        f"- **Overall Quality Grade**: **{health.grade} ({health.overall_score}/100)**",
        f"",
        f"### Health Score Breakdown",
        f"- Completeness Score: {health.completeness_score}/100",
        f"- Uniqueness Score: {health.uniqueness_score}/100",
        f"- Outlier Health Score: {health.outlier_score}/100",
        f"- Consistency Score: {health.consistency_score}/100",
        f"",
        f"### Key Recommendations",
    ]
    for rec in health.recommendations:
        lines.append(f"- {rec}")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Feature Dictionary & Descriptive Statistics",
        f"| Feature | Inferred Type | Missing | Unique Values | Mean | Median | Std Dev | Min / Max | Outliers |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])

    for col in profile.columns:
        mean_str = f"{col.mean}" if col.mean is not None else "-"
        median_str = f"{col.median}" if col.median is not None else "-"
        std_str = f"{col.std}" if col.std is not None else "-"
        min_max_str = f"{col.min} / {col.max}" if col.min is not None else "-"
        outlier_str = f"{col.outlier_count} ({col.outlier_percentage}%)" if col.outlier_count is not None else "-"
        
        lines.append(
            f"| `{col.column}` | {col.inferred_type} | {col.null_count} ({col.null_percentage}%) | "
            f"{col.unique_count} | {mean_str} | {median_str} | {std_str} | {min_max_str} | {outlier_str} |"
        )

    if profile.top_correlations:
        lines.extend([
            f"",
            f"---",
            f"",
            f"## 3. Notable Correlations",
            f"| Feature 1 | Feature 2 | Pearson Correlation (r) | Strength Interpretation |",
            f"| :--- | :--- | :--- | :--- |"
        ])
        for pair in profile.top_correlations:
            abs_r = abs(pair.correlation)
            strength = "Strong" if abs_r >= 0.7 else ("Moderate" if abs_r >= 0.35 else "Weak")
            lines.append(f"| `{pair.col1}` | `{pair.col2}` | **{pair.correlation}** | {strength} |")

    report_content = "\n".join(lines)
    return Response(
        content=report_content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{meta.filename}_DataSense_Report.md"'}
    )


@app.get("/api/dataset/{dataset_id}/export/cleaned")
def export_cleaned_dataset(
    dataset_id: str,
    deduplicate: bool = Query(True),
    impute_numeric: bool = Query(True)
):
    """Export cleaned version of the active dataset with duplicates removed and numeric nulls imputed."""
    if dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = DATASET_STORAGE[dataset_id].copy()
    if deduplicate:
        df = df.drop_duplicates()
        
    if impute_numeric:
        for col in df.select_dtypes(include=["number"]).columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    
    meta = METADATA_STORAGE.get(dataset_id, {"filename": "dataset.csv"})
    clean_filename = f"cleaned_{meta['filename']}"
    
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{clean_filename}"'}
    )


@app.get("/api/dataset/{dataset_id}/export/csv-report")
def export_csv_report(dataset_id: str):
    """Export structured analytical report and feature dictionary as a downloadable CSV."""
    if dataset_id not in PROFILE_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    profile = PROFILE_STORAGE[dataset_id]
    meta = profile.metadata
    health = profile.health

    rows = []
    rows.append(["=== EXECUTIVE DATASET SUMMARY ===", "", "", "", "", "", "", "", "", ""])
    rows.append(["Dataset Name", meta.name, "", "", "", "", "", "", "", ""])
    rows.append(["Source Filename", meta.filename, "", "", "", "", "", "", "", ""])
    rows.append(["Total Records (Rows)", meta.rows, "", "", "", "", "", "", "", ""])
    rows.append(["Total Features (Columns)", meta.columns, "", "", "", "", "", "", "", ""])
    rows.append(["Completeness", f"{100 - meta.missing_percentage:.1f}%", "", "", "", "", "", "", "", ""])
    rows.append(["Missing Cells", meta.missing_cells, "", "", "", "", "", "", "", ""])
    rows.append(["Duplicate Rows", meta.duplicate_rows, "", "", "", "", "", "", "", ""])
    rows.append(["Overall Health Score", f"{health.overall_score}/100", "", "", "", "", "", "", "", ""])
    rows.append(["Health Grade", health.grade, "", "", "", "", "", "", "", ""])
    rows.append(["", "", "", "", "", "", "", "", "", ""])

    rows.append(["=== FEATURE DICTIONARY & DESCRIPTIVE STATISTICS ===", "", "", "", "", "", "", "", "", ""])
    rows.append(["Feature Name", "Data Type", "Missing Count", "Missing %", "Unique Count", "Mean", "Median", "Std Dev", "Min / Max", "Outlier Count"])
    for col in profile.columns:
        mean_v = col.mean if col.mean is not None else ""
        med_v = col.median if col.median is not None else ""
        std_v = col.std if col.std is not None else ""
        min_max = f"{col.min} to {col.max}" if col.min is not None else ""
        out_v = f"{col.outlier_count} ({col.outlier_percentage}%)" if col.outlier_count is not None else ""
        rows.append([
            col.column, col.inferred_type, col.null_count, f"{col.null_percentage}%",
            col.unique_count, mean_v, med_v, std_v, min_max, out_v
        ])
    rows.append(["", "", "", "", "", "", "", "", "", ""])

    if profile.top_correlations:
        rows.append(["=== TOP STATISTICAL CORRELATIONS ===", "", "", "", "", "", "", "", "", ""])
        rows.append(["Variable 1", "Variable 2", "Pearson Correlation (r)", "Strength", "", "", "", "", "", ""])
        for pair in profile.top_correlations:
            abs_r = abs(pair.correlation)
            strength = "Strong" if abs_r >= 0.7 else ("Moderate" if abs_r >= 0.35 else "Weak")
            rows.append([pair.col1, pair.col2, pair.correlation, strength, "", "", "", "", "", ""])
        rows.append(["", "", "", "", "", "", "", "", "", ""])

    if profile.is_restaurant_dataset and profile.restaurant_insights:
        rows.append(["=== ZOMATO RESTAURANT INTELLIGENCE ===", "", "", "", "", "", "", "", "", ""])
        ri = profile.restaurant_insights
        rows.append(["Average Rating", ri.get("avg_rating"), "", "", "", "", "", "", "", ""])
        rows.append(["Cost vs Rating Correlation", ri.get("cost_vs_rating_corr"), "", "", "", "", "", "", "", ""])
        rows.append(["Top Cuisines Analyzed", len(ri.get("top_cuisines", [])), "", "", "", "", "", "", "", ""])
        rows.append(["Top Locations Analyzed", len(ri.get("top_locations", [])), "", "", "", "", "", "", "", ""])

    # Write to CSV
    stream = io.StringIO()
    import csv
    writer = csv.writer(stream)
    for r in rows:
        writer.writerow(r)
    stream.seek(0)

    filename = f"{meta.filename.rsplit('.', 1)[0]}_DataSense_Report.csv"
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# =====================================================================
# STAGE 2: AUDITABLE DATA CLEANING ENDPOINTS
# =====================================================================

@app.get("/api/clean/{dataset_id}/recommendations", response_model=List[CleaningRecommendation])
def get_cleaning_recommendations_endpoint(dataset_id: str):
    """Diagnose dataset anomalies and generate evidence-backed cleaning recommendations."""
    if dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset session not found.")
    df = DATASET_STORAGE[dataset_id]
    recs = generate_cleaning_recommendations(df)
    return recs


@app.post("/api/clean/preview")
def preview_cleaning_endpoint(request: CleaningActionRequest):
    """Generate uncommitted before/after diff preview of proposed transformation."""
    if request.dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset session not found.")
    df = DATASET_STORAGE[request.dataset_id]
    try:
        preview = preview_cleaning_action(df, request)
        return preview
    except Exception as e:
        logger.exception(f"Cleaning preview error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/clean/apply", response_model=CleaningSummaryResponse)
def apply_cleaning_endpoint(request: CleaningActionRequest):
    """Execute approved cleaning action, update active dataset, record audit provenance."""
    if request.dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset session not found.")
    
    df = DATASET_STORAGE[request.dataset_id]
    meta = METADATA_STORAGE.get(request.dataset_id, {"name": "Dataset", "filename": "data.csv"})

    try:
        clean_df, audit_record = apply_cleaning_action(df, request)
        
        # Update active dataset in memory
        DATASET_STORAGE[request.dataset_id] = clean_df
        
        # Append to audit history log
        if request.dataset_id not in AUDIT_LOG_STORAGE:
            AUDIT_LOG_STORAGE[request.dataset_id] = []
        AUDIT_LOG_STORAGE[request.dataset_id].append(audit_record)

        # Refresh profile report
        new_profile = profile_dataframe(
            clean_df,
            request.dataset_id,
            meta["name"],
            meta["filename"]
        )
        PROFILE_STORAGE[request.dataset_id] = new_profile

        # Remaining recommendations
        remaining_recs = generate_cleaning_recommendations(clean_df)

        return CleaningSummaryResponse(
            dataset_id=request.dataset_id,
            total_records=len(clean_df),
            total_cleaned_actions_applied=len(AUDIT_LOG_STORAGE[request.dataset_id]),
            audit_history=AUDIT_LOG_STORAGE[request.dataset_id],
            current_health=new_profile.health,
            remaining_recommendations=remaining_recs,
            cleaned_filename=f"cleaned_{meta['filename']}"
        )
    except Exception as e:
        logger.exception(f"Cleaning application error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/clean/{dataset_id}/audit", response_model=CleaningSummaryResponse)
def get_cleaning_audit_endpoint(dataset_id: str):
    """Retrieve complete audit history and remaining cleaning recommendations."""
    if dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset session not found.")
    
    df = DATASET_STORAGE[dataset_id]
    meta = METADATA_STORAGE.get(dataset_id, {"name": "Dataset", "filename": "data.csv"})
    profile = PROFILE_STORAGE.get(dataset_id) or profile_dataframe(df, dataset_id, meta["name"], meta["filename"])
    audit_history = AUDIT_LOG_STORAGE.get(dataset_id, [])
    remaining_recs = generate_cleaning_recommendations(df)

    return CleaningSummaryResponse(
        dataset_id=dataset_id,
        total_records=len(df),
        total_cleaned_actions_applied=len(audit_history),
        audit_history=audit_history,
        current_health=profile.health,
        remaining_recommendations=remaining_recs,
        cleaned_filename=f"cleaned_{meta['filename']}"
    )


@app.post("/api/clean/{dataset_id}/reset-to-raw")
def reset_to_raw_endpoint(dataset_id: str):
    """Revert active working dataset to original raw state and clear audit records."""
    if dataset_id not in RAW_DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Raw dataset snapshot not found.")

    raw_copy = RAW_DATASET_STORAGE[dataset_id].copy()
    DATASET_STORAGE[dataset_id] = raw_copy
    AUDIT_LOG_STORAGE[dataset_id] = []
    
    meta = METADATA_STORAGE.get(dataset_id, {"name": "Dataset", "filename": "data.csv"})
    new_profile = profile_dataframe(raw_copy, dataset_id, meta["name"], meta["filename"])
    PROFILE_STORAGE[dataset_id] = new_profile

    return {"status": "reverted", "rows": len(raw_copy), "health": new_profile.health}


# =====================================================================
# STAGE 4: AUTOMATED MACHINE LEARNING & WHAT-IF ENDPOINTS
# =====================================================================

@app.post("/api/automl/train", response_model=AutoMLResponse)
def train_automl_endpoint(request: AutoMLRequest):
    """Execute leakage-free AutoML tournament across baseline and competitive models."""
    if request.dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset session not found.")

    df = DATASET_STORAGE[request.dataset_id]
    try:
        response = train_automl_pipeline(df, request)
        LAST_ML_STORAGE[request.dataset_id] = response
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"AutoML Training Error: {e}")
        raise HTTPException(status_code=500, detail=f"AutoML failed: {str(e)}")


@app.post("/api/automl/what-if", response_model=WhatIfResponse)
def what_if_endpoint(request: WhatIfRequest):
    """Run counterfactual simulation against the active trained model."""
    try:
        response = simulate_what_if(request)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"What-If Simulation Error: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


# =====================================================================
# STAGE 3: AGENT GOAL ORCHESTRATION ENDPOINT
# =====================================================================

@app.post("/api/agent/goal", response_model=AgentGoalResponse)
def execute_agent_goal_endpoint(request: AgentGoalRequest):
    """Decompose high-level data objective into step-by-step plan and execute tools."""
    if request.dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset session not found.")

    df = DATASET_STORAGE[request.dataset_id]
    meta = METADATA_STORAGE.get(request.dataset_id, {"name": "Dataset", "filename": "data.csv"})

    try:
        response = process_agent_goal(
            df=df,
            req=request,
            dataset_name=meta["name"],
            filename=meta["filename"]
        )
        return response
    except Exception as e:
        logger.exception(f"Agent Orchestration Error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent workflow error: {str(e)}")


# =====================================================================
# STAGE 6: EXECUTIVE REPORTING ENDPOINTS
# =====================================================================

@app.post("/api/report/generate", response_model=ExecutiveReportResponse)
def generate_report_endpoint(request: ExecutiveReportRequest):
    """Synthesize complete data health, audit provenance, and ML evaluation into executive brief."""
    if request.dataset_id not in PROFILE_STORAGE and request.dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset session not found.")

    df = DATASET_STORAGE[request.dataset_id]
    meta = METADATA_STORAGE.get(request.dataset_id, {"name": "Dataset", "filename": "data.csv"})
    profile = PROFILE_STORAGE.get(request.dataset_id) or profile_dataframe(df, request.dataset_id, meta["name"], meta["filename"])

    audit_history = AUDIT_LOG_STORAGE.get(request.dataset_id, [])
    ml_result = LAST_ML_STORAGE.get(request.dataset_id)

    try:
        report = generate_executive_report(
            profile=profile,
            audit_history=audit_history if request.include_audit else [],
            ml_response=ml_result if request.include_ml else None,
            objective=request.objective
        )
        return report
    except Exception as e:
        logger.exception(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@app.get("/api/report/{dataset_id}/download/html")
def download_html_report_endpoint(dataset_id: str):
    """Download executive brief styled HTML document."""
    if dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset session not found.")

    df = DATASET_STORAGE[dataset_id]
    meta = METADATA_STORAGE.get(dataset_id, {"name": "Dataset", "filename": "data.csv"})
    profile = PROFILE_STORAGE.get(dataset_id) or profile_dataframe(df, dataset_id, meta["name"], meta["filename"])
    audit_history = AUDIT_LOG_STORAGE.get(dataset_id, [])
    ml_result = LAST_ML_STORAGE.get(dataset_id)

    report = generate_executive_report(
        profile=profile,
        audit_history=audit_history,
        ml_response=ml_result,
        objective=None
    )

    filename = f"{meta['filename'].rsplit('.', 1)[0]}_Executive_Brief.html"
    return Response(
        content=report.html_content,
        media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.get("/api/report/{dataset_id}/download/markdown")
def download_markdown_report_endpoint(dataset_id: str):
    """Download executive brief Markdown document."""
    if dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset session not found.")

    df = DATASET_STORAGE[dataset_id]
    meta = METADATA_STORAGE.get(dataset_id, {"name": "Dataset", "filename": "data.csv"})
    profile = PROFILE_STORAGE.get(dataset_id) or profile_dataframe(df, dataset_id, meta["name"], meta["filename"])
    audit_history = AUDIT_LOG_STORAGE.get(dataset_id, [])
    ml_result = LAST_ML_STORAGE.get(dataset_id)

    report = generate_executive_report(
        profile=profile,
        audit_history=audit_history,
        ml_response=ml_result,
        objective=None
    )

    filename = f"{meta['filename'].rsplit('.', 1)[0]}_Executive_Brief.md"
    return Response(
        content=report.markdown_content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


