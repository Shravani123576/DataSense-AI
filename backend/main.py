import os
import io
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()


from datetime import datetime, timezone
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
    ExecutiveReportResponse,
    DatasetVersionInfo,
    CleaningProposal,
    ProposalResponse,
    CoercionAuditItem,
    HypothesisTestRequest,
    HypothesisTestResponse,
    GraphIntelligenceAnalyzeRequest,
    GraphIntelligenceResponse,
    GraphIntelligenceAskRequest,
    GraphIntelligenceAskResponse,
    ToolExecutionApiRequest,
    WorksheetInfo,
    WorkbookInspectResponse,
    TableInventoryItem,
    MultiFileUploadResponse,
    CandidateJoinKey,
    TableRelationshipItem,
    RelationshipDetectionResponse,
    JoinPreviewRequest,
    JoinPreviewResponse,
    JoinApplyRequest,
    AgentChatRequest
)
from tools import default_registry, ToolExecutionResult
from sample_datasets import SAMPLE_REGISTRY
from data_profiler import profile_dataframe, sanitize_for_json, infer_column_type, clean_and_prepare_dataframe
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
import analyst_agent
from table_relationship_engine import (
    safe_extract_zip,
    inspect_excel_worksheets,
    detect_table_relationships,
    compute_join_preview,
    execute_join
)
from data_utils import safe_read_csv, normalize_dataframe_headers_and_columns

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

# In-Memory Storage for Active Dataset Sessions & Provenance
RAW_BYTE_STORAGE: Dict[str, bytes] = {}                 # Raw file bytes (size-limited to 50MB)
PARSED_DATASET_STORAGE: Dict[str, pd.DataFrame] = {}     # Full parsed data stored separately
UNMODIFIED_RAW_STORAGE: Dict[str, pd.DataFrame] = {}   # Unmodified parsed snapshot before type coercion/preparation
RAW_DATASET_STORAGE: Dict[str, pd.DataFrame] = {}       # Reference to genuinely raw unmodified parsed data
DATASET_STORAGE: Dict[str, pd.DataFrame] = {}           # Active working dataset reflecting approved changes
DATASET_VERSIONS_STORAGE: Dict[str, List[Dict[str, Any]]] = {} # Committed dataset versions [V0, V1, ...]
PROPOSED_TRANSFORMATIONS_STORAGE: Dict[str, Dict[str, CleaningProposal]] = {} # Staged proposals awaiting user approval
COERCION_AUDIT_STORAGE: Dict[str, List[Dict[str, Any]]] = {}     # Traceable type coercion & conversion failure logs
PROFILE_STORAGE: Dict[str, ProfileReport] = {}
METADATA_STORAGE: Dict[str, Dict[str, Any]] = {}
AUDIT_LOG_STORAGE: Dict[str, List[CleaningAuditRecord]] = {}
LAST_ML_STORAGE: Dict[str, AutoMLResponse] = {}
MULTI_TABLE_SESSIONS: Dict[str, Dict[str, Tuple[str, pd.DataFrame]]] = {} # Multi-table session inventory


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


def register_dataset_session(
    dataset_id: str,
    name: str,
    filename: str,
    content: bytes,
    unmodified_df: pd.DataFrame,
    preset_questions: Optional[List[str]] = None,
    custom_v0_desc: Optional[str] = None
) -> ProfileReport:
    """
    Register an ingested or generated DataFrame as an isolated, fully auditable dataset session.
    Preserves raw bytes, stores parsed data separately, detects partial years, applies annual series filtering
    when J-D is present (with explicit audit logging), initializes traceable version history,
    and stores one unified record count in the dataset object read by all screens.
    """
    # 1. Keep raw upload bytes unchanged
    RAW_BYTE_STORAGE[dataset_id] = content
    # Store full parsed data separately
    PARSED_DATASET_STORAGE[dataset_id] = unmodified_df.copy()
    UNMODIFIED_RAW_STORAGE[dataset_id] = unmodified_df.copy()
    RAW_DATASET_STORAGE[dataset_id] = unmodified_df.copy()
    AUDIT_LOG_STORAGE[dataset_id] = []
    PROPOSED_TRANSFORMATIONS_STORAGE[dataset_id] = {}

    raw_rows = len(unmodified_df)
    missing_monthly = getattr(unmodified_df, "attrs", {}).get("missing_monthly_cells", None)

    # 2. Normalize headers and columns to 1D
    parsed_normalized = normalize_dataframe_headers_and_columns(unmodified_df)

    # 3. Check for annual series column (J-D) and drop rows with no J-D value, logging the reason
    jd_col = None
    for c in parsed_normalized.columns:
        if str(c).strip().upper() in ["J-D", "J_D"]:
            jd_col = c
            break

    working_df = parsed_normalized.copy()
    dropped_years_info = []
    no_jd_count = 0

    if jd_col is not None:
        jd_num = pd.to_numeric(working_df[jd_col], errors="coerce")
        no_jd_mask = jd_num.isna()
        no_jd_count = int(no_jd_mask.sum())
        if no_jd_count > 0:
            year_col = None
            for c in working_df.columns:
                if str(c).strip().lower() == "year":
                    year_col = c
                    break
            if year_col:
                dropped_years_info = working_df.loc[no_jd_mask, year_col].tolist()
            else:
                dropped_years_info = list(working_df.index[no_jd_mask])

            exclusion_reason = (
                f"Excluded {no_jd_count} incomplete annual record(s) lacking J-D annual mean "
                f"(Year(s): {dropped_years_info}). Partial monthly records preserved in raw dataset."
            )
            logger.info(exclusion_reason)
            AUDIT_LOG_STORAGE[dataset_id].append(CleaningAuditRecord(
                id=str(uuid.uuid4())[:8],
                timestamp=datetime.now(timezone.utc).isoformat(),
                rule="incomplete_annual_exclusion",
                column=str(jd_col),
                affected_rows=no_jd_count,
                affected_percentage=round((no_jd_count / raw_rows * 100), 2) if raw_rows > 0 else 0.0,
                strategy_used="exclude_incomplete_years",
                transformation_summary=exclusion_reason,
                reason=exclusion_reason
            ))
            # Drop the incomplete annual rows without J-D value
            working_df = working_df[~no_jd_mask].copy()

    # 4. Clean and infer disguised numeric & date columns with explicit audit recording
    working_df, coercion_audit = clean_and_prepare_dataframe(working_df, return_audit=True)
    COERCION_AUDIT_STORAGE[dataset_id] = coercion_audit

    DATASET_STORAGE[dataset_id] = working_df.copy()

    # 5. Version History
    versions = [
        {
            "version_id": 0,
            "parent_version_id": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": "raw_parsed",
            "description": custom_v0_desc or f"Raw parsed file '{filename}' ({raw_rows} total records including partial years)",
            "affected_rows": 0,
            "affected_cols": 0,
            "total_rows": raw_rows,
            "total_cols": len(unmodified_df.columns),
        }
    ]

    curr_v = 1
    if no_jd_count > 0:
        versions.append({
            "version_id": curr_v,
            "parent_version_id": curr_v - 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": "annual_series_filter",
            "description": f"Filtered to complete annual series: {len(working_df)} complete years ({no_jd_count} partial year excluded: {dropped_years_info})",
            "affected_rows": no_jd_count,
            "affected_cols": 0,
            "total_rows": len(working_df),
            "total_cols": len(working_df.columns),
        })
        curr_v += 1

    versions.append({
        "version_id": curr_v,
        "parent_version_id": curr_v - 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "type_preparation",
        "description": f"Traceable type inference & coercion ({len(coercion_audit)} columns adjusted)",
        "affected_rows": sum(c.get("affected_rows", 0) for c in coercion_audit),
        "affected_cols": len(coercion_audit),
        "total_rows": len(working_df),
        "total_cols": len(working_df.columns),
    })
    DATASET_VERSIONS_STORAGE[dataset_id] = versions

    # 6. Store one record count in the dataset object read from single source everywhere
    unified_record_count = len(working_df)
    record_label = (
        f"{unified_record_count} complete years, {no_jd_count} partial year excluded"
        if no_jd_count > 0
        else f"{unified_record_count} records"
    )

    report = profile_dataframe(
        df=working_df,
        dataset_id=dataset_id,
        name=name,
        filename=filename,
        preset_questions=preset_questions,
        record_count_label=record_label,
        partial_rows_excluded=no_jd_count,
        total_raw_rows=raw_rows,
        missing_monthly_cells=missing_monthly
    )
    PROFILE_STORAGE[dataset_id] = report
    METADATA_STORAGE[dataset_id] = {
        "name": name,
        "filename": filename,
        "rows": unified_record_count,
        "record_count": unified_record_count,
        "record_count_label": record_label,
        "partial_rows_excluded": no_jd_count,
        "total_raw_rows": raw_rows,
        "missing_monthly_cells": missing_monthly
    }
    return report


@app.post("/api/samples/load", response_model=ProfileReport)
def load_sample_dataset(sample_id: str = Query(..., description="Sample dataset identifier")):
    """Load one of the pre-configured sample datasets into the active session."""
    if sample_id not in SAMPLE_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Sample dataset '{sample_id}' not found.")
        
    info = SAMPLE_REGISTRY[sample_id]
    unmodified_df = info["generator"]()
    session_id = f"sample_{sample_id}"
    csv_bytes = unmodified_df.to_csv(index=False).encode("utf-8")

    return register_dataset_session(
        dataset_id=session_id,
        name=info["name"],
        filename=f"{sample_id}.csv",
        content=csv_bytes,
        unmodified_df=unmodified_df,
        preset_questions=info["sample_questions"],
        custom_v0_desc=f"Original unmodified sample dataset '{info['name']}'"
    )


@app.post("/api/upload", response_model=ProfileReport)
async def upload_dataset(file: UploadFile = File(...), sheet_name: Optional[str] = Form(None)):
    """Upload user CSV or Excel file (.csv, .xlsx, .xls) or single ZIP archive and perform instant profiling."""
    filename = file.filename or "uploaded_dataset.csv"
    fn_lower = filename.lower()
    content = await file.read()
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    # Security & Resource limit: 50 MB max
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 50 MB limit.")

    try:
        if fn_lower.endswith(".zip"):
            try:
                extracted = safe_extract_zip(content)
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))
            first_name, first_bytes = extracted[0]
            filename = first_name
            fn_lower = first_name.lower()
            content = first_bytes

        banner_title = None
        if fn_lower.endswith(".csv") or fn_lower.endswith(".txt") or fn_lower.endswith(".tsv"):
            delim = "\t" if fn_lower.endswith(".tsv") else None
            df, banner_title = safe_read_csv(content, delimiter=delim)
        elif fn_lower.endswith((".xlsx", ".xls")):
            try:
                excel_file = pd.ExcelFile(io.BytesIO(content))
            except Exception as e:
                logger.warning(f"Excel read error: {e}")
                raise HTTPException(
                    status_code=400,
                    detail="Failed to read Excel workbook: The file appears to be corrupted, password-protected, or in an invalid format."
                )

            available_sheets = excel_file.sheet_names
            if not available_sheets:
                raise HTTPException(status_code=400, detail="Uploaded Excel workbook contains no worksheets.")

            # Select appropriate sheet
            target_sheet = None
            if sheet_name and sheet_name.strip():
                req_sheet = sheet_name.strip()
                if req_sheet not in available_sheets:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Worksheet '{req_sheet}' not found in workbook. Available sheets: {', '.join(available_sheets)}"
                    )
                target_sheet = req_sheet
                df = pd.read_excel(excel_file, sheet_name=target_sheet)
            else:
                # Default: Check sheet 0; if empty, find first non-empty sheet
                df = pd.read_excel(excel_file, sheet_name=available_sheets[0])
                target_sheet = available_sheets[0]
                if len(df) == 0 and len(available_sheets) > 1:
                    for s in available_sheets[1:]:
                        candidate_df = pd.read_excel(excel_file, sheet_name=s)
                        if len(candidate_df) > 0:
                            df = candidate_df
                            target_sheet = s
                            logger.info(f"Loaded non-empty worksheet '{s}' (sheet 0 was blank).")
                            break
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload a CSV (.csv), Excel (.xlsx, .xls), or ZIP (.zip) file."
            )
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Uploaded file contains no data rows or columns.")
    except pd.errors.ParserError as pe:
        raise HTTPException(status_code=400, detail=f"Malformed CSV: Failed to parse rows due to inconsistent columns: {str(pe)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error parsing uploaded file: {e}")
        err_msg = str(e)
        if "No module named 'xlrd'" in err_msg:
            clean_err = "Legacy .xls format requires xlrd library. Please upload a modern .xlsx file or convert to CSV."
        elif "BadZipFile" in err_msg or "not a zip file" in err_msg.lower():
            clean_err = "The file is not a valid Excel (.xlsx) file or is corrupted."
        else:
            clean_err = f"Failed to parse file: {err_msg[:120]}"
        raise HTTPException(status_code=400, detail=clean_err)

    df = normalize_dataframe_headers_and_columns(df)
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Dataset contains no data rows (empty table).")

    dataset_id = str(uuid.uuid4())[:8]
    clean_name = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    if banner_title and ("glb" in filename.lower() or "untitled" in clean_name.lower()):
        clean_name = banner_title

    report = register_dataset_session(
        dataset_id=dataset_id,
        name=clean_name,
        filename=filename,
        content=content,
        unmodified_df=df
    )
    return report


@app.post("/api/upload/inspect-sheets", response_model=WorkbookInspectResponse)
async def inspect_sheets_endpoint(file: UploadFile = File(...)):
    """Inspect all worksheets inside an Excel workbook without committing them."""
    filename = file.filename or "workbook.xlsx"
    if not filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Sheet inspection is only available for Excel files (.xlsx, .xls).")
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 50 MB limit.")
    try:
        inspect_resp = inspect_excel_worksheets(content, filename)
        return inspect_resp
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Sheet inspection error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to inspect workbook sheets: {str(e)[:120]}")


@app.post("/api/upload/multi", response_model=MultiFileUploadResponse)
async def upload_multi_files(files: List[UploadFile] = File(...)):
    """
    Ingest multiple CSV and Excel files, or unpack a ZIP archive.
    Builds a table inventory with row counts, columns, inferred types, and previews.
    Registers each table into session storage for individual profiling or relational analysis.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    session_id = f"session_{uuid.uuid4().hex[:8]}"
    tables_inventory: List[TableInventoryItem] = []
    session_tables: Dict[str, Tuple[str, pd.DataFrame]] = {}
    warnings: List[str] = []

    file_payloads: List[Tuple[str, bytes]] = []

    total_upload_size = 0
    for f in files:
        content = await f.read()
        total_upload_size += len(content)
        if total_upload_size > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Total upload size exceeds 100 MB limit.")
        
        fname = f.filename or "file.csv"
        if fname.lower().endswith(".zip"):
            try:
                extracted = safe_extract_zip(content)
                file_payloads.extend(extracted)
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))
        else:
            file_payloads.append((fname, content))

    if not file_payloads:
        raise HTTPException(status_code=400, detail="No valid data files found in upload.")

    for fname, data_bytes in file_payloads:
        fn_lower = fname.lower()
        if len(data_bytes) == 0:
            warnings.append(f"Skipped empty file '{fname}' (0 bytes).")
            continue

        if fn_lower.endswith((".csv", ".tsv", ".txt")):
            try:
                delim = "\t" if fn_lower.endswith(".tsv") else None
                df, banner_title = safe_read_csv(data_bytes, delimiter=delim)
                
                table_id = str(uuid.uuid4())[:8]
                table_name = banner_title if banner_title and ("glb" in fname.lower() or "untitled" in fname.lower()) else fname.rsplit(".", 1)[0].replace("_", " ").title()
                
                session_tables[table_id] = (table_name, df)
                register_dataset_session(table_id, table_name, fname, data_bytes, df)

                inferred_types = {str(c): infer_column_type(df[c]) for c in df.columns}
                preview = [
                    {str(c): sanitize_for_json(v) for c, v in row.items()}
                    for row in df.head(5).to_dict(orient="records")
                ]

                tables_inventory.append(TableInventoryItem(
                    table_id=table_id,
                    filename=fname,
                    rows=len(df),
                    columns=len(df.columns),
                    column_names=[str(c) for c in df.columns],
                    inferred_types=inferred_types,
                    parsing_status="success",
                    preview_rows=preview
                ))
            except Exception as e:
                warnings.append(f"Failed to parse CSV '{fname}': {str(e)[:100]}")

        elif fn_lower.endswith((".xlsx", ".xls")):
            try:
                excel_file = pd.ExcelFile(io.BytesIO(data_bytes))
                for s_name in excel_file.sheet_names:
                    try:
                        sheet_df = pd.read_excel(excel_file, sheet_name=s_name)
                        if len(sheet_df) == 0:
                            warnings.append(f"Sheet '{s_name}' in '{fname}' is empty (0 rows).")
                            continue

                        table_id = str(uuid.uuid4())[:8]
                        table_name = f"{fname.rsplit('.', 1)[0]} - {s_name}".replace("_", " ").title()
                        session_tables[table_id] = (table_name, sheet_df)
                        
                        register_dataset_session(table_id, table_name, f"{fname}#{s_name}", data_bytes, sheet_df)

                        inferred_types = {str(c): infer_column_type(sheet_df[c]) for c in sheet_df.columns}
                        preview = [
                            {str(c): sanitize_for_json(v) for c, v in row.items()}
                            for row in sheet_df.head(5).to_dict(orient="records")
                        ]

                        tables_inventory.append(TableInventoryItem(
                            table_id=table_id,
                            filename=fname,
                            sheet_name=s_name,
                            rows=len(sheet_df),
                            columns=len(sheet_df.columns),
                            column_names=[str(c) for c in sheet_df.columns],
                            inferred_types=inferred_types,
                            parsing_status="success",
                            preview_rows=preview
                        ))
                    except Exception as se:
                        warnings.append(f"Failed to parse sheet '{s_name}' in '{fname}': {str(se)[:100]}")
            except Exception as e:
                warnings.append(f"Failed to read Excel workbook '{fname}': {str(e)[:100]}")
        else:
            warnings.append(f"Skipped unsupported file format '{fname}'.")

    if not tables_inventory:
        raise HTTPException(status_code=400, detail="Could not successfully parse any tables from the uploaded files.")

    MULTI_TABLE_SESSIONS[session_id] = session_tables

    return MultiFileUploadResponse(
        session_id=session_id,
        total_tables=len(tables_inventory),
        tables=tables_inventory,
        warnings=warnings
    )


@app.post("/api/tables/relationships", response_model=RelationshipDetectionResponse)
def get_table_relationships_endpoint(
    session_id: Optional[str] = Query(None),
    table_ids: Optional[List[str]] = Query(None)
):
    """
    Inspect schemas and value distributions across tables to detect candidate join keys,
    cardinality (1:1, 1:N, N:M), and value overlap.
    """
    tables_to_check: Dict[str, Tuple[str, pd.DataFrame]] = {}
    if session_id and session_id in MULTI_TABLE_SESSIONS:
        tables_to_check.update(MULTI_TABLE_SESSIONS[session_id])
    
    if table_ids:
        for tid in table_ids:
            if tid in DATASET_STORAGE and tid not in tables_to_check:
                name = METADATA_STORAGE.get(tid, {}).get("name", tid)
                tables_to_check[tid] = (name, DATASET_STORAGE[tid])

    if len(tables_to_check) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two tables are required to detect relationships. Provide a valid session_id or at least two table_ids."
        )

    return detect_table_relationships(tables_to_check)


@app.post("/api/tables/join-preview", response_model=JoinPreviewResponse)
def preview_table_join_endpoint(request: JoinPreviewRequest):
    """
    Preview a proposed join between two tables without mutating or creating datasets.
    Calculates expected row count, unmatched counts, and warns of Cartesian multiplication risk.
    """
    if request.table_a_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail=f"Table A ('{request.table_a_id}') not found.")
    if request.table_b_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail=f"Table B ('{request.table_b_id}') not found.")

    df_a = DATASET_STORAGE[request.table_a_id]
    df_b = DATASET_STORAGE[request.table_b_id]
    name_a = METADATA_STORAGE.get(request.table_a_id, {}).get("name", "Table A")
    name_b = METADATA_STORAGE.get(request.table_b_id, {}).get("name", "Table B")

    try:
        preview = compute_join_preview(
            df_a=df_a,
            name_a=name_a,
            df_b=df_b,
            name_b=name_b,
            col_a=request.join_column_a,
            col_b=request.join_column_b,
            join_type=request.join_type
        )
        return preview
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Join preview error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to compute join preview: {str(e)[:120]}")


@app.post("/api/tables/join-apply", response_model=ProfileReport)
def apply_table_join_endpoint(request: JoinApplyRequest):
    """
    Execute an approved join between two tables, creating an isolated joined dataset session
    with full data lineage, version history (V0: Join), and instant profiling.
    """
    if request.table_a_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail=f"Table A ('{request.table_a_id}') not found.")
    if request.table_b_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail=f"Table B ('{request.table_b_id}') not found.")

    df_a = DATASET_STORAGE[request.table_a_id]
    df_b = DATASET_STORAGE[request.table_b_id]
    name_a = METADATA_STORAGE.get(request.table_a_id, {}).get("name", "Table A")
    name_b = METADATA_STORAGE.get(request.table_b_id, {}).get("name", "Table B")

    try:
        joined_df = execute_join(
            df_a=df_a,
            name_a=name_a,
            df_b=df_b,
            name_b=name_b,
            col_a=request.join_column_a,
            col_b=request.join_column_b,
            join_type=request.join_type
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Join execution error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to execute join: {str(e)[:120]}")

    new_dataset_id = str(uuid.uuid4())[:8]
    default_name = request.new_dataset_name or f"Joined ({name_a} + {name_b})"
    csv_bytes = joined_df.to_csv(index=False).encode("utf-8")

    report = register_dataset_session(
        dataset_id=new_dataset_id,
        name=default_name,
        filename=f"joined_{new_dataset_id}.csv",
        content=csv_bytes,
        unmodified_df=joined_df,
        custom_v0_desc=f"Joined '{name_a}' and '{name_b}' on {request.join_column_a} == {request.join_column_b} ({request.join_type})"
    )
    return report


@app.post("/api/dataset/reset")
def reset_active_datasets():
    """Clear in-memory datasets and reset state."""
    RAW_BYTE_STORAGE.clear()
    UNMODIFIED_RAW_STORAGE.clear()
    RAW_DATASET_STORAGE.clear()
    DATASET_STORAGE.clear()
    DATASET_VERSIONS_STORAGE.clear()
    PROPOSED_TRANSFORMATIONS_STORAGE.clear()
    COERCION_AUDIT_STORAGE.clear()
    PROFILE_STORAGE.clear()
    METADATA_STORAGE.clear()
    AUDIT_LOG_STORAGE.clear()
    LAST_ML_STORAGE.clear()
    MULTI_TABLE_SESSIONS.clear()
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
    
    filtered_df = df
    search_str = search if isinstance(search, str) and search.strip() else None
    if search_str:
        s_lower = search_str.lower()
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
    """Process natural language question with strictly grounded agent execution."""
    if request.dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload or load a dataset first.")

    active_key = request.api_key or os.getenv("GEMINI_API_KEY")
    df = DATASET_STORAGE[request.dataset_id]

    if active_key:
        try:
            from analyst_agent import AnalystAgent
            from models import ActiveChartContext

            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            val_col = num_cols[0] if num_cols else df.columns[0]
            active_chart = ActiveChartContext(
                chart_type="line",
                x_column="Year",
                y_column=str(val_col),
                aggregation="none",
                filter_summary=None,
                rows_rendered=len(df),
                displayed_categories=None,
                total_categories=None
            )
            agent = AnalystAgent(dataset_id=request.dataset_id, df=df, active_chart=active_chart)
            agent_res = agent.run(request.question)

            summary_figs = {}
            if agent_res.column_value and not str(agent_res.column_value).startswith("Not computed"):
                summary_figs["value"] = agent_res.column_value

            return QueryResponse(
                question=request.question,
                answer_text=agent_res.narrative,
                code_executed="# Computed via DataSense AnalystAgent & Deterministic Tools",
                result_type="text",
                summary_figures=summary_figs,
                execution_time_ms=round(agent_res.duration_ms, 2),
                grounded=agent_res.grounding_verified,
                error=None
            )
        except Exception as e:
            logger.exception(f"Error in AnalystAgent execution for Copilot: {e}")
            return process_user_query(
                df=df,
                question=request.question,
                api_key=active_key
            )
    else:
        # LLM not configured
        res = process_user_query(df=df, question=request.question, api_key=None)
        if not res.answer_text or "couldn't compute" in res.answer_text.lower():
            res.answer_text = "AI model not configured. Add GEMINI_API_KEY to backend/.env"
            res.error = "AI model not configured. Add GEMINI_API_KEY to backend/.env"
        return res


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
        logger.exception(f"ValueError in build_custom_chart: {ve}")
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

    audit_logs = AUDIT_LOG_STORAGE.get(dataset_id, [])
    if audit_logs:
        lines.extend([
            f"",
            f"---",
            f"",
            f"## 4. Approved Cleaning Actions Log",
            f"| Timestamp | Rule | Column | Affected Rows | Strategy | Transformation Summary |",
            f"| :--- | :--- | :--- | :--- | :--- | :--- |"
        ])
        for record in audit_logs:
            col_d = f"`{record.column}`" if record.column else "Dataset-Wide"
            lines.append(
                f"| {record.timestamp} | {record.rule} | {col_d} | {record.affected_rows:,} ({record.affected_percentage}%) | "
                f"{record.strategy_used} | {record.transformation_summary} |"
            )

    report_content = "\n".join(lines)
    return Response(
        content=report_content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{meta.filename}_DataSense_Report.md"'}
    )


@app.get("/api/dataset/{dataset_id}/export/cleaned")
def export_cleaned_dataset(
    dataset_id: str,
    deduplicate: bool = Query(False, description="Apply auto-deduplication if not already applied"),
    impute_numeric: bool = Query(False, description="Apply auto-median imputation if not already applied")
):
    """Export the active cleaned dataset with all user-approved transformations preserved."""
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


@app.get("/api/dataset/{dataset_id}/export/raw")
def export_raw_dataset(dataset_id: str):
    """Export the original pristine parsed dataset snapshot exactly as uploaded."""
    if dataset_id not in UNMODIFIED_RAW_STORAGE and dataset_id not in RAW_DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Raw dataset not found.")

    raw_source = UNMODIFIED_RAW_STORAGE[dataset_id] if dataset_id in UNMODIFIED_RAW_STORAGE else RAW_DATASET_STORAGE[dataset_id]
    raw_df = raw_source.copy()
    stream = io.StringIO()
    raw_df.to_csv(stream, index=False)
    stream.seek(0)
    
    meta = METADATA_STORAGE.get(dataset_id, {"filename": "dataset.csv"})
    raw_filename = f"raw_{meta['filename']}"
    
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{raw_filename}"'}
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
    """Execute approved cleaning action, update active dataset, record audit provenance, and advance version."""
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

        # Advance version in committed version history
        current_versions = DATASET_VERSIONS_STORAGE.get(request.dataset_id, [])
        parent_id = current_versions[-1]["version_id"] if current_versions else 0
        new_version_id = parent_id + 1
        DATASET_VERSIONS_STORAGE.setdefault(request.dataset_id, []).append({
            "version_id": new_version_id,
            "parent_version_id": parent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": request.rule,
            "description": f"Applied {request.rule} (strategy: {request.strategy}) on {request.column or 'dataset'}",
            "affected_rows": audit_record.affected_rows,
            "affected_cols": 1 if request.column else len(clean_df.columns),
            "total_rows": len(clean_df),
            "total_cols": len(clean_df.columns)
        })

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


@app.post("/api/clean/propose", response_model=ProposalResponse)
def propose_cleaning_endpoint(request: CleaningActionRequest):
    """
    Generate an uncommitted cleaning proposal with diff preview and consequential-action check.
    Does NOT mutate the active dataset until explicitly approved.
    """
    if request.dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset session not found.")
    
    df = DATASET_STORAGE[request.dataset_id]
    try:
        preview = preview_cleaning_action(df, request)
    except Exception as e:
        logger.exception(f"Proposal preview error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    affected_rows = preview.get("affected_rows", 0)
    affected_pct = preview.get("affected_percentage", 0.0)
    is_consequential = (
        request.strategy == "drop_rows" or
        request.rule in ["drop_rows", "drop_column"] or
        affected_pct >= 5.0
    )
    
    current_versions = DATASET_VERSIONS_STORAGE.get(request.dataset_id, [])
    current_version_id = current_versions[-1]["version_id"] if current_versions else 0

    proposal_id = f"prop_{uuid.uuid4().hex[:8]}"
    proposal = CleaningProposal(
        proposal_id=proposal_id,
        dataset_id=request.dataset_id,
        base_version_id=current_version_id,
        rule=request.rule,
        column=request.column,
        strategy=request.strategy,
        fill_value=request.fill_value,
        reason=f"Proposed {request.rule} on {request.column or 'dataset'} (strategy: {request.strategy})",
        is_consequential=is_consequential,
        affected_rows=affected_rows,
        affected_percentage=affected_pct,
        transformation_summary=preview.get("transformation_summary", f"{request.rule} on {request.column}"),
        diff_preview=preview,
        created_at=datetime.now(timezone.utc).isoformat(),
        status="pending"
    )
    
    if request.dataset_id not in PROPOSED_TRANSFORMATIONS_STORAGE:
        PROPOSED_TRANSFORMATIONS_STORAGE[request.dataset_id] = {}
    PROPOSED_TRANSFORMATIONS_STORAGE[request.dataset_id][proposal_id] = proposal
    
    return ProposalResponse(
        status="staged_pending_approval",
        proposal=proposal,
        message=f"Transformation proposal staged safely. {'[CONSEQUENTIAL ACTION - Requires explicit review]' if is_consequential else 'Ready for approval.'}"
    )


@app.post("/api/clean/approve/{proposal_id}", response_model=CleaningSummaryResponse)
def approve_proposal_endpoint(proposal_id: str):
    """
    Commit an approved transformation proposal, advancing the dataset version and recording provenance.
    Enforces that active dataset version has not advanced since proposal was staged.
    """
    target_proposal: Optional[CleaningProposal] = None
    target_dataset_id: Optional[str] = None
    
    for d_id, proposals in PROPOSED_TRANSFORMATIONS_STORAGE.items():
        if proposal_id in proposals:
            target_proposal = proposals[proposal_id]
            target_dataset_id = d_id
            break
            
    if not target_proposal or not target_dataset_id:
        raise HTTPException(status_code=404, detail="Proposal not found.")
        
    if target_proposal.status != "pending":
        raise HTTPException(status_code=400, detail=f"Proposal cannot be approved; status is already '{target_proposal.status}'.")

    current_versions = DATASET_VERSIONS_STORAGE.get(target_dataset_id, [])
    current_version_id = current_versions[-1]["version_id"] if current_versions else 0
    if target_proposal.base_version_id is not None and target_proposal.base_version_id != current_version_id:
        raise HTTPException(
            status_code=409,
            detail=f"Proposal is stale: created against dataset version {target_proposal.base_version_id}, but current version is {current_version_id}. Please re-preview the proposal against the current dataset version."
        )

    req = CleaningActionRequest(
        dataset_id=target_dataset_id,
        rule=target_proposal.rule,
        column=target_proposal.column,
        strategy=target_proposal.strategy,
        fill_value=target_proposal.fill_value
    )
    
    summary = apply_cleaning_endpoint(req)
    target_proposal.status = "approved"
    return summary


@app.post("/api/clean/reject/{proposal_id}")
def reject_proposal_endpoint(proposal_id: str):
    """
    Reject a transformation proposal without mutating the active dataset.
    """
    target_proposal: Optional[CleaningProposal] = None
    for d_id, proposals in PROPOSED_TRANSFORMATIONS_STORAGE.items():
        if proposal_id in proposals:
            target_proposal = proposals[proposal_id]
            break
            
    if not target_proposal:
        raise HTTPException(status_code=404, detail="Proposal not found.")
        
    if target_proposal.status != "pending":
        raise HTTPException(status_code=400, detail=f"Proposal is already '{target_proposal.status}'.")

    target_proposal.status = "rejected"
    return {
        "status": "rejected",
        "proposal_id": proposal_id,
        "message": "Proposal rejected without mutating the active working dataset."
    }


@app.get("/api/clean/{dataset_id}/proposals", response_model=List[CleaningProposal])
def get_dataset_proposals_endpoint(dataset_id: str):
    """List all staged and historical cleaning proposals for the dataset."""
    if dataset_id not in DATASET_STORAGE and dataset_id not in PROPOSED_TRANSFORMATIONS_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return list(PROPOSED_TRANSFORMATIONS_STORAGE.get(dataset_id, {}).values())


@app.get("/api/clean/{dataset_id}/versions", response_model=List[DatasetVersionInfo])
def get_dataset_versions_endpoint(dataset_id: str):
    """List all committed versions of the dataset from raw ingestion to current state."""
    if dataset_id not in DATASET_STORAGE and dataset_id not in DATASET_VERSIONS_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    
    raw_versions = DATASET_VERSIONS_STORAGE.get(dataset_id, [])
    return [DatasetVersionInfo(**v) for v in raw_versions]


@app.get("/api/clean/{dataset_id}/coercions", response_model=List[CoercionAuditItem])
def get_dataset_coercions_endpoint(dataset_id: str):
    """Retrieve explicit type inference and coercion provenance records."""
    if dataset_id not in COERCION_AUDIT_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return [CoercionAuditItem(**c) for c in COERCION_AUDIT_STORAGE.get(dataset_id, [])]


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
    """Revert active working dataset to original raw state, record version transition, and clear cleaning audit records."""
    if dataset_id not in UNMODIFIED_RAW_STORAGE and dataset_id not in RAW_DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Raw dataset snapshot not found.")

    raw_source = UNMODIFIED_RAW_STORAGE[dataset_id] if dataset_id in UNMODIFIED_RAW_STORAGE else RAW_DATASET_STORAGE[dataset_id]
    raw_copy = raw_source.copy()
    DATASET_STORAGE[dataset_id] = raw_copy.copy()
    AUDIT_LOG_STORAGE[dataset_id] = []
    
    current_versions = DATASET_VERSIONS_STORAGE.get(dataset_id, [])
    parent_id = current_versions[-1]["version_id"] if current_versions else 0
    new_version_id = parent_id + 1
    DATASET_VERSIONS_STORAGE.setdefault(dataset_id, []).append({
        "version_id": new_version_id,
        "parent_version_id": parent_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "reset_to_raw",
        "description": "Reverted active dataset to original raw parsed snapshot",
        "affected_rows": len(raw_copy),
        "affected_cols": len(raw_copy.columns),
        "total_rows": len(raw_copy),
        "total_cols": len(raw_copy.columns)
    })
    
    meta = METADATA_STORAGE.get(dataset_id, {"name": "Dataset", "filename": "data.csv"})
    new_profile = profile_dataframe(raw_copy, dataset_id, meta["name"], meta["filename"])
    PROFILE_STORAGE[dataset_id] = new_profile

    return {"status": "reverted", "rows": len(raw_copy), "health": new_profile.health, "version": new_version_id}


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


# =====================================================================
# STAGE 7: PHASE 3 TOOL REGISTRY, STATISTICAL & GRAPH INTELLIGENCE
# =====================================================================

def build_tool_context(dataset_id: Optional[str]) -> Dict[str, Any]:
    """Assemble session storage and dataset version state into tool execution context."""
    context: Dict[str, Any] = {
        "dataset_storage": DATASET_STORAGE,
        "versions_storage": DATASET_VERSIONS_STORAGE,
        "proposals_storage": PROPOSED_TRANSFORMATIONS_STORAGE,
        "audit_storage": AUDIT_LOG_STORAGE,
        "profile_storage": PROFILE_STORAGE,
        "metadata_storage": METADATA_STORAGE,
        "last_ml_storage": LAST_ML_STORAGE,
    }
    if dataset_id:
        context["dataset_id"] = dataset_id
        if dataset_id not in DATASET_STORAGE:
            # Support on-demand loading of built-in sample datasets for standalone MCP tools
            sample_key = dataset_id[7:] if dataset_id.startswith("sample_") else dataset_id
            if sample_key in SAMPLE_REGISTRY:
                info = SAMPLE_REGISTRY[sample_key]
                sample_df = info["generator"]()
                DATASET_STORAGE[dataset_id] = sample_df.copy()
                RAW_DATASET_STORAGE[dataset_id] = sample_df.copy()
                METADATA_STORAGE[dataset_id] = {
                    "name": info.get("name", sample_key),
                    "filename": f"{sample_key}.csv"
                }
                DATASET_VERSIONS_STORAGE[dataset_id] = [{
                    "version_id": 0,
                    "description": f"Initial loaded sample {sample_key}",
                    "row_count": len(sample_df),
                    "column_count": len(sample_df.columns),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "df": sample_df.copy()
                }]

        context["df"] = DATASET_STORAGE.get(dataset_id)
        current_versions = DATASET_VERSIONS_STORAGE.get(dataset_id, [])
        context["dataset_version"] = current_versions[-1]["version_id"] if current_versions else 0
        context["metadata"] = METADATA_STORAGE.get(dataset_id, {"name": "Dataset", "filename": "data.csv"})
    return context


@app.get("/api/tools/list")
def list_tools_endpoint():
    """Retrieve catalog of all registered data-science tools with Pydantic JSON schemas."""
    return default_registry.list_tools()


@app.post("/api/tools/execute", response_model=ToolExecutionResult)
def execute_tool_endpoint(request: ToolExecutionApiRequest):
    """Invoke a registered data-science tool with structured parameters and verified session context."""
    context = build_tool_context(request.dataset_id)
    return default_registry.execute_tool(request.tool_name, request.arguments, context)


@app.post("/api/statistical/test", response_model=HypothesisTestResponse)
def run_statistical_test_endpoint(request: HypothesisTestRequest):
    """Execute formal hypothesis test with assumption validation, effect sizes, and caveats."""
    if request.dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset session not found.")

    context = build_tool_context(request.dataset_id)
    tool_args = request.dict() if hasattr(request, "dict") else request.model_dump()
    exec_res = default_registry.execute_tool("run_statistical_test", tool_args, context)

    if exec_res.status == "error":
        raise HTTPException(status_code=400, detail=exec_res.error_details or "Hypothesis test execution failed.")

    return HypothesisTestResponse(**exec_res.result)


@app.post("/api/graph/analyze", response_model=GraphIntelligenceResponse)
def analyze_graph_endpoint(request: GraphIntelligenceAnalyzeRequest):
    """Analyze underlying dataset behind the selected chart and return data-grounded intelligence."""
    if request.dataset_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset session not found.")

    context = build_tool_context(request.dataset_id)
    tool_args = {
        "dataset_id": request.dataset_id,
        "x_column": request.x_column,
        "y_column": request.y_column,
        "chart_type": request.chart_type,
        "aggregation": request.aggregation or "mean",
        "chart_spec": request.chart_spec
    }

    exec_res = default_registry.execute_tool("analyze_chart", tool_args, context)
    if exec_res.status == "error":
        raise HTTPException(status_code=400, detail=exec_res.error_details or "Graph analysis failed.")

    return GraphIntelligenceResponse(**exec_res.result)


@app.post("/api/graph/ask", response_model=GraphIntelligenceAskResponse)
def ask_graph_endpoint(request: GraphIntelligenceAskRequest):
    """
    Answer analytical questions about a graph using actual underlying dataset observations,
    relevant computed results, and statistical caveats via the Runtime AI Analyst.
    """
    ds_id = request.dataset_id
    if ds_id not in DATASET_STORAGE:
        # Check case-insensitive / normalized match in DATASET_STORAGE and METADATA_STORAGE
        t_clean = (ds_id or "").lower().replace("+", "").replace(".", "").replace("-", "").replace("_", "")
        for k in DATASET_STORAGE:
            k_clean = k.lower().replace("+", "").replace(".", "").replace("-", "").replace("_", "")
            if t_clean and (t_clean in k_clean or k_clean in t_clean):
                ds_id = k
                break
        if ds_id not in DATASET_STORAGE:
            for k, meta in METADATA_STORAGE.items():
                m_name = (meta.get("name") or "").lower().replace("+", "").replace(".", "").replace("-", "").replace("_", "")
                m_fn = (meta.get("filename") or "").lower().replace("+", "").replace(".", "").replace("-", "").replace("_", "")
                if t_clean and (t_clean in m_name or t_clean in m_fn or "glb" in t_clean):
                    ds_id = k
                    break
        # Auto-load GLB.Ts+dSST.csv if referencing GLB and not in storage
        if ds_id not in DATASET_STORAGE and ("glb" in t_clean or not DATASET_STORAGE):
            glb_path = os.path.join(os.path.dirname(__file__), "data", "GLB.Ts+dSST.csv")
            if os.path.exists(glb_path):
                with open(glb_path, "rb") as f:
                    csv_content = f.read()
                loaded_df, _ = safe_read_csv(csv_content)
                ds_id = "glb_ts_dsst"
                register_dataset_session(
                    dataset_id=ds_id,
                    name="GLB.Ts+dSST",
                    filename="GLB.Ts+dSST.csv",
                    content=csv_content,
                    unmodified_df=loaded_df
                )

    if ds_id not in DATASET_STORAGE and len(DATASET_STORAGE) == 1:
        ds_id = next(iter(DATASET_STORAGE.keys()))

    if ds_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail="Dataset session not found.")

    request.dataset_id = ds_id
    context = build_tool_context(ds_id)
    return analyst_agent.ask_graph_analyst(request, context)


@app.post("/api/agent/chat", response_model=GraphIntelligenceAskResponse)
def agent_chat_endpoint(request: AgentChatRequest):
    """
    Live AI Agent Chat endpoint.
    Accepts dataset ('Glb.Ts+Dsst' or dataset_id) and message/question,
    auto-resolves/loads the dataset session, executes the AI Analyst agent with real tools,
    and returns the grounded structured response and tool trace.
    """
    target_ds = request.dataset_id or request.dataset or ""
    ds_id = None

    if target_ds in DATASET_STORAGE:
        ds_id = target_ds
    else:
        # Check case-insensitive / normalized match in DATASET_STORAGE and METADATA_STORAGE
        t_clean = target_ds.lower().replace("+", "").replace(".", "").replace("-", "").replace("_", "")
        for k in DATASET_STORAGE:
            k_clean = k.lower().replace("+", "").replace(".", "").replace("-", "").replace("_", "")
            if t_clean and (t_clean in k_clean or k_clean in t_clean):
                ds_id = k
                break
        if not ds_id:
            for k, meta in METADATA_STORAGE.items():
                m_name = (meta.get("name") or "").lower().replace("+", "").replace(".", "").replace("-", "").replace("_", "")
                m_fn = (meta.get("filename") or "").lower().replace("+", "").replace(".", "").replace("-", "").replace("_", "")
                if t_clean and (t_clean in m_name or t_clean in m_fn or "glb" in t_clean):
                    ds_id = k
                    break

        # Auto-load GLB.Ts+dSST.csv if referencing GLB and not in storage
        if not ds_id and ("glb" in t_clean or not DATASET_STORAGE):
            glb_path = os.path.join(os.path.dirname(__file__), "data", "GLB.Ts+dSST.csv")
            if os.path.exists(glb_path):
                with open(glb_path, "rb") as f:
                    csv_content = f.read()
                loaded_df, _ = safe_read_csv(csv_content)
                ds_id = "glb_ts_dsst"
                register_dataset_session(
                    dataset_id=ds_id,
                    name="GLB.Ts+dSST",
                    filename="GLB.Ts+dSST.csv",
                    content=csv_content,
                    unmodified_df=loaded_df
                )

    if not ds_id or ds_id not in DATASET_STORAGE:
        raise HTTPException(status_code=404, detail=f"Dataset '{target_ds}' not found in active session.")

    df = DATASET_STORAGE[ds_id]
    q = request.message or request.question or "What is the trend?"

    x_col = request.x_column
    y_col = request.y_column
    if not x_col:
        cols_lower = [str(c).lower() for c in df.columns]
        if "year" in cols_lower:
            x_col = df.columns[cols_lower.index("year")]
        else:
            time_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["date", "time", "year", "ds"])]
            x_col = time_cols[0] if time_cols else df.columns[0]
    if not y_col:
        if "J-D" in df.columns:
            y_col = "J-D"
        else:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != x_col]
            y_col = str(num_cols[0]) if num_cols else (str(df.columns[0]) if df.columns[0] != x_col else None)

    ask_req = GraphIntelligenceAskRequest(
        dataset_id=ds_id,
        x_column=x_col,
        y_column=y_col,
        chart_type=request.chart_type or "line",
        question=q,
        api_key=request.api_key,
        conversation_history=request.conversation_history
    )
    context = build_tool_context(ds_id)
    return analyst_agent.ask_graph_analyst(ask_req, context)




