"""
DataSense AI V6 — Table Relationship & Safe Ingestion Engine
Handles:
1. Safe ZIP extraction (path traversal defense, decompression bomb check, extension validation).
2. Excel workbook worksheet inspection.
3. Cross-table schema analysis, candidate join key discovery, value overlap calculation, and relationship classification.
4. Deterministic Join Preview with Cartesian multiplication risk detection.
5. Grounded join execution.
"""
import io
import os
import zipfile
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
import pandas as pd
import numpy as np

from models import (
    WorksheetInfo,
    WorkbookInspectResponse,
    TableInventoryItem,
    CandidateJoinKey,
    TableRelationshipItem,
    RelationshipDetectionResponse,
    JoinPreviewResponse
)
from data_profiler import infer_column_type, sanitize_for_json

logger = logging.getLogger("datasense.table_relationships")

SUPPORTED_DATA_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
FORBIDDEN_EXTENSIONS = {".exe", ".bat", ".cmd", ".sh", ".py", ".js", ".vbs", ".ps1", ".jar", ".dll", ".so", ".bin"}


def safe_extract_zip(
    zip_bytes: bytes,
    max_uncompressed_bytes: int = 100 * 1024 * 1024,  # 100 MB max uncompressed
    max_files: int = 50
) -> List[Tuple[str, bytes]]:
    """
    Safely extract supported data files from a ZIP archive in-memory.
    Guards against:
    - Path traversal attacks (leading slashes, '..' components, absolute paths)
    - Decompression bombs (sum of uncompressed sizes > threshold)
    - Excessive file counts (> max_files)
    - Unsupported / executable / dangerous file types
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as e:
        raise ValueError(f"Uploaded file is not a valid ZIP archive: {str(e)}")

    infolist = zf.infolist()
    if len(infolist) == 0:
        raise ValueError("ZIP archive is empty.")

    if len(infolist) > max_files:
        raise ValueError(f"ZIP archive contains {len(infolist)} files, exceeding the limit of {max_files} files.")

    total_uncompressed = sum(info.file_size for info in infolist)
    if total_uncompressed > max_uncompressed_bytes:
        raise ValueError(
            f"Decompression bomb risk: Archive total uncompressed size ({total_uncompressed / (1024 * 1024):.1f} MB) "
            f"exceeds maximum allowed limit of {max_uncompressed_bytes / (1024 * 1024):.0f} MB."
        )

    extracted_files: List[Tuple[str, bytes]] = []

    for info in infolist:
        name = info.filename
        # Ignore macOS metadata folders and directory markers
        if name.startswith("__MACOSX/") or name.endswith("/"):
            continue

        # Path traversal checks
        if ".." in name or os.path.isabs(name) or name.startswith(("/", "\\")):
            raise ValueError(f"Unsafe archive member '{name}': Path traversal detected.")

        basename = os.path.basename(name)
        if not basename or basename.startswith("."):
            continue

        ext = os.path.splitext(basename)[1].lower()
        if ext in FORBIDDEN_EXTENSIONS:
            raise ValueError(f"Forbidden file type '{ext}' in archive member '{basename}'. Executable files are strictly rejected.")

        if ext in SUPPORTED_DATA_EXTENSIONS:
            file_data = zf.read(info)
            extracted_files.append((basename, file_data))

    if not extracted_files:
        raise ValueError("ZIP archive contains no supported data files (.csv, .xlsx, .xls).")

    return extracted_files


def inspect_excel_worksheets(content: bytes, filename: str) -> WorkbookInspectResponse:
    """
    Inspect an Excel workbook's worksheets without committing them.
    Returns row counts, column counts, column names, preview, and empty/non-empty status.
    """
    try:
        excel_file = pd.ExcelFile(io.BytesIO(content))
    except Exception as e:
        raise ValueError(f"Failed to read Excel workbook: The file is invalid, corrupted, or password-protected ({str(e)}).")

    sheet_names = excel_file.sheet_names
    if not sheet_names:
        raise ValueError("Uploaded Excel workbook contains no worksheets.")

    sheets_info: List[WorksheetInfo] = []
    for s_name in sheet_names:
        try:
            df = pd.read_excel(excel_file, sheet_name=s_name)
            rows = len(df)
            cols = len(df.columns)
            col_names = [str(c) for c in df.columns]
            is_empty = rows == 0
            
            # First 5 preview rows
            preview = []
            if not is_empty:
                preview = [
                    {str(c): sanitize_for_json(val) for c, val in row.items()}
                    for row in df.head(5).to_dict(orient="records")
                ]

            sheets_info.append(WorksheetInfo(
                sheet_name=s_name,
                rows=rows,
                columns=cols,
                column_names=col_names,
                is_empty=is_empty,
                preview_rows=preview
            ))
        except Exception as e:
            logger.warning(f"Error inspecting sheet '{s_name}': {e}")
            sheets_info.append(WorksheetInfo(
                sheet_name=s_name,
                rows=0,
                columns=0,
                column_names=[],
                is_empty=True,
                preview_rows=[]
            ))

    return WorkbookInspectResponse(
        filename=filename,
        total_sheets=len(sheet_names),
        sheets=sheets_info
    )


def detect_table_relationships(
    tables: Dict[str, Tuple[str, pd.DataFrame]]
) -> RelationshipDetectionResponse:
    """
    Inspect schemas and value distributions across multiple tables.
    Identifies candidate join keys using:
    - Column name similarity (exact, id suffixes, prefix matches)
    - Compatible data types (both numeric, or both string)
    - Key uniqueness in Table A and Table B
    - Actual value overlap between non-null unique values

    Classifies relationships as:
    - 'related' (high or moderate confidence key with substantial overlap)
    - 'ambiguous' (low overlap or multiple conflicting candidate keys)
    - 'independent' (no candidate keys or 0 value overlap)
    """
    table_ids = list(tables.keys())
    relationships: List[TableRelationshipItem] = []

    for i in range(len(table_ids)):
        for j in range(i + 1, len(table_ids)):
            id_a, id_b = table_ids[i], table_ids[j]
            name_a, df_a = tables[id_a]
            name_b, df_b = tables[id_b]

            candidate_keys: List[CandidateJoinKey] = []

            for col_a in df_a.columns:
                series_a = df_a[col_a].dropna()
                if len(series_a) == 0:
                    continue
                type_a = infer_column_type(series_a)
                clean_name_a = str(col_a).strip().lower()

                for col_b in df_b.columns:
                    series_b = df_b[col_b].dropna()
                    if len(series_b) == 0:
                        continue
                    type_b = infer_column_type(series_b)
                    clean_name_b = str(col_b).strip().lower()

                    # 1. Type compatibility check
                    types_compatible = (
                        (type_a == type_b) or
                        (type_a == "numeric" and type_b == "numeric") or
                        (type_a in ["categorical", "text"] and type_b in ["categorical", "text"])
                    )
                    if not types_compatible:
                        continue

                    # 2. Name match heuristics
                    exact_name_match = (clean_name_a == clean_name_b)
                    id_pattern_match = (
                        clean_name_a.endswith("id") and clean_name_b.endswith("id")
                    ) or (
                        clean_name_a == "id" or clean_name_b == "id"
                    )

                    # Only test overlap if there is a name signal or high cardinality
                    if not (exact_name_match or id_pattern_match):
                        continue

                    # 3. Compute Value Overlap
                    # Sample up to 10,000 unique values for performance
                    set_a: Set[Any] = set(series_a.astype(str).unique()[:10000])
                    set_b: Set[Any] = set(series_b.astype(str).unique()[:10000])

                    intersection = set_a & set_b
                    overlap_count = len(intersection)
                    if overlap_count == 0:
                        continue

                    overlap_pct_a = round((overlap_count / len(set_a)) * 100, 2) if len(set_a) > 0 else 0.0
                    overlap_pct_b = round((overlap_count / len(set_b)) * 100, 2) if len(set_b) > 0 else 0.0

                    is_unique_a = bool(df_a[col_a].is_unique)
                    is_unique_b = bool(df_b[col_b].is_unique)

                    # Classify relationship cardinality
                    if is_unique_a and is_unique_b:
                        rel_type = "one-to-one"
                    elif is_unique_a and not is_unique_b:
                        rel_type = "one-to-many"
                    elif not is_unique_a and is_unique_b:
                        rel_type = "many-to-one"
                    else:
                        rel_type = "many-to-many"

                    # Confidence scoring
                    max_overlap = max(overlap_pct_a, overlap_pct_b)
                    if exact_name_match and max_overlap >= 40.0:
                        confidence = "high"
                    elif max_overlap >= 15.0:
                        confidence = "moderate"
                    else:
                        confidence = "low"

                    explanation = (
                        f"Found {overlap_count} overlapping unique keys between '{col_a}' in {name_a} "
                        f"({overlap_pct_a}% of unique values) and '{col_b}' in {name_b} ({overlap_pct_b}%). "
                        f"Key uniqueness indicates a {rel_type} relationship."
                    )

                    candidate_keys.append(CandidateJoinKey(
                        table_a_column=str(col_a),
                        table_b_column=str(col_b),
                        data_type_a=type_a,
                        data_type_b=type_b,
                        is_unique_a=is_unique_a,
                        is_unique_b=is_unique_b,
                        overlap_count=overlap_count,
                        overlap_percentage_a=overlap_pct_a,
                        overlap_percentage_b=overlap_pct_b,
                        relationship_type=rel_type,
                        confidence=confidence,
                        explanation=explanation
                    ))

            # Determine overall relationship status between Table A and Table B
            if not candidate_keys:
                status = "independent"
                summary = f"No candidate join keys or overlapping identifiers detected between '{name_a}' and '{name_b}'."
            elif any(k.confidence == "high" for k in candidate_keys):
                status = "related"
                top_key = max(candidate_keys, key=lambda k: max(k.overlap_percentage_a, k.overlap_percentage_b))
                summary = (
                    f"Strong relationship detected between '{name_a}' and '{name_b}' on key "
                    f"'{top_key.table_a_column}' == '{top_key.table_b_column}' ({top_key.relationship_type})."
                )
            elif any(k.confidence == "moderate" for k in candidate_keys):
                status = "related"
                top_key = max(candidate_keys, key=lambda k: max(k.overlap_percentage_a, k.overlap_percentage_b))
                summary = (
                    f"Moderate relationship detected between '{name_a}' and '{name_b}' on key "
                    f"'{top_key.table_a_column}' == '{top_key.table_b_column}' ({top_key.relationship_type})."
                )
            else:
                status = "ambiguous"
                summary = (
                    f"Ambiguous relationship: Low key overlap (<15%) or conflicting join keys found between '{name_a}' and '{name_b}'. "
                    f"Independent analysis recommended unless explicit join is verified."
                )

            relationships.append(TableRelationshipItem(
                table_a_id=id_a,
                table_a_name=name_a,
                table_b_id=id_b,
                table_b_name=name_b,
                status=status,
                candidate_keys=candidate_keys,
                evidence_summary=summary
            ))

    return RelationshipDetectionResponse(relationships=relationships)


def compute_join_preview(
    df_a: pd.DataFrame,
    name_a: str,
    df_b: pd.DataFrame,
    name_b: str,
    col_a: str,
    col_b: str,
    join_type: str = "inner"
) -> JoinPreviewResponse:
    """
    Compute a non-mutating preview of a proposed join between two tables.
    Checks:
    - Row count before and after
    - Unmatched rows on left and right
    - Cartesian multiplication risk (warns if non-unique keys on both sides will cause row explosion)
    - Returns preview rows of joined output
    """
    if col_a not in df_a.columns:
        raise ValueError(f"Join column '{col_a}' not found in table '{name_a}'.")
    if col_b not in df_b.columns:
        raise ValueError(f"Join column '{col_b}' not found in table '{name_b}'.")

    valid_join_types = {"inner", "left", "right", "outer"}
    if join_type not in valid_join_types:
        raise ValueError(f"Invalid join type '{join_type}'. Must be one of: {', '.join(valid_join_types)}.")

    # Cast keys to common string representation for safe joining
    left_keys = df_a[col_a].astype(str)
    right_keys = df_b[col_b].astype(str)

    # Compute unmatched
    unmatched_a = int((~left_keys.isin(right_keys)).sum())
    unmatched_b = int((~right_keys.isin(left_keys)).sum())

    # Check Cartesian risk
    left_counts = left_keys.value_counts()
    right_counts = right_keys.value_counts()
    common_keys = set(left_counts.index) & set(right_counts.index)
    
    cartesian_risk = False
    risk_warning = None
    if common_keys:
        max_left_dup = max([left_counts[k] for k in common_keys])
        max_right_dup = max([right_counts[k] for k in common_keys])
        if max_left_dup > 1 and max_right_dup > 1:
            cartesian_risk = True
            risk_warning = (
                f"Cartesian multiplication risk: Non-unique keys detected on both sides (up to {max_left_dup} copies in '{name_a}' "
                f"and {max_right_dup} copies in '{name_b}'). This join may substantially multiply row counts."
            )

    # Perform sample/preview merge
    # Prepare temporary DataFrames with string-keyed merge columns to avoid type mismatch errors
    temp_a = df_a.copy()
    temp_b = df_b.copy()
    temp_a["_ds_join_key"] = left_keys
    temp_b["_ds_join_key"] = right_keys

    try:
        merged_df = pd.merge(
            temp_a,
            temp_b,
            on="_ds_join_key",
            how=join_type,
            suffixes=(f"_{name_a[:6]}", f"_{name_b[:6]}")
        )
    except Exception as e:
        raise ValueError(f"Failed to compute join preview: {str(e)}")

    merged_df = merged_df.drop(columns=["_ds_join_key"], errors="ignore")
    expected_rows = len(merged_df)

    cols_before = [f"{name_a}.{c}" for c in df_a.columns] + [f"{name_b}.{c}" for c in df_b.columns]
    cols_after = [str(c) for c in merged_df.columns]

    preview_rows = [
        {str(c): sanitize_for_json(val) for c, val in row.items()}
        for row in merged_df.head(5).to_dict(orient="records")
    ]

    return JoinPreviewResponse(
        table_a_name=name_a,
        table_b_name=name_b,
        join_column_a=col_a,
        join_column_b=col_b,
        join_type=join_type,
        table_a_rows=len(df_a),
        table_b_rows=len(df_b),
        expected_rows=expected_rows,
        unmatched_rows_a=unmatched_a,
        unmatched_rows_b=unmatched_b,
        cartesian_multiplication_risk=cartesian_risk,
        risk_warning=risk_warning,
        columns_before_join=cols_before,
        columns_after_join=cols_after,
        preview_rows=preview_rows
    )


def execute_join(
    df_a: pd.DataFrame,
    name_a: str,
    df_b: pd.DataFrame,
    name_b: str,
    col_a: str,
    col_b: str,
    join_type: str = "inner"
) -> pd.DataFrame:
    """
    Execute a validated join between two DataFrames.
    Returns the merged DataFrame with clean column names.
    """
    if col_a not in df_a.columns:
        raise ValueError(f"Join column '{col_a}' not found in table '{name_a}'.")
    if col_b not in df_b.columns:
        raise ValueError(f"Join column '{col_b}' not found in table '{name_b}'.")

    temp_a = df_a.copy()
    temp_b = df_b.copy()

    # If types differ, cast to string safely
    if df_a[col_a].dtype != df_b[col_b].dtype:
        temp_a["_ds_join_key"] = temp_a[col_a].astype(str)
        temp_b["_ds_join_key"] = temp_b[col_b].astype(str)
        left_on = "_ds_join_key"
        right_on = "_ds_join_key"
    else:
        left_on = col_a
        right_on = col_b

    suf_a = f"_{name_a[:6]}" if name_a else "_left"
    suf_b = f"_{name_b[:6]}" if name_b else "_right"

    merged = pd.merge(
        temp_a,
        temp_b,
        left_on=left_on,
        right_on=right_on,
        how=join_type,
        suffixes=(suf_a, suf_b)
    )

    if "_ds_join_key" in merged.columns:
        merged = merged.drop(columns=["_ds_join_key"])

    # Sanitize column names
    clean_cols = []
    col_counts: Dict[str, int] = {}
    for idx, raw_c in enumerate(merged.columns):
        c_name = str(raw_c).strip() if raw_c is not None and str(raw_c).strip() else f"col_{idx+1}"
        if c_name in col_counts:
            col_counts[c_name] += 1
            c_name = f"{c_name}_{col_counts[c_name]}"
        else:
            col_counts[c_name] = 1
        clean_cols.append(c_name)
    merged.columns = clean_cols

    return merged
