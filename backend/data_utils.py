"""
DataSense AI — Tabular Ingestion & Header Normalization Utilities
Safely parses CSV and Excel headers, detects metadata title banners,
flattens MultiIndex headers/indices, and deterministically deduplicates columns.
"""
import io
import csv
import logging
from typing import Tuple, Optional, List, Dict, Iterable, Any
import pandas as pd

logger = logging.getLogger("datasense.data_utils")


def flatten_multiindex_columns(columns: pd.MultiIndex) -> List[str]:
    """Flatten a MultiIndex column structure into single, descriptive strings."""
    clean_cols: List[str] = []
    for col in columns:
        parts = [
            str(lvl).strip()
            for lvl in col
            if lvl is not None and str(lvl).strip() and not str(lvl).startswith("Unnamed:")
        ]
        clean_cols.append("_".join(parts) if parts else "column")
    return clean_cols


def deduplicate_column_names(columns: Iterable[Any]) -> List[str]:
    """
    Deterministically deduplicate column names without overwriting or merging data.
    e.g. ['A', 'A', 'B'] -> ['A', 'A_2', 'B']
    """
    seen: Dict[str, int] = {}
    clean_cols: List[str] = []
    for idx, raw_c in enumerate(columns):
        base_name = str(raw_c).strip() if raw_c is not None and str(raw_c).strip() else f"column_{idx+1}"
        if base_name in seen:
            seen[base_name] += 1
            clean_cols.append(f"{base_name}_{seen[base_name]}")
        else:
            seen[base_name] = 1
            clean_cols.append(base_name)
    return clean_cols


def normalize_dataframe_headers_and_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has a clean, 1D flat Index and 1D unique, non-empty column names:
    - If index is a MultiIndex, resets index so index levels are preserved as table columns.
    - If columns is a MultiIndex, flattens levels into single strings.
    - Deterministically deduplicates any duplicate column names.
    """
    clean_df = df.copy()

    # 1. Reset MultiIndex row index if present
    if isinstance(clean_df.index, pd.MultiIndex):
        clean_df = clean_df.reset_index()

    # 2. Flatten MultiIndex columns if present
    if isinstance(clean_df.columns, pd.MultiIndex):
        clean_df.columns = flatten_multiindex_columns(clean_df.columns)

    # 3. Deterministically deduplicate column names
    clean_df.columns = deduplicate_column_names(clean_df.columns)

    return clean_df


def safe_read_csv(content: bytes, delimiter: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Safely parse CSV/TSV/TXT content with intelligent metadata banner detection:
    - Automatically detects single-token title banners (e.g. 'Land-Ocean: Global Means')
      where line 0 has 1 column and line 1 has multiple columns, skipping row 0.
    - Tries UTF-8-sig, UTF-8, and Latin-1 encodings seamlessly.
    - Flattens MultiIndex headers and ensures unique column names.
    - Returns (df, detected_banner_title).
    """
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            text = content.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        text = content.decode("latin-1", errors="replace")

    raw_lines = text.splitlines()
    if not any(l.strip() for l in raw_lines):
        raise pd.errors.EmptyDataError("Uploaded file contains no data rows.")

    # 1. Dynamic Header Line Detection:
    # Search for the line that starts with "Year,", not by position.
    header_idx = None
    for idx, line in enumerate(raw_lines):
        line_clean = line.strip()
        if line_clean.lower().startswith("year,"):
            header_idx = idx
            break

    skip = 0
    banner_title = None

    if header_idx is not None:
        skip = header_idx
        if header_idx > 0:
            banner_title = " ".join(raw_lines[i].strip() for i in range(header_idx) if raw_lines[i].strip())
            logger.info(f"Detected header starting with 'Year,' at line index {header_idx}. Description line: '{banner_title}'")
    else:
        # Fallback to general title banner detection
        start_idx = 0
        while start_idx < len(raw_lines) and not raw_lines[start_idx].strip():
            start_idx += 1
        non_empty_lines = [l for l in raw_lines[start_idx:] if l.strip()]
        skip = start_idx
        if len(non_empty_lines) >= 2:
            try:
                delim_cand = delimiter or ("\t" if "\t" in non_empty_lines[1] and "," not in non_empty_lines[1] else ",")
                r0 = next(csv.reader([non_empty_lines[0]], delimiter=delim_cand))
                r1 = next(csv.reader([non_empty_lines[1]], delimiter=delim_cand))
                r0_valid = [c for c in r0 if c.strip()]
                r1_valid = [c for c in r1 if c.strip()]
                if len(r0_valid) == 1 and len(r1_valid) > 1:
                    skip = start_idx + 1
                    banner_title = r0_valid[0].strip()
                    logger.info(f"Detected and skipped metadata title banner: '{banner_title}'")
            except Exception as e:
                logger.debug(f"CSV header inspection note: {e}")

    # Auto-detect tab delimiter if not provided
    delim = delimiter
    if delim is None and len(raw_lines) > skip:
        sample_line = raw_lines[skip]
        if "\t" in sample_line and "," not in sample_line:
            delim = "\t"

    read_kwargs: Dict[str, Any] = {
        "na_values": ["***", "NA", "N/A", "null", "-"]
    }
    if delim:
        read_kwargs["sep"] = delim

    try:
        df = pd.read_csv(io.StringIO(text), skiprows=skip, **read_kwargs)
    except Exception:
        df = pd.read_csv(io.StringIO(text), **read_kwargs)

    df = normalize_dataframe_headers_and_columns(df)

    # 2. Monthly Columns & Missing Cell Detection (before any filtering)
    standard_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_cols = [c for c in df.columns if str(c).strip().title() in standard_months]
    missing_monthly_cells = int(df[monthly_cols].isna().sum().sum()) if monthly_cols else 0

    # 3. Detect partial years: a year is partial if it has any "***" in its monthly columns.
    # Keep the flag, do not drop it silently.
    if monthly_cols:
        df["is_partial_year"] = df[monthly_cols].isna().any(axis=1)

    # Attach parsing metadata to DataFrame attributes
    df.attrs["raw_rows"] = len(df)
    df.attrs["missing_monthly_cells"] = missing_monthly_cells
    df.attrs["banner_title"] = banner_title
    df.attrs["monthly_cols"] = monthly_cols

    return df, banner_title
