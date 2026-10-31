"""
Tests for NASA GISTEMP CSV Ingestion & Unified Record Count
Verifies:
1. Parsing: 147 rows of data are read, and "***" becomes NaN.
2. Years: min year = 1880, max year with J-D present = 2025.
3. Annual series length: exactly 146 non-null J-D values.
4. Record count: every screen/endpoint shows 146 (or clearly labeled '146 complete years, 1 partial year excluded').
   No screen shows 148 or 48.
5. Missing monthly cells in the table: exactly 7 (before any filtering).
6. Raw file bytes are unchanged after upload.
"""
import os
import pytest
import pandas as pd

from data_utils import safe_read_csv
from main import (
    register_dataset_session,
    RAW_BYTE_STORAGE,
    PARSED_DATASET_STORAGE,
    DATASET_STORAGE,
    PROFILE_STORAGE,
    METADATA_STORAGE,
    DATASET_VERSIONS_STORAGE,
    get_dataset_preview,
    generate_custom_chart,
    CustomChartRequest
)


@pytest.fixture
def glb_bytes_and_path():
    path = os.path.join(os.path.dirname(__file__), "data", "GLB.Ts+dSST.csv")
    assert os.path.exists(path), f"File {path} does not exist"
    with open(path, "rb") as f:
        content = f.read()
    return content, path


def test_1_parsing_147_rows_and_stars_become_nan(glb_bytes_and_path):
    """Test 1: Parsing: 147 rows of data are read, and '***' becomes NaN."""
    content, _ = glb_bytes_and_path
    df, banner = safe_read_csv(content)
    
    # 147 data rows (1880 through 2026)
    assert len(df) == 147, f"Expected 147 rows, got {len(df)}"
    assert banner == "Land-Ocean: Global Means"

    # Verify that "***" string was converted to NaN
    # Check 1880 row which had *** for D-N and DJF
    row_1880 = df[df["Year"].astype(str) == "1880"].iloc[0]
    assert pd.isna(row_1880["D-N"])
    assert pd.isna(row_1880["DJF"])
    
    # Check 2026 row which had *** for June through December
    row_2026 = df[df["Year"].astype(str) == "2026"].iloc[0]
    assert pd.isna(row_2026["Jun"])
    assert pd.isna(row_2026["Dec"])
    assert pd.isna(row_2026["J-D"])


def test_2_years_range(glb_bytes_and_path):
    """Test 2: Years: min year = 1880, max year with J-D present = 2025."""
    content, _ = glb_bytes_and_path
    df, _ = safe_read_csv(content)
    
    year_col = pd.to_numeric(df["Year"], errors="coerce")
    jd_col = pd.to_numeric(df["J-D"], errors="coerce")
    
    assert int(year_col.min()) == 1880
    
    # Max year with J-D present must be 2025
    years_with_jd = year_col[jd_col.notna()]
    assert int(years_with_jd.max()) == 2025


def test_3_annual_series_length_exactly_146(glb_bytes_and_path):
    """Test 3: Annual series length: exactly 146 non-null J-D values."""
    content, _ = glb_bytes_and_path
    df, _ = safe_read_csv(content)
    jd_col = pd.to_numeric(df["J-D"], errors="coerce")
    
    non_null_jd_count = int(jd_col.notna().sum())
    assert non_null_jd_count == 146, f"Expected 146 non-null J-D values, got {non_null_jd_count}"


def test_4_record_count_single_source_of_truth_everywhere(glb_bytes_and_path):
    """
    Test 4: Record count: every screen shows 146 (or a clearly labeled value
    like '146 complete years, 1 partial year excluded'). No screen shows 148 or 48.
    """
    content, _ = glb_bytes_and_path
    df, _ = safe_read_csv(content)
    
    ds_id = "test_glb_session"
    report = register_dataset_session(
        dataset_id=ds_id,
        name="GLB.Ts+dSST",
        filename="GLB.Ts+dSST.csv",
        content=content,
        unmodified_df=df
    )
    
    # 1. Dataset Object & Profile Metadata (Data Quality & Clean Data)
    assert report.metadata.rows == 146
    assert report.metadata.record_count == 146
    assert "146 complete years" in report.metadata.record_count_label
    assert report.metadata.rows != 148
    assert report.metadata.rows != 48

    # 2. In-memory Working Dataset (Predict, ML, Explore)
    working_df = DATASET_STORAGE[ds_id]
    assert len(working_df) == 146
    assert len(working_df) != 148
    assert len(working_df) != 48

    # 3. Metadata Storage (Single Source of Truth)
    meta = METADATA_STORAGE[ds_id]
    assert meta["rows"] == 146
    assert meta["record_count"] == 146
    assert meta["rows"] != 148
    assert meta["rows"] != 48

    # 4. Preview Endpoint (Data Grid & Preview Rows)
    preview = get_dataset_preview(dataset_id=ds_id, page=1, page_size=20)
    assert preview["total_rows"] == 146
    assert preview["total_rows"] != 148
    assert preview["total_rows"] != 48

    # 5. Chart Spec (Explore / Visualizer Studio)
    chart_req = CustomChartRequest(
        dataset_id=ds_id,
        x_column="Year",
        y_column="J-D",
        chart_type="line"
    )
    chart_spec = generate_custom_chart(working_df, chart_req)
    assert chart_spec.rows_used == 146
    assert chart_spec.rows_used != 148
    assert chart_spec.rows_used != 48

    # 6. Committed Versions (Clean Studio Version History)
    versions = DATASET_VERSIONS_STORAGE[ds_id]
    # Final active version has 146 rows
    latest_v = versions[-1]
    assert latest_v["total_rows"] == 146


def test_5_missing_monthly_cells_in_table_exactly_7(glb_bytes_and_path):
    """Test 5: Missing monthly cells in the table: exactly 7 (before any filtering)."""
    content, _ = glb_bytes_and_path
    df, _ = safe_read_csv(content)
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_cols = [c for c in df.columns if c in months]
    assert len(monthly_cols) == 12, f"Expected 12 monthly columns, found {monthly_cols}"
    
    missing_monthly = int(df[monthly_cols].isna().sum().sum())
    assert missing_monthly == 7, f"Expected exactly 7 missing monthly cells, got {missing_monthly}"


def test_6_raw_file_bytes_unchanged_after_upload(glb_bytes_and_path):
    """Test 6: Raw file bytes are unchanged after upload."""
    content, _ = glb_bytes_and_path
    df, _ = safe_read_csv(content)
    
    ds_id = "test_raw_bytes_preservation"
    register_dataset_session(
        dataset_id=ds_id,
        name="GLB.Ts+dSST",
        filename="GLB.Ts+dSST.csv",
        content=content,
        unmodified_df=df
    )
    
    stored_bytes = RAW_BYTE_STORAGE[ds_id]
    assert stored_bytes == content, "Raw byte storage altered uploaded file bytes!"
    assert len(stored_bytes) == len(content)

    # Parsed data stored separately
    parsed_stored = PARSED_DATASET_STORAGE[ds_id]
    assert len(parsed_stored) == 147
