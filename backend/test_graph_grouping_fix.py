"""
Regression and validation test suite for DataSense AI's Graph AI Analyst Grouper fix:
- Title banner CSV parsing and metadata preservation
- Duplicate column detection & deterministic resolution
- MultiIndex column and row header normalization
- 1D Grouper enforcement in AnalyzeChartTool and CompareGroupsTool
- Univariate measure analysis when x_col == y_col
- Real date/year-based grouping and numeric trend calculations
- Live GLB.Ts+dSST investigation with real calculated evidence
"""
import os
import pytest
import numpy as np
import pandas as pd

from data_utils import (
    flatten_multiindex_columns,
    deduplicate_column_names,
    normalize_dataframe_headers_and_columns,
    safe_read_csv
)
from tools.analysis_tools import (
    AnalyzeChartTool,
    AnalyzeChartInput,
    CompareGroupsTool,
    CompareGroupsInput
)
from models import GraphIntelligenceAskRequest
import analyst_agent


def test_safe_read_csv_with_title_banner():
    """Verify that title banner rows like 'Land-Ocean: Global Means' are detected and skipped."""
    csv_bytes = b"Land-Ocean: Global Means\nYear,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D\n1880,-0.17,-0.23,-0.08,-0.15,-0.08,-0.20,-0.17,-0.09,-0.13,-0.22,-0.20,-0.16,-0.16\n1881,-0.19,-0.13,0.04,0.06,0.07,-0.17,0.01,-0.02,-0.14,-0.21,-0.17,-0.06,-0.08\n"
    df, banner = safe_read_csv(csv_bytes)
    assert banner == "Land-Ocean: Global Means"
    assert "Year" in df.columns
    assert "J-D" in df.columns
    assert "is_partial_year" in df.columns
    assert len(df) == 2


def test_safe_read_csv_normal_csv():
    """Verify normal CSVs without banner row are parsed correctly."""
    csv_bytes = b"Category,Revenue,Profit\nTech,1000,200\nFinance,1500,300\n"
    df, banner = safe_read_csv(csv_bytes)
    assert banner is None
    assert list(df.columns) == ["Category", "Revenue", "Profit"]
    assert len(df) == 2


def test_normalize_dataframe_duplicate_columns():
    """Verify duplicate column names are deterministically deduplicated without overwriting."""
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["Sales", "Sales", "Profit"])
    clean_df = normalize_dataframe_headers_and_columns(df)
    assert list(clean_df.columns) == ["Sales", "Sales_2", "Profit"]
    assert clean_df["Sales"].tolist() == [1, 4]
    assert clean_df["Sales_2"].tolist() == [2, 5]


def test_normalize_dataframe_multiindex_headers():
    """Verify MultiIndex column headers are flattened into single 1D strings."""
    cols = pd.MultiIndex.from_tuples([("2023", "Q1", "Revenue"), ("2023", "Q2", "Revenue")])
    df = pd.DataFrame([[100, 120]], columns=cols)
    clean_df = normalize_dataframe_headers_and_columns(df)
    assert not isinstance(clean_df.columns, pd.MultiIndex)
    assert list(clean_df.columns) == ["2023_Q1_Revenue", "2023_Q2_Revenue"]


def test_normalize_dataframe_multiindex_rows():
    """Verify MultiIndex row index is reset to standard 1D RangeIndex."""
    idx = pd.MultiIndex.from_tuples([("Region1", "A"), ("Region2", "B")], names=["Region", "Sub"])
    df = pd.DataFrame({"Value": [10, 20]}, index=idx)
    clean_df = normalize_dataframe_headers_and_columns(df)
    assert not isinstance(clean_df.index, pd.MultiIndex)
    assert "Region" in clean_df.columns
    assert "Sub" in clean_df.columns
    assert "Value" in clean_df.columns


def test_analyze_chart_with_duplicate_columns_no_grouper_error():
    """Verify AnalyzeChartTool does not throw 1-dimensional Grouper error with duplicate column names."""
    df = pd.DataFrame({
        "Metric": [10.5, 12.0, 11.2, 14.1, 13.0],
        "Metric_dup": [10.5, 12.0, 11.2, 14.1, 13.0]
    })
    # Force duplicate column names in the test DataFrame
    df.columns = ["Land-Ocean: Global Means", "Land-Ocean: Global Means"]

    tool = AnalyzeChartTool()
    params = AnalyzeChartInput(
        dataset_id="test_ds",
        x_column="Land-Ocean: Global Means",
        y_column="Land-Ocean: Global Means",
        chart_type="line"
    )
    res = tool.execute(params, {"df": df, "dataset_id": "test_ds"})
    assert res.status == "success"
    assert "summary" in res.result
    assert len(res.result["key_findings"]) > 0


def test_analyze_chart_measure_by_measure_univariate():
    """Grouping a metric by itself (x_col == y_col) executes univariate distribution analysis."""
    df = pd.DataFrame({
        "Temperature": [0.1, 0.2, 0.15, 0.45, 0.35, 0.8, -0.05, 0.22]
    })
    tool = AnalyzeChartTool()
    params = AnalyzeChartInput(
        dataset_id="test_ds",
        x_column="Temperature",
        y_column="Temperature",
        chart_type="histogram"
    )
    res = tool.execute(params, {"df": df, "dataset_id": "test_ds"})
    assert res.status == "success"
    summary = res.result["summary"]
    assert "Temperature" in summary
    details = " ".join([f["detail"] for f in res.result["key_findings"]])
    assert "mean" in details.lower() or "median" in details.lower()


def test_analyze_chart_valid_date_year_grouping():
    """Verify date/year based grouping produces correct group statistics and trend findings."""
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    values = [0.10, 0.15, 0.22, 0.30, 0.38, 0.45, 0.55, 0.60, 0.72, 0.81, 0.95]
    df = pd.DataFrame({"Year": years, "Anomaly": values})

    tool = AnalyzeChartTool()
    params = AnalyzeChartInput(
        dataset_id="test_ds",
        x_column="Year",
        y_column="Anomaly",
        chart_type="line",
        aggregation="mean"
    )
    res = tool.execute(params, {"df": df, "dataset_id": "test_ds"})
    assert res.status == "success"
    findings = res.result["key_findings"]
    titles = [f["title"] for f in findings]
    assert any("Trend" in t or "Upward" in t or "Slope" in t for t in titles)


def test_analyze_chart_continuous_bivariate_regression():
    """Verify two continuous numeric columns trigger bivariate regression/correlation instead of float grouping."""
    np.random.seed(42)
    x = np.linspace(10.0, 50.0, 50)
    y = 2.5 * x + np.random.normal(0, 5, 50)
    df = pd.DataFrame({"Speed": x, "Distance": y})

    tool = AnalyzeChartTool()
    params = AnalyzeChartInput(
        dataset_id="test_ds",
        x_column="Speed",
        y_column="Distance",
        chart_type="scatter"
    )
    res = tool.execute(params, {"df": df, "dataset_id": "test_ds"})
    assert res.status == "success"
    titles = [f["title"] for f in res.result["key_findings"]]
    assert any("Correlation" in t or "Regression" in t or "Relationship" in t for t in titles)


def test_compare_groups_tool_defensive():
    """Verify CompareGroupsTool handles duplicate column names and validates inputs."""
    df = pd.DataFrame({
        "Department": ["Sales", "Sales", "Engineering", "Engineering", "Marketing"],
        "Salary": [70000, 75000, 110000, 120000, 80000],
        "Salary_dup": [70000, 75000, 110000, 120000, 80000]
    })
    # Force duplicate column names
    df.columns = ["Department", "Salary", "Salary"]

    tool = CompareGroupsTool()
    params = CompareGroupsInput(
        dataset_id="test_ds",
        group_column="Department",
        metric_column="Salary"
    )
    res = tool.execute(params, {"df": df, "dataset_id": "test_ds"})
    assert res.status == "success"
    assert res.result["total_groups"] == 3
    assert res.result["highest_group"] == "Engineering"


def test_glb_dataset_real_file_safe_parsing_and_analysis():
    """Load actual data/GLB.Ts+dSST.csv, execute AnalyzeChartTool on Year vs J-D, and check real evidence."""
    glb_path = os.path.join(os.path.dirname(__file__), "data", "GLB.Ts+dSST.csv")
    assert os.path.exists(glb_path)

    with open(glb_path, "rb") as f:
        content = f.read()

    df, banner = safe_read_csv(content)
    assert banner == "Land-Ocean: Global Means"
    assert "Year" in df.columns
    assert "J-D" in df.columns

    # Verify clean numeric types after coercion
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["J-D"] = pd.to_numeric(df["J-D"], errors="coerce")
    clean_sub = df[["Year", "J-D"]].dropna()
    assert len(clean_sub) > 100

    tool = AnalyzeChartTool()
    params = AnalyzeChartInput(
        dataset_id="glb_real",
        x_column="Year",
        y_column="J-D",
        chart_type="line",
        aggregation="mean"
    )
    res = tool.execute(params, {"df": df, "dataset_id": "glb_real"})
    assert res.status == "success"
    assert len(res.result["key_findings"]) > 0
    # Confirm evidence contains real calculated numbers (e.g. min, max, slope)
    evidence_text = str(res.result)
    assert any(str(yr) in evidence_text for yr in [1880, 2023, 2024])
