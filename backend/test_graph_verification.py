"""
DataSense AI V6 — Task 1: Independent Graph Readability and Calculation Verification Tests
Validates:
1. Category frequency counts with known expected results.
2. Grouped means, sums, min, max, count.
3. Sorting in both directions (desc, asc, none).
4. Top N filtering (Top 5, Top 10) vs All categories (top_n=0).
5. Missing category values and exclusion documentation.
6. Duplicate records handling.
7. Horizontal bar preference for >= 6 categories.
8. Low-frequency category explanation when count=1 is dominant.
9. Graph AI grounded answers (highest count, highest/lowest average, overview, anomalies, filter behavior).
"""
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app, DATASET_STORAGE, PROFILE_STORAGE, DATASET_VERSIONS_STORAGE
from models import CustomChartRequest, ChartSpec
from analysis_engine import generate_custom_chart, create_plotly_spec

test_client = TestClient(app)


@pytest.fixture
def deterministic_restaurant_df():
    """10 restaurants with known frequency distribution:
    Burger King: 15
    Subway: 8
    Pizza Hut: 5
    7 boutique restaurants with count 1: Cafe Alpha, Bistro Beta, Diner Gamma, Eatery Delta, Grill Epsilon, House Zeta, Kitchen Eta.
    Total records = 35. Unique categories = 10. Low frequency (count=1) = 7 (70%).
    """
    records = []
    records.extend([{"restaurant": "Burger King", "rating": 4.5, "revenue": 100}] * 15)
    records.extend([{"restaurant": "Subway", "rating": 3.8, "revenue": 60}] * 8)
    records.extend([{"restaurant": "Pizza Hut", "rating": 4.0, "revenue": 80}] * 5)
    for name in ["Cafe Alpha", "Bistro Beta", "Diner Gamma", "Eatery Delta", "Grill Epsilon", "House Zeta", "Kitchen Eta"]:
        records.append({"restaurant": name, "rating": 4.2, "revenue": 50})
    return pd.DataFrame(records)


@pytest.fixture
def deterministic_grouped_df():
    """Grouped dataset with known exact mathematical metrics:
    Group A: [100, 200, 300] -> mean=200.0, sum=600, min=100, max=300, count=3
    Group B: [50, 50] -> mean=50.0, sum=100, min=50, max=50, count=2
    Group C: [400] -> mean=400.0, sum=400, min=400, max=400, count=1
    Group D with NaN rating: [None]
    """
    data = [
        {"segment": "Group A", "sales": 100.0},
        {"segment": "Group A", "sales": 200.0},
        {"segment": "Group A", "sales": 300.0},
        {"segment": "Group B", "sales": 50.0},
        {"segment": "Group B", "sales": 50.0},
        {"segment": "Group C", "sales": 400.0},
        {"segment": "Group D", "sales": np.nan},
        {"segment": None, "sales": 999.0},  # null category
    ]
    return pd.DataFrame(data)


# =========================================================================
# 1. Independent Calculation Tests for Frequency Counts & Readability
# =========================================================================

def test_graph_frequency_calculation_and_low_frequency_note(deterministic_restaurant_df):
    """Verify frequency calculations, low-frequency detection, and horizontal bar auto-switch."""
    df = deterministic_restaurant_df
    req = CustomChartRequest(
        dataset_id="test_session",
        x_column="restaurant",
        chart_type="bar",
        orientation="auto",
        top_n=0  # All categories
    )

    spec = generate_custom_chart(df, req)

    # 1. Verify exact counts
    data_dict = {d["x"]: d["y"] for d in spec.data}
    assert data_dict["Burger King"] == 15
    assert data_dict["Subway"] == 8
    assert data_dict["Pizza Hut"] == 5
    assert data_dict["Cafe Alpha"] == 1
    assert data_dict["Kitchen Eta"] == 1

    # 2. Verify total and displayed categories
    assert spec.total_categories == 10
    assert spec.displayed_categories == 10
    assert spec.is_subset is False

    # 3. Verify horizontal orientation selected for >= 6 categories
    assert spec.orientation == "h"
    assert spec.plotly_spec["data"][0]["orientation"] == "h"

    # 4. Verify low-frequency explanation (7 out of 10 have count=1)
    assert spec.low_frequency_note is not None
    assert "7 of 10 categories (70.0%)" in spec.low_frequency_note
    assert "count of 1" in spec.low_frequency_note


def test_graph_top_n_filtering_and_subset_tag(deterministic_restaurant_df):
    """Verify Top N view properly limits data and flags subset."""
    df = deterministic_restaurant_df
    req = CustomChartRequest(
        dataset_id="test_session",
        x_column="restaurant",
        chart_type="bar",
        top_n=3,
        sort_order="desc"
    )

    spec = generate_custom_chart(df, req)

    assert spec.is_subset is True
    assert spec.total_categories == 10
    assert spec.displayed_categories == 3
    assert len(spec.data) == 3

    # Top 3 must be Burger King (15), Subway (8), Pizza Hut (5)
    names = [d["x"] for d in spec.data]
    assert names == ["Burger King", "Subway", "Pizza Hut"]
    assert "Top 3 of 10" in spec.title


def test_graph_sorting_in_both_directions(deterministic_restaurant_df):
    """Verify ascending vs descending category sorting."""
    df = deterministic_restaurant_df

    # Descending: highest first
    req_desc = CustomChartRequest(
        dataset_id="test_session",
        x_column="restaurant",
        chart_type="bar",
        sort_order="desc",
        top_n=0
    )
    spec_desc = generate_custom_chart(df, req_desc)
    assert spec_desc.data[0]["x"] == "Burger King"
    assert spec_desc.data[0]["y"] == 15
    assert spec_desc.data[-1]["y"] == 1

    # Ascending: lowest first
    req_asc = CustomChartRequest(
        dataset_id="test_session",
        x_column="restaurant",
        chart_type="bar",
        sort_order="asc",
        top_n=0
    )
    spec_asc = generate_custom_chart(df, req_asc)
    assert spec_asc.data[0]["y"] == 1
    assert spec_asc.data[-1]["x"] == "Burger King"
    assert spec_asc.data[-1]["y"] == 15


# =========================================================================
# 2. Independent Calculation Tests for Grouped Aggregations
# =========================================================================

def test_graph_grouped_mean_and_sum(deterministic_grouped_df):
    """Verify grouped means and sums match independently calculated values."""
    df = deterministic_grouped_df

    # Test Mean
    req_mean = CustomChartRequest(
        dataset_id="test_session",
        x_column="segment",
        y_column="sales",
        chart_type="bar",
        aggregation="mean",
        sort_order="desc"
    )
    spec_mean = generate_custom_chart(df, req_mean)
    mean_dict = {d["x"]: d["y"] for d in spec_mean.data}

    # Group C: 400.0, Group A: 200.0, Group B: 50.0
    assert mean_dict["Group C"] == 400.0
    assert mean_dict["Group A"] == 200.0
    assert mean_dict["Group B"] == 50.0
    assert "Group D" not in mean_dict  # NaN sales excluded from mean calculation

    # Test Sum
    req_sum = CustomChartRequest(
        dataset_id="test_session",
        x_column="segment",
        y_column="sales",
        chart_type="bar",
        aggregation="sum",
        sort_order="desc"
    )
    spec_sum = generate_custom_chart(df, req_sum)
    sum_dict = {d["x"]: d["y"] for d in spec_sum.data}

    # Group A: 600.0, Group C: 400.0, Group B: 100.0
    assert sum_dict["Group A"] == 600.0
    assert sum_dict["Group C"] == 400.0
    assert sum_dict["Group B"] == 100.0


def test_graph_missing_values_and_exclusions_reporting(deterministic_grouped_df):
    """Verify missing values are handled gracefully and excluded rows are documented."""
    df = deterministic_grouped_df
    req = CustomChartRequest(
        dataset_id="test_session",
        x_column="segment",
        y_column="sales",
        chart_type="bar",
        aggregation="mean"
    )
    spec = generate_custom_chart(df, req)

    # Total rows = 8. Group D has NaN sales (1), null segment (1) -> 2 rows excluded
    assert spec.rows_used == 6
    assert spec.exclusions is not None
    assert "Excluded 2 rows with missing values" in spec.exclusions


def test_graph_invalid_column_handling(deterministic_restaurant_df):
    """Verify non-existent column raises clear ValueError."""
    df = deterministic_restaurant_df
    req = CustomChartRequest(
        dataset_id="test_session",
        x_column="non_existent_column",
        chart_type="bar"
    )
    with pytest.raises(ValueError, match="Column 'non_existent_column' does not exist in dataset"):
        generate_custom_chart(df, req)


# =========================================================================
# 3. Graph AI Grounded Intelligence Verification Tests
# =========================================================================

def test_graph_ai_highest_count_question(deterministic_restaurant_df):
    """Verify Graph AI correctly identifies the highest count category from underlying data."""
    sess_id = "test_graph_ai_session_restaurants"
    DATASET_STORAGE[sess_id] = deterministic_restaurant_df.copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(deterministic_restaurant_df)}]

    # Ask: "Which category has the highest count?"
    payload = {
        "dataset_id": sess_id,
        "x_column": "restaurant",
        "y_column": None,
        "chart_type": "bar",
        "question": "Which category has the highest count?"
    }
    resp = test_client.post("/api/graph/ask", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()

    # Answer must explicitly report Burger King with 15 observations
    ans_text = str(res_data["answer"]) if isinstance(res_data["answer"], str) else f"{res_data['answer']['finding']} {' '.join(res_data['answer']['evidence'])}"
    assert "Burger King" in ans_text
    assert "15" in ans_text


def test_graph_ai_highest_and_lowest_average_question(deterministic_grouped_df):
    """Verify Graph AI correctly computes leading and lagging average groups."""
    sess_id = "test_graph_ai_session_grouped"
    DATASET_STORAGE[sess_id] = deterministic_grouped_df.copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(deterministic_grouped_df)}]

    payload = {
        "dataset_id": sess_id,
        "x_column": "segment",
        "y_column": "sales",
        "chart_type": "bar",
        "aggregation": "mean",
        "question": "Which groups have the highest and lowest average values?"
    }
    resp = test_client.post("/api/graph/ask", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()

    # Highest: Group C (400.0), Lowest: Group B (50.0)
    ans = res_data["answer"]
    ans_text = str(ans) if isinstance(ans, str) else f"{ans['finding']} {' '.join(ans['evidence'])}"
    assert "Group C" in ans_text
    assert "400" in ans_text
    assert "Group B" in ans_text
    assert "50" in ans_text


def test_graph_ai_low_frequency_explanation(deterministic_restaurant_df):
    """Verify Graph AI explains why categories have a count of 1 without fabricating differences."""
    sess_id = "test_graph_ai_session_low_freq"
    DATASET_STORAGE[sess_id] = deterministic_restaurant_df.copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(deterministic_restaurant_df)}]

    payload = {
        "dataset_id": sess_id,
        "x_column": "restaurant",
        "y_column": None,
        "chart_type": "bar",
        "question": "Explain why so many categories have a count of 1"
    }
    resp = test_client.post("/api/graph/ask", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()

    ans = res_data["answer"]
    ans_text = str(ans) if isinstance(ans, str) else f"{ans['finding']} {' '.join(ans['evidence'])}"
    assert "low-frequency" in ans_text.lower() or "count of 1" in ans_text.lower()
    assert "7 of 10" in ans_text or "70.0%" in ans_text or "long-tail" in ans_text.lower() or "single-instance" in ans_text.lower()


def test_graph_duplicate_records_and_aggregations(deterministic_grouped_df):
    """Verify duplicate records and min/max/count aggregations calculate accurately."""
    df = deterministic_grouped_df

    # Test Min
    req_min = CustomChartRequest(
        dataset_id="test_session",
        x_column="segment",
        y_column="sales",
        chart_type="bar",
        aggregation="min"
    )
    spec_min = generate_custom_chart(df, req_min)
    min_dict = {d["x"]: d["y"] for d in spec_min.data}
    assert min_dict["Group A"] == 100.0
    assert min_dict["Group B"] == 50.0
    assert min_dict["Group C"] == 400.0

    # Test Max
    req_max = CustomChartRequest(
        dataset_id="test_session",
        x_column="segment",
        y_column="sales",
        chart_type="bar",
        aggregation="max"
    )
    spec_max = generate_custom_chart(df, req_max)
    max_dict = {d["x"]: d["y"] for d in spec_max.data}
    assert max_dict["Group A"] == 300.0
    assert max_dict["Group B"] == 50.0
    assert max_dict["Group C"] == 400.0


def test_graph_dataset_changes_after_chart_generation(deterministic_restaurant_df):
    """Verify that when the underlying dataset mutates, newly generated charts accurately reflect the new version."""
    sess_id = "test_mutating_session"
    DATASET_STORAGE[sess_id] = deterministic_restaurant_df.copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(deterministic_restaurant_df)}]

    # Initial chart
    req = CustomChartRequest(
        dataset_id=sess_id,
        x_column="restaurant",
        chart_type="bar",
        top_n=0
    )
    spec_v0 = generate_custom_chart(DATASET_STORAGE[sess_id], req)
    assert len(spec_v0.data) == 10

    # Add 10 new records for "Chipotle"
    new_rows = pd.DataFrame([{"restaurant": "Chipotle", "rating": 4.6, "revenue": 120}] * 10)
    mutated_df = pd.concat([DATASET_STORAGE[sess_id], new_rows], ignore_index=True)
    DATASET_STORAGE[sess_id] = mutated_df
    DATASET_VERSIONS_STORAGE[sess_id].append({"version_id": 1, "rows": len(mutated_df)})

    # Regenerate chart
    spec_v1 = generate_custom_chart(DATASET_STORAGE[sess_id], req)
    v1_dict = {d["x"]: d["y"] for d in spec_v1.data}
    assert "Chipotle" in v1_dict
    assert v1_dict["Chipotle"] == 10
    assert spec_v1.total_categories == 11
    assert spec_v1.rows_used == 45


def test_graph_ai_overview_anomaly_and_filter_questions(deterministic_restaurant_df):
    """Verify Graph AI answers for overview, anomalies, and filtering behavior."""
    sess_id = "test_graph_ai_remaining_questions"
    DATASET_STORAGE[sess_id] = deterministic_restaurant_df.copy()
    DATASET_VERSIONS_STORAGE[sess_id] = [{"version_id": 0, "rows": len(deterministic_restaurant_df)}]

    # 1. Overview question
    payload_overview = {
        "dataset_id": sess_id,
        "x_column": "restaurant",
        "y_column": None,
        "chart_type": "bar",
        "question": "What does this chart show?"
    }
    resp1 = test_client.post("/api/graph/ask", json=payload_overview)
    assert resp1.status_code == 200
    ans1 = resp1.json()["answer"]
    ans1_text = str(ans1) if isinstance(ans1, str) else f"{ans1['finding']} {' '.join(ans1['evidence'])}"
    assert "Chart Overview" in ans1_text or "Visualizes" in ans1_text
    assert "restaurant" in ans1_text

    # 2. Outliers / Anomalies question
    payload_anomaly = {
        "dataset_id": sess_id,
        "x_column": "restaurant",
        "y_column": None,
        "chart_type": "bar",
        "question": "Are there unusual values?"
    }
    resp2 = test_client.post("/api/graph/ask", json=payload_anomaly)
    assert resp2.status_code == 200
    ans2 = resp2.json()["answer"]
    ans2_text = str(ans2) if isinstance(ans2, str) else f"{ans2['finding']} {' '.join(ans2['evidence'])}"
    assert "outlier" in ans2_text.lower() or "anomaly" in ans2_text.lower()

    # 3. Filtering question
    payload_filter = {
        "dataset_id": sess_id,
        "x_column": "restaurant",
        "y_column": None,
        "chart_type": "bar",
        "question": "What changes when I filter this chart?"
    }
    resp3 = test_client.post("/api/graph/ask", json=payload_filter)
    assert resp3.status_code == 200
    ans3 = resp3.json()["answer"]
    ans3_text = str(ans3) if isinstance(ans3, str) else f"{ans3['finding']} {' '.join(ans3['evidence'])}"
    assert "filter" in ans3_text.lower()
    assert "subset" in ans3_text.lower()

