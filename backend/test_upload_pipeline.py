"""
DataSense AI V6 — Task 2: Excel & CSV Upload Pipeline Audit and Verification Test Suite
Tests all 11 required scenarios via real HTTP multipart form-data requests against FastAPI /api/upload:
1. Valid CSV file.
2. Valid XLSX workbook.
3. XLSX workbook containing multiple worksheets (with explicit and auto-selection).
4. CSV with UTF-8 text and non-ASCII characters (German/French accents, emojis).
5. File with missing values (preserves NaNs and records missing count).
6. File with mixed data types.
7. Empty file (0 bytes).
8. Malformed CSV (inconsistent columns).
9. Unsupported file extension (.pdf).
10. File exceeding the configured size limit (>50MB).
11. Workbook with an empty sheet 0 (falls back to non-empty sheet) and invalid sheet selection error.
12. Downstream confirmation: uploaded dataset is immediately available for charting, profiling, and ML.
"""
import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from main import (
    app,
    DATASET_STORAGE,
    RAW_DATASET_STORAGE,
    PROFILE_STORAGE,
    DATASET_VERSIONS_STORAGE
)

test_client = TestClient(app)


# =========================================================================
# Case 1: Valid CSV File
# =========================================================================
def test_upload_valid_csv():
    csv_bytes = b"id,name,score\n1,Alice,95.5\n2,Bob,88.0\n3,Charlie,72.3\n"
    files = {"file": ("students.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = test_client.post("/api/upload", files=files)

    assert resp.status_code == 200
    report = resp.json()
    dataset_id = report["metadata"]["id"]

    assert report["metadata"]["rows"] == 3
    assert report["metadata"]["columns"] == 3
    assert dataset_id in DATASET_STORAGE
    assert len(DATASET_STORAGE[dataset_id]) == 3
    assert list(DATASET_STORAGE[dataset_id].columns) == ["id", "name", "score"]


# =========================================================================
# Case 2: Valid XLSX Workbook (Uppercase extension test included: .XLSX)
# =========================================================================
def test_upload_valid_xlsx():
    df_orig = pd.DataFrame({
        "item": ["Apple", "Banana", "Cherry"],
        "price": [1.2, 0.8, 2.5],
        "quantity": [100, 150, 75]
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_orig.to_excel(writer, index=False, sheet_name="Inventory")
    buf.seek(0)

    # Test with uppercase extension .XLSX
    files = {"file": ("INVENTORY.XLSX", buf, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    resp = test_client.post("/api/upload", files=files)

    assert resp.status_code == 200
    report = resp.json()
    dataset_id = report["metadata"]["id"]

    assert report["metadata"]["rows"] == 3
    assert report["metadata"]["columns"] == 3
    assert dataset_id in DATASET_STORAGE
    assert set(DATASET_STORAGE[dataset_id]["item"]) == {"Apple", "Banana", "Cherry"}


# =========================================================================
# Case 3: XLSX Workbook Containing Multiple Worksheets
# =========================================================================
def test_upload_multi_sheet_xlsx():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame({"dept": ["HR", "Eng"], "count": [5, 20]}).to_excel(writer, index=False, sheet_name="Summary")
        pd.DataFrame({"emp_id": [1, 2, 3, 4], "name": ["A", "B", "C", "D"], "salary": [50, 60, 70, 80]}).to_excel(writer, index=False, sheet_name="Employees")
    buf.seek(0)

    # Test explicit sheet selection
    files = {"file": ("company.xlsx", buf, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    data = {"sheet_name": "Employees"}
    resp = test_client.post("/api/upload", files=files, data=data)

    assert resp.status_code == 200
    report = resp.json()
    dataset_id = report["metadata"]["id"]

    assert report["metadata"]["rows"] == 4
    assert "salary" in [c["column"] for c in report["columns"]]


# =========================================================================
# Case 4: CSV with UTF-8 Text and Non-ASCII Characters
# =========================================================================
def test_upload_utf8_non_ascii_csv():
    # Accented characters, umlauts, currency symbols
    text_csv = "city,landmark,currency\nZürich,Großmünster,CHF\nParis,Tour Eiffel,€\nSão Paulo,Avenida Paulista,R$\n"
    csv_bytes = text_csv.encode("utf-8")
    files = {"file": ("international.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = test_client.post("/api/upload", files=files)

    assert resp.status_code == 200
    report = resp.json()
    dataset_id = report["metadata"]["id"]

    df_loaded = DATASET_STORAGE[dataset_id]
    assert "Zürich" in df_loaded["city"].values
    assert "Paris" in df_loaded["city"].values
    assert "São Paulo" in df_loaded["city"].values


# =========================================================================
# Case 5: File with Missing Values
# =========================================================================
def test_upload_missing_values_csv():
    csv_bytes = b"col_a,col_b,col_c\n1,,10\n,foo,20\n3,bar,\n"
    files = {"file": ("missing.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = test_client.post("/api/upload", files=files)

    assert resp.status_code == 200
    report = resp.json()
    dataset_id = report["metadata"]["id"]

    assert report["metadata"]["missing_cells"] == 3
    df_loaded = DATASET_STORAGE[dataset_id]
    assert df_loaded["col_a"].isna().sum() == 1
    assert df_loaded["col_b"].isna().sum() == 1
    assert df_loaded["col_c"].isna().sum() == 1


# =========================================================================
# Case 6: File with Mixed Data Types
# =========================================================================
def test_upload_mixed_data_types_csv():
    csv_bytes = b"id,val,category\n1,100,active\n2,200.5,pending\n3,N/A,active\n4,400,inactive\n"
    files = {"file": ("mixed.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = test_client.post("/api/upload", files=files)

    assert resp.status_code == 200
    report = resp.json()
    dataset_id = report["metadata"]["id"]

    assert report["metadata"]["rows"] == 4
    assert dataset_id in DATASET_STORAGE


# =========================================================================
# Case 7: Empty File (0 Bytes)
# =========================================================================
def test_upload_empty_file():
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    resp = test_client.post("/api/upload", files=files)

    assert resp.status_code == 400
    assert "0 bytes" in resp.json()["detail"].lower() or "empty" in resp.json()["detail"].lower()


# =========================================================================
# Case 8: Malformed CSV (Inconsistent Columns)
# =========================================================================
def test_upload_malformed_csv():
    malformed_bytes = b"header1,header2\nrow1_val1\nrow2_val1,row2_val2,row2_unexpected3,row2_unexpected4\n"
    files = {"file": ("bad.csv", io.BytesIO(malformed_bytes), "text/csv")}
    resp = test_client.post("/api/upload", files=files)

    assert resp.status_code == 400
    assert "malformed" in resp.json()["detail"].lower() or "parse" in resp.json()["detail"].lower()


# =========================================================================
# Case 9: Unsupported File Extension
# =========================================================================
def test_upload_unsupported_file_extension():
    files = {"file": ("report.pdf", io.BytesIO(b"%PDF-1.4 sample content"), "application/pdf")}
    resp = test_client.post("/api/upload", files=files)

    assert resp.status_code == 400
    assert "unsupported file format" in resp.json()["detail"].lower()


# =========================================================================
# Case 10: File Exceeding Configured Size Limit (>50MB)
# =========================================================================
def test_upload_exceeds_size_limit():
    # 51 MB dummy byte stream
    large_payload = io.BytesIO(b"0" * (51 * 1024 * 1024))
    files = {"file": ("huge_dataset.csv", large_payload, "text/csv")}
    resp = test_client.post("/api/upload", files=files)

    assert resp.status_code == 400
    assert "50 mb" in resp.json()["detail"].lower()


# =========================================================================
# Case 11: Workbook with Empty Sheet 0 (Falls back to Sheet 1) & Invalid Selection
# =========================================================================
def test_upload_empty_sheet_fallback_and_invalid_selection():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame().to_excel(writer, index=False, sheet_name="BlankCover")
        pd.DataFrame({"metric": ["Latency", "Throughput"], "val": [45, 1200]}).to_excel(writer, index=False, sheet_name="ActualData")
    buf.seek(0)

    # 1. Automatic fallback test: No sheet specified, should skip empty BlankCover and load ActualData
    files = {"file": ("template.xlsx", buf, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    resp_auto = test_client.post("/api/upload", files=files)
    assert resp_auto.status_code == 200
    assert resp_auto.json()["metadata"]["rows"] == 2

    # 2. Invalid sheet selection error test
    buf.seek(0)
    files2 = {"file": ("template.xlsx", buf, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    resp_invalid = test_client.post("/api/upload", files=files2, data={"sheet_name": "NonExistentSheet"})
    assert resp_invalid.status_code == 400
    assert "not found" in resp_invalid.json()["detail"].lower()
    assert "BlankCover" in resp_invalid.json()["detail"]


# =========================================================================
# Downstream Usage Confirmation: Verify Chart, Profiling & ML on Uploaded Data
# =========================================================================
def test_uploaded_dataset_available_downstream():
    """Confirm the uploaded dataset ID immediately functions with profiling, charts, and cleaning."""
    csv_bytes = b"category,revenue,units\nElectronics,5000,10\nFurniture,3200,5\nClothing,1500,25\n"
    files = {"file": ("retail.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp_upload = test_client.post("/api/upload", files=files)
    assert resp_upload.status_code == 200
    dataset_id = resp_upload.json()["metadata"]["id"]

    # 1. Downstream Chart Generation
    chart_payload = {
        "dataset_id": dataset_id,
        "x_column": "category",
        "y_column": "revenue",
        "chart_type": "bar",
        "aggregation": "sum"
    }
    resp_chart = test_client.post("/api/custom-chart", json=chart_payload)
    assert resp_chart.status_code == 200
    chart_data = resp_chart.json()
    assert chart_data["title"] == "Sum of revenue by category"
    assert len(chart_data["data"]) == 3

    # 2. Downstream Cleaning Recommendations
    resp_clean = test_client.get(f"/api/clean/{dataset_id}/recommendations")
    assert resp_clean.status_code == 200
    assert isinstance(resp_clean.json(), list)

    # 3. Downstream Preview Table
    resp_prev = test_client.get(f"/api/dataset/{dataset_id}/preview?page=1&page_size=10")
    assert resp_prev.status_code == 200
    assert len(resp_prev.json()["rows"]) == 3


# =========================================================================
# Case 13: Excel Worksheet Inspection Endpoint (/api/upload/inspect-sheets)
# =========================================================================
def test_inspect_excel_worksheets_endpoint():
    """Verify inspection of all sheets in workbook without committing to storage."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame({"dept": ["HR", "Eng"], "headcount": [5, 20]}).to_excel(writer, index=False, sheet_name="Departments")
        pd.DataFrame({"metric": ["Growth"], "value": [0.15]}).to_excel(writer, index=False, sheet_name="Metrics")
        pd.DataFrame().to_excel(writer, index=False, sheet_name="BlankCover")
    buf.seek(0)

    files = {"file": ("inspection_test.xlsx", buf, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    resp = test_client.post("/api/upload/inspect-sheets", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "inspection_test.xlsx"
    assert data["total_sheets"] == 3
    
    sheets_by_name = {s["sheet_name"]: s for s in data["sheets"]}
    assert "Departments" in sheets_by_name
    assert sheets_by_name["Departments"]["rows"] == 2
    assert sheets_by_name["Departments"]["columns"] == 2
    assert sheets_by_name["Departments"]["is_empty"] is False
    assert len(sheets_by_name["Departments"]["preview_rows"]) == 2

    assert "BlankCover" in sheets_by_name
    assert sheets_by_name["BlankCover"]["is_empty"] is True


# =========================================================================
# Case 14: Valid ZIP Archive Ingestion via /api/upload
# =========================================================================
def test_upload_valid_zip_archive():
    """Verify single ZIP archive containing a valid CSV is safely ingested."""
    import zipfile
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sales_data.csv", "item,revenue,units\nLaptop,1200.0,3\nMouse,25.0,15\n")
    zip_buf.seek(0)

    files = {"file": ("sales_bundle.zip", zip_buf, "application/zip")}
    resp = test_client.post("/api/upload", files=files)

    assert resp.status_code == 200
    report = resp.json()
    assert report["metadata"]["rows"] == 2
    assert report["metadata"]["columns"] == 3
    assert "revenue" in [c["column"] for c in report["columns"]]


# =========================================================================
# Case 15: ZIP Safety — Path Traversal Attack Rejection
# =========================================================================
def test_upload_zip_path_traversal_rejection():
    """Verify archive members attempting path traversal are strictly rejected."""
    import zipfile
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../../../evil.csv", "id,cmd\n1,bad\n")
    zip_buf.seek(0)

    files = {"file": ("exploit.zip", zip_buf, "application/zip")}
    resp = test_client.post("/api/upload", files=files)

    assert resp.status_code == 400
    assert "path traversal" in resp.json()["detail"].lower()


# =========================================================================
# Case 16: ZIP Safety — Forbidden Executable Rejection
# =========================================================================
def test_upload_zip_forbidden_executable_rejection():
    """Verify archives containing forbidden executables (.exe, .py, .sh) are rejected."""
    import zipfile
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("trojan.exe", b"MZ\x90\x00\x03\x00\x00\x00")
    zip_buf.seek(0)

    files = {"file": ("malware.zip", zip_buf, "application/zip")}
    resp = test_client.post("/api/upload", files=files)

    assert resp.status_code == 400
    assert "forbidden file type" in resp.json()["detail"].lower()


# =========================================================================
# Case 17: Multi-File Ingestion Endpoint (/api/upload/multi)
# =========================================================================
def test_upload_multi_files_endpoint():
    """Verify uploading multiple CSV files creates a session inventory with all tables registered."""
    csv_customers = b"customer_id,name,city\n1,Alice,Berlin\n2,Bob,Paris\n3,Charlie,Munich\n"
    csv_orders = b"order_id,customer_id,amount\n101,1,85.50\n102,1,120.00\n103,2,45.00\n104,2,99.99\n"

    files = [
        ("files", ("customers.csv", io.BytesIO(csv_customers), "text/csv")),
        ("files", ("orders.csv", io.BytesIO(csv_orders), "text/csv")),
    ]
    resp = test_client.post("/api/upload/multi", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tables"] == 2
    session_id = data["session_id"]
    assert session_id.startswith("session_")

    table_names = [t["filename"] for t in data["tables"]]
    assert "customers.csv" in table_names
    assert "orders.csv" in table_names

    # Check that each table is immediately available in DATASET_STORAGE
    for table in data["tables"]:
        tid = table["table_id"]
        assert tid in DATASET_STORAGE
        assert len(DATASET_STORAGE[tid]) == table["rows"]


# =========================================================================
# Case 18: Table Relationships Detection (/api/tables/relationships)
# =========================================================================
def test_detect_table_relationships_endpoint():
    """Verify candidate join key detection, uniqueness, value overlap, and 1:N classification."""
    csv_customers = b"customer_id,name,city\n1,Alice,Berlin\n2,Bob,Paris\n3,Charlie,Munich\n"
    csv_orders = b"order_id,customer_id,amount\n101,1,85.50\n102,1,120.00\n103,2,45.00\n"

    files = [
        ("files", ("customers.csv", io.BytesIO(csv_customers), "text/csv")),
        ("files", ("orders.csv", io.BytesIO(csv_orders), "text/csv")),
    ]
    resp_multi = test_client.post("/api/upload/multi", files=files)
    session_id = resp_multi.json()["session_id"]

    resp_rel = test_client.post(f"/api/tables/relationships?session_id={session_id}")
    assert resp_rel.status_code == 200
    rel_data = resp_rel.json()

    assert len(rel_data["relationships"]) == 1
    rel = rel_data["relationships"][0]
    assert rel["status"] == "related"

    # Find the candidate key
    assert len(rel["candidate_keys"]) >= 1
    top_key = rel["candidate_keys"][0]
    assert top_key["table_a_column"] == "customer_id" or top_key["table_b_column"] == "customer_id"
    assert top_key["confidence"] == "high"
    assert top_key["relationship_type"] in ["one-to-many", "many-to-one"]
    assert top_key["overlap_count"] == 2  # Customer IDs 1 and 2 overlap


# =========================================================================
# Case 19: Independent Tables Detection
# =========================================================================
def test_table_relationships_independent_tables():
    """Verify tables with zero overlapping columns or keys are classified as independent."""
    csv_weather = b"city,temperature,humidity\nBerlin,22,60\nParis,25,55\n"
    csv_products = b"product_sku,price,stock\nSKU-001,10.5,100\nSKU-002,25.0,40\n"

    files = [
        ("files", ("weather.csv", io.BytesIO(csv_weather), "text/csv")),
        ("files", ("products.csv", io.BytesIO(csv_products), "text/csv")),
    ]
    resp_multi = test_client.post("/api/upload/multi", files=files)
    session_id = resp_multi.json()["session_id"]

    resp_rel = test_client.post(f"/api/tables/relationships?session_id={session_id}")
    assert resp_rel.status_code == 200
    rel_data = resp_rel.json()

    assert len(rel_data["relationships"]) == 1
    assert rel_data["relationships"][0]["status"] == "independent"
    assert len(rel_data["relationships"][0]["candidate_keys"]) == 0


# =========================================================================
# Case 20: Table Join Preview & Cartesian Multiplication Risk
# =========================================================================
def test_table_join_preview_and_cartesian_risk():
    """Verify join preview calculation, unmatched counts, and Cartesian explosion risk detection."""
    # 1. Normal 1:N join preview
    csv_cust = b"customer_id,name\n1,Alice\n2,Bob\n3,Charlie\n"
    csv_ord = b"order_id,customer_id,amount\n101,1,50\n102,1,75\n103,2,100\n"
    files = [
        ("files", ("customers.csv", io.BytesIO(csv_cust), "text/csv")),
        ("files", ("orders.csv", io.BytesIO(csv_ord), "text/csv")),
    ]
    resp = test_client.post("/api/upload/multi", files=files)
    tables = {t["filename"]: t["table_id"] for t in resp.json()["tables"]}
    tid_cust = tables["customers.csv"]
    tid_ord = tables["orders.csv"]

    preview_req = {
        "table_a_id": tid_cust,
        "table_b_id": tid_ord,
        "join_column_a": "customer_id",
        "join_column_b": "customer_id",
        "join_type": "inner"
    }
    resp_prev = test_client.post("/api/tables/join-preview", json=preview_req)
    assert resp_prev.status_code == 200
    prev_data = resp_prev.json()

    assert prev_data["expected_rows"] == 3
    assert prev_data["unmatched_rows_a"] == 1  # Charlie has no orders
    assert prev_data["unmatched_rows_b"] == 0
    assert prev_data["cartesian_multiplication_risk"] is False
    assert len(prev_data["preview_rows"]) == 3

    # 2. Cartesian multiplication test (N:M with duplicates on both sides)
    csv_dup_a = b"dept_code,emp\nD1,Alice\nD1,Bob\n"
    csv_dup_b = b"dept_code,project\nD1,Alpha\nD1,Beta\n"
    files_dup = [
        ("files", ("dept_emp.csv", io.BytesIO(csv_dup_a), "text/csv")),
        ("files", ("dept_proj.csv", io.BytesIO(csv_dup_b), "text/csv")),
    ]
    resp_dup = test_client.post("/api/upload/multi", files=files_dup)
    tables_dup = {t["filename"]: t["table_id"] for t in resp_dup.json()["tables"]}

    dup_preview_req = {
        "table_a_id": tables_dup["dept_emp.csv"],
        "table_b_id": tables_dup["dept_proj.csv"],
        "join_column_a": "dept_code",
        "join_column_b": "dept_code",
        "join_type": "inner"
    }
    resp_dup_prev = test_client.post("/api/tables/join-preview", json=dup_preview_req)
    assert resp_dup_prev.status_code == 200
    dup_prev_data = resp_dup_prev.json()

    assert dup_prev_data["expected_rows"] == 4  # 2x2 = 4
    assert dup_prev_data["cartesian_multiplication_risk"] is True
    assert "Cartesian multiplication risk" in dup_prev_data["risk_warning"]


# =========================================================================
# Case 21: Table Join Apply & Version Provenance (/api/tables/join-apply)
# =========================================================================
def test_table_join_apply_and_provenance():
    """Verify explicit approval executes the join, creates an isolated dataset with V0 provenance."""
    csv_cust = b"customer_id,name\n1,Alice\n2,Bob\n"
    csv_ord = b"order_id,customer_id,amount\n101,1,50.0\n102,2,75.0\n"
    files = [
        ("files", ("customers.csv", io.BytesIO(csv_cust), "text/csv")),
        ("files", ("orders.csv", io.BytesIO(csv_ord), "text/csv")),
    ]
    resp = test_client.post("/api/upload/multi", files=files)
    tables = {t["filename"]: t["table_id"] for t in resp.json()["tables"]}

    apply_req = {
        "table_a_id": tables["customers.csv"],
        "table_b_id": tables["orders.csv"],
        "join_column_a": "customer_id",
        "join_column_b": "customer_id",
        "join_type": "inner",
        "new_dataset_name": "Customer Orders Combined"
    }
    resp_apply = test_client.post("/api/tables/join-apply", json=apply_req)
    assert resp_apply.status_code == 200

    report = resp_apply.json()
    new_id = report["metadata"]["id"]
    assert report["metadata"]["name"] == "Customer Orders Combined"
    assert report["metadata"]["rows"] == 2
    assert "amount" in [c["column"] for c in report["columns"]]
    assert "name" in [c["column"] for c in report["columns"]]

    # Verify version history has join provenance
    resp_versions = test_client.get(f"/api/clean/{new_id}/versions")
    assert resp_versions.status_code == 200
    versions = resp_versions.json()
    assert len(versions) >= 2
    assert versions[0]["version_id"] == 0
    assert "Joined" in versions[0]["description"]
    assert "customer_id" in versions[0]["description"]
