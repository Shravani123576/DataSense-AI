"""
DataSense AI Tool Layer — Dataset Inspection and Profiling Tools (Phase 3)
Deterministic tools for schema inspection, quality profiling, and version history.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolCategory, ToolPermission, ToolExecutionResult, ToolExecutionStatus
from data_profiler import profile_dataframe, sanitize_for_json


class DatasetIdInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")


class ProfileDatasetTool(BaseTool):
    name = "profile_dataset"
    description = "Compute comprehensive dataset health, column statistics, distributions, and correlation patterns."
    category = ToolCategory.DATASET
    permission = ToolPermission.READ_ONLY
    input_schema = DatasetIdInput

    def execute(self, params: DatasetIdInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found in active session."
            )

        meta = context.get("metadata", {"name": "Dataset", "filename": "data.csv"})
        v_idx = context.get("dataset_version", 0)

        report = profile_dataframe(
            df=df,
            dataset_id=params.dataset_id,
            name=meta.get("name", "Dataset"),
            filename=meta.get("filename", "data.csv")
        )

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result=report.dict() if hasattr(report, "dict") else report.model_dump(),
            dataset_id=params.dataset_id,
            dataset_version=v_idx,
            warnings=[f"Dataset has {report.metadata.missing_cells} missing cells"] if report.metadata.missing_cells > 0 else [],
            mutated_data=False,
            requires_approval=False,
            provenance={"rows": len(df), "columns": len(df.columns)}
        )


class InspectDataQualityTool(BaseTool):
    name = "inspect_data_quality"
    description = "Audit missing data rates, duplicate rows, numerical outliers (1.5x IQR), and constant-variance columns."
    category = ToolCategory.DATASET
    permission = ToolPermission.READ_ONLY
    input_schema = DatasetIdInput

    def execute(self, params: DatasetIdInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found in active session."
            )

        v_idx = context.get("dataset_version", 0)
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols if total_rows > 0 and total_cols > 0 else 1

        missing_cells = int(df.isna().sum().sum())
        missing_pct = round((missing_cells / total_cells) * 100, 2)
        duplicate_rows = int(df.duplicated().sum())
        duplicate_pct = round((duplicate_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0

        column_issues: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for col in df.columns:
            col_missing = int(df[col].isna().sum())
            col_missing_pct = round((col_missing / total_rows) * 100, 2) if total_rows > 0 else 0.0
            unique_count = int(df[col].nunique(dropna=True))

            outlier_count = 0
            if pd.api.types.is_numeric_dtype(df[col]):
                valid_num = df[col].dropna()
                if len(valid_num) >= 4:
                    q25 = float(valid_num.quantile(0.25))
                    q75 = float(valid_num.quantile(0.75))
                    iqr = q75 - q25
                    lower = q25 - 1.5 * iqr
                    upper = q75 + 1.5 * iqr
                    outlier_count = int(((valid_num < lower) | (valid_num > upper)).sum())

            is_constant = (unique_count <= 1 and total_rows > 1)
            if is_constant:
                warnings.append(f"Column '{col}' has zero variance (constant value).")

            column_issues.append({
                "column": str(col),
                "data_type": str(df[col].dtype),
                "missing_count": col_missing,
                "missing_percentage": col_missing_pct,
                "unique_values": unique_count,
                "outlier_count": outlier_count,
                "is_constant": is_constant
            })

        quality_score = 100
        if missing_pct > 0:
            quality_score -= min(30, int(missing_pct * 1.5))
        if duplicate_pct > 0:
            quality_score -= min(25, int(duplicate_pct * 2.0))
        quality_score = max(10, quality_score)

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result={
                "overall_health_score": quality_score,
                "total_rows": total_rows,
                "total_columns": total_cols,
                "missing_cells": missing_cells,
                "missing_percentage": missing_pct,
                "duplicate_rows": duplicate_rows,
                "duplicate_percentage": duplicate_pct,
                "column_audits": column_issues
            },
            dataset_id=params.dataset_id,
            dataset_version=v_idx,
            warnings=warnings,
            mutated_data=False,
            requires_approval=False
        )


class GetDatasetSchemaTool(BaseTool):
    name = "get_dataset_schema"
    description = "Retrieve column names, inferred data types, null rates, and sample values."
    category = ToolCategory.DATASET
    permission = ToolPermission.READ_ONLY
    input_schema = DatasetIdInput

    def execute(self, params: DatasetIdInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found."
            )

        v_idx = context.get("dataset_version", 0)
        total_rows = len(df)
        columns_schema = []

        for col in df.columns:
            s = df[col]
            samples = [sanitize_for_json(v) for v in s.dropna().head(3).tolist()]
            columns_schema.append({
                "column": str(col),
                "inferred_type": "numeric" if pd.api.types.is_numeric_dtype(s) else ("datetime" if pd.api.types.is_datetime64_any_dtype(s) else "categorical"),
                "pandas_dtype": str(s.dtype),
                "null_count": int(s.isna().sum()),
                "null_percentage": round((s.isna().sum() / total_rows) * 100, 2) if total_rows > 0 else 0.0,
                "unique_count": int(s.nunique(dropna=True)),
                "sample_values": samples
            })

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result={
                "dataset_id": params.dataset_id,
                "dataset_version": v_idx,
                "rows": total_rows,
                "columns": columns_schema
            },
            dataset_id=params.dataset_id,
            dataset_version=v_idx,
            warnings=[],
            mutated_data=False,
            requires_approval=False
        )


class GetDatasetVersionHistoryTool(BaseTool):
    name = "get_dataset_version_history"
    description = "List committed immutable dataset versions, audit operations, and row/column impact."
    category = ToolCategory.DATASET
    permission = ToolPermission.READ_ONLY
    input_schema = DatasetIdInput

    def execute(self, params: DatasetIdInput, context: Dict[str, Any]) -> ToolExecutionResult:
        versions = context.get("versions_storage", {}).get(params.dataset_id, [])
        v_idx = context.get("dataset_version", 0)

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result={
                "dataset_id": params.dataset_id,
                "current_version_id": v_idx,
                "total_versions": len(versions),
                "versions": versions
            },
            dataset_id=params.dataset_id,
            dataset_version=v_idx,
            warnings=[],
            mutated_data=False,
            requires_approval=False
        )
