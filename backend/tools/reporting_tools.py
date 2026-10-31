"""
DataSense AI Tool Layer — Reporting and Export Tools (Phase 3)
Synthesizes verified data profiling, auditable provenance, and ML evaluation.
"""
import io
import csv
from typing import Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolCategory, ToolPermission, ToolExecutionResult, ToolExecutionStatus
from models import ExecutiveReportRequest, ExecutiveReportResponse
from report_generator import generate_executive_report
from data_profiler import profile_dataframe


class GenerateReportInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")
    objective: Optional[str] = Field(default=None, description="Analytical goal or question")
    include_audit: Optional[bool] = Field(default=True, description="Include cleaning audit provenance")
    include_ml: Optional[bool] = Field(default=True, description="Include AutoML evaluation results")


class ExportArtifactInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")
    artifact_format: str = Field(default="markdown", description="'markdown', 'html', or 'csv'")


class GenerateAnalysisReportTool(BaseTool):
    name = "generate_analysis_report"
    description = "Synthesize data health, audit provenance, and ML evaluation into structured executive brief."
    category = ToolCategory.REPORTING
    permission = ToolPermission.READ_ONLY
    input_schema = GenerateReportInput

    def execute(self, params: GenerateReportInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found."
            )

        meta = context.get("metadata", {"name": "Dataset", "filename": "data.csv"})
        profile_storage = context.get("profile_storage", {})
        profile = profile_storage.get(params.dataset_id) or profile_dataframe(
            df, params.dataset_id, meta.get("name", "Dataset"), meta.get("filename", "data.csv")
        )

        audit_history = context.get("audit_storage", {}).get(params.dataset_id, [])
        ml_result = context.get("last_ml_storage", {}).get(params.dataset_id)

        try:
            report: ExecutiveReportResponse = generate_executive_report(
                profile=profile,
                audit_history=audit_history if params.include_audit else [],
                ml_response=ml_result if params.include_ml else None,
                objective=params.objective
            )
        except Exception as e:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Report generation error: {str(e)}"
            )

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result=report.dict() if hasattr(report, "dict") else report.model_dump(),
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            warnings=[],
            mutated_data=False,
            requires_approval=False
        )


class ExportAnalysisArtifactTool(BaseTool):
    name = "export_analysis_artifact"
    description = "Export verified analysis summary as Markdown, HTML, or CSV artifact."
    category = ToolCategory.REPORTING
    permission = ToolPermission.READ_ONLY
    input_schema = ExportArtifactInput

    def execute(self, params: ExportArtifactInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found."
            )

        fmt = params.artifact_format.lower()
        meta = context.get("metadata", {"name": "Dataset", "filename": "data.csv"})
        profile_storage = context.get("profile_storage", {})
        profile = profile_storage.get(params.dataset_id) or profile_dataframe(
            df, params.dataset_id, meta.get("name", "Dataset"), meta.get("filename", "data.csv")
        )

        audit_history = context.get("audit_storage", {}).get(params.dataset_id, [])
        ml_result = context.get("last_ml_storage", {}).get(params.dataset_id)

        report = generate_executive_report(
            profile=profile,
            audit_history=audit_history,
            ml_response=ml_result,
            objective=None
        )

        if fmt == "html":
            content = report.html_content
            mime_type = "text/html"
            filename = f"{meta['filename'].rsplit('.', 1)[0]}_Executive_Brief.html"
        elif fmt == "csv":
            stream = io.StringIO()
            writer = csv.writer(stream)
            writer.writerow(["=== EXECUTIVE DATA HEALTH & SUMMARY ==="])
            writer.writerow(["Dataset Name", meta["name"]])
            writer.writerow(["Filename", meta["filename"]])
            writer.writerow(["Total Rows", profile.metadata.rows])
            writer.writerow(["Total Columns", profile.metadata.columns])
            writer.writerow(["Completeness", f"{100 - profile.metadata.missing_percentage:.1f}%"])
            writer.writerow(["Duplicate Rows", profile.metadata.duplicate_rows])
            writer.writerow(["Health Score", f"{profile.health.overall_score}/100 ({profile.health.grade})"])
            writer.writerow([])
            writer.writerow(["=== FEATURE DICTIONARY ==="])
            writer.writerow(["Feature", "Type", "Missing Count", "Unique Count", "Mean", "Median", "Std"])
            for c in profile.columns:
                writer.writerow([
                    c.column, c.inferred_type, c.null_count, c.unique_count,
                    c.mean if c.mean is not None else "",
                    c.median if c.median is not None else "",
                    c.std if c.std is not None else ""
                ])
            content = stream.getvalue()
            mime_type = "text/csv"
            filename = f"{meta['filename'].rsplit('.', 1)[0]}_Report.csv"
        else:
            content = report.markdown_content
            mime_type = "text/markdown"
            filename = f"{meta['filename'].rsplit('.', 1)[0]}_Executive_Brief.md"

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result={
                "format": fmt,
                "mime_type": mime_type,
                "filename": filename,
                "content_length": len(content),
                "content_preview": content[:500]
            },
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            warnings=[],
            mutated_data=False,
            requires_approval=False
        )
