"""
DataSense AI Tool Layer — Cleaning Tools (Phase 3)
Guarantees raw data immutability, uncommitted proposal staging, and versioned approval.
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolCategory, ToolPermission, ToolExecutionResult, ToolExecutionStatus
from models import CleaningActionRequest, CleaningProposal
from data_cleaner import preview_cleaning_action, apply_cleaning_action
from data_profiler import profile_dataframe


class PreviewCleaningInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")
    rule: str = Field(description="Cleaning rule: 'deduplicate', 'impute_missing', 'normalize_categories', 'cap_outliers', 'drop_rows', 'drop_column'")
    column: Optional[str] = Field(default=None, description="Target column name")
    strategy: Optional[str] = Field(default=None, description="Cleaning strategy: 'median', 'mean', 'mode', 'cap_iqr', 'strip_and_titlecase', 'drop_rows'")
    fill_value: Optional[str] = Field(default=None, description="Custom fill value for imputation")


class ApplyCleaningInput(BaseModel):
    proposal_id: str = Field(description="ID of staged proposal to apply after human approval")


class RejectCleaningInput(BaseModel):
    proposal_id: str = Field(description="ID of staged proposal to reject")


class PreviewCleaningChangesTool(BaseTool):
    name = "preview_cleaning_changes"
    description = "Generate uncommitted diff preview of proposed data cleaning without modifying raw dataset."
    category = ToolCategory.CLEANING
    permission = ToolPermission.READ_ONLY
    input_schema = PreviewCleaningInput

    def execute(self, params: PreviewCleaningInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found."
            )

        v_idx = context.get("dataset_version", 0)
        req = CleaningActionRequest(
            dataset_id=params.dataset_id,
            rule=params.rule,
            column=params.column,
            strategy=params.strategy,
            fill_value=params.fill_value
        )

        try:
            preview = preview_cleaning_action(df, req)
        except Exception as e:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Failed to generate cleaning preview: {str(e)}"
            )

        affected_rows = preview.get("affected_rows", 0)
        affected_pct = preview.get("affected_percentage", 0.0)
        is_consequential = (
            params.strategy == "drop_rows" or
            params.rule in ["drop_rows", "drop_column"] or
            affected_pct >= 5.0
        )

        proposal_id = f"prop_{uuid.uuid4().hex[:8]}"
        proposal = CleaningProposal(
            proposal_id=proposal_id,
            dataset_id=params.dataset_id,
            base_version_id=v_idx,
            rule=params.rule,
            column=params.column,
            strategy=params.strategy,
            fill_value=params.fill_value,
            reason=f"Proposed {params.rule} on {params.column or 'dataset'} (strategy: {params.strategy})",
            is_consequential=is_consequential,
            affected_rows=affected_rows,
            affected_percentage=affected_pct,
            transformation_summary=preview.get("transformation_summary", f"{params.rule} on {params.column}"),
            diff_preview=preview,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="pending"
        )

        # Store proposal in context proposals storage if available
        proposals_store = context.get("proposals_storage")
        if proposals_store is not None:
            proposals_store.setdefault(params.dataset_id, {})[proposal_id] = proposal

        warnings = []
        if is_consequential:
            warnings.append(
                f"Consequential action: affects {affected_rows} rows ({affected_pct}%). Explicit user approval required."
            )

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.PENDING_APPROVAL.value if is_consequential else ToolExecutionStatus.SUCCESS.value,
            result={
                "proposal_id": proposal_id,
                "is_consequential": is_consequential,
                "affected_rows": affected_rows,
                "affected_percentage": affected_pct,
                "base_version_id": v_idx,
                "diff_preview": preview
            },
            dataset_id=params.dataset_id,
            dataset_version=v_idx,
            warnings=warnings,
            mutated_data=False,
            requires_approval=is_consequential,
            provenance={"staged_at": proposal.created_at, "rule": params.rule, "column": params.column}
        )


class ApplyApprovedCleaningTool(BaseTool):
    name = "apply_approved_cleaning"
    description = "Commit an approved transformation proposal, advancing dataset version and preserving immutable raw data."
    category = ToolCategory.CLEANING
    permission = ToolPermission.REQUIRES_APPROVAL
    input_schema = ApplyCleaningInput

    def execute(self, params: ApplyCleaningInput, context: Dict[str, Any]) -> ToolExecutionResult:
        proposals_store = context.get("proposals_storage", {})
        target_proposal: Optional[CleaningProposal] = None
        target_dataset_id: Optional[str] = None

        for d_id, p_map in proposals_store.items():
            if params.proposal_id in p_map:
                target_proposal = p_map[params.proposal_id]
                target_dataset_id = d_id
                break

        if not target_proposal or not target_dataset_id:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Proposal '{params.proposal_id}' not found."
            )

        if target_proposal.status != "pending":
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Proposal '{params.proposal_id}' cannot be applied; current status is '{target_proposal.status}'."
            )

        current_v_idx = context.get("dataset_version", 0)
        if target_proposal.base_version_id is not None and target_proposal.base_version_id != current_v_idx:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=(
                    f"Proposal is stale: created against dataset version {target_proposal.base_version_id}, "
                    f"but active dataset version is {current_v_idx}. Re-preview required."
                ),
                dataset_id=target_dataset_id,
                dataset_version=current_v_idx
            )

        df = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{target_dataset_id}' not found in active session."
            )

        req = CleaningActionRequest(
            dataset_id=target_dataset_id,
            rule=target_proposal.rule,
            column=target_proposal.column,
            strategy=target_proposal.strategy,
            fill_value=target_proposal.fill_value
        )

        try:
            clean_df, audit_record = apply_cleaning_action(df, req)
        except Exception as e:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Failed to apply approved transformation: {str(e)}"
            )

        # Mutate dataset in storage if provided
        dataset_storage = context.get("dataset_storage")
        if dataset_storage is not None:
            dataset_storage[target_dataset_id] = clean_df

        # Advance version
        new_v_idx = current_v_idx + 1
        versions_storage = context.get("versions_storage")
        if versions_storage is not None:
            versions_storage.setdefault(target_dataset_id, []).append({
                "version_id": new_v_idx,
                "parent_version_id": current_v_idx,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation": target_proposal.rule,
                "description": f"Applied {target_proposal.rule} (strategy: {target_proposal.strategy}) on {target_proposal.column or 'dataset'}",
                "affected_rows": audit_record.affected_rows,
                "affected_cols": 1 if target_proposal.column else len(clean_df.columns),
                "total_rows": len(clean_df),
                "total_cols": len(clean_df.columns)
            })

        # Append audit record
        audit_storage = context.get("audit_storage")
        if audit_storage is not None:
            audit_storage.setdefault(target_dataset_id, []).append(audit_record)

        target_proposal.status = "approved"

        # Update profile if profile storage exists
        profile_storage = context.get("profile_storage")
        meta = context.get("metadata", {"name": "Dataset", "filename": "data.csv"})
        if profile_storage is not None:
            new_profile = profile_dataframe(clean_df, target_dataset_id, meta.get("name", "Dataset"), meta.get("filename", "data.csv"))
            profile_storage[target_dataset_id] = new_profile

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result={
                "proposal_id": params.proposal_id,
                "status": "approved_and_committed",
                "new_version_id": new_v_idx,
                "affected_rows": audit_record.affected_rows,
                "total_records": len(clean_df),
                "columns_remaining": len(clean_df.columns)
            },
            dataset_id=target_dataset_id,
            dataset_version=new_v_idx,
            warnings=[],
            mutated_data=True,
            requires_approval=False,
            provenance={"applied_rule": target_proposal.rule, "affected_rows": audit_record.affected_rows}
        )


class RejectCleaningProposalTool(BaseTool):
    name = "reject_cleaning_proposal"
    description = "Reject a proposed data transformation without mutating the active dataset."
    category = ToolCategory.CLEANING
    permission = ToolPermission.READ_ONLY
    input_schema = RejectCleaningInput

    def execute(self, params: RejectCleaningInput, context: Dict[str, Any]) -> ToolExecutionResult:
        proposals_store = context.get("proposals_storage", {})
        target_proposal: Optional[CleaningProposal] = None

        for d_id, p_map in proposals_store.items():
            if params.proposal_id in p_map:
                target_proposal = p_map[params.proposal_id]
                break

        if not target_proposal:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Proposal '{params.proposal_id}' not found."
            )

        if target_proposal.status != "pending":
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Proposal '{params.proposal_id}' is already '{target_proposal.status}'."
            )

        target_proposal.status = "rejected"

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result={
                "proposal_id": params.proposal_id,
                "status": "rejected",
                "message": "Proposal rejected without mutating dataset."
            },
            dataset_id=target_proposal.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            warnings=[],
            mutated_data=False,
            requires_approval=False
        )
