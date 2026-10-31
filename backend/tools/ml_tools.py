"""
DataSense AI Tool Layer — Machine Learning Tools (Phase 3)
Guarantees strict validation isolation, prevents test set leakage, and provides model diagnostics.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolCategory, ToolPermission, ToolExecutionResult, ToolExecutionStatus
from models import AutoMLRequest, AutoMLResponse
from ml_engine import train_automl_pipeline, detect_task_type, ACTIVE_ML_SESSIONS


class PrepareModelingInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")
    target_column: str = Field(description="Feature column to predict")
    task_type: Optional[str] = Field(default="auto", description="'auto', 'classification', or 'regression'")


class TrainAndCompareModelsInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")
    target_column: str = Field(description="Feature column to predict")
    task_type: Optional[str] = Field(default="auto", description="'auto', 'classification', or 'regression'")


class EvaluateFinalModelInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")


class AnalyzeModelErrorsInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")


class ExplainFeatureImportanceInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")


class PrepareModelingTaskTool(BaseTool):
    name = "prepare_modeling_task"
    description = "Validate target feature, verify class distributions/imbalance, and check for leakage or constant variance."
    category = ToolCategory.ML
    permission = ToolPermission.READ_ONLY
    input_schema = PrepareModelingInput

    def execute(self, params: PrepareModelingInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found."
            )

        if params.target_column not in df.columns:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Target column '{params.target_column}' does not exist in dataset."
            )

        target_series = df[params.target_column].dropna()
        if len(target_series) < 10:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Target column '{params.target_column}' has insufficient observations ({len(target_series)} rows; minimum 10 required for ML)."
            )

        inferred_task = detect_task_type(target_series, params.task_type)
        warnings: List[str] = []

        class_distribution = None
        if inferred_task == "classification":
            unique_classes = target_series.nunique()
            if unique_classes < 2:
                return ToolExecutionResult(
                    tool_name=self.name,
                    status=ToolExecutionStatus.ERROR.value,
                    error_details=f"Classification target '{params.target_column}' has only 1 unique class ({target_series.iloc[0]}). Modeling requires at least 2 distinct classes."
                )
            vc = target_series.value_counts()
            class_distribution = {str(k): int(v) for k, v in vc.items()}
            minority_pct = (vc.min() / len(target_series)) * 100
            if minority_pct < 15.0:
                warnings.append(
                    f"Severe class imbalance detected: minority class represents only {round(minority_pct, 1)}% of samples. Stratification and balanced scoring (Macro F1) will be enforced."
                )
        else:
            std_val = float(target_series.std())
            if std_val == 0 or np.isnan(std_val):
                return ToolExecutionResult(
                    tool_name=self.name,
                    status=ToolExecutionStatus.ERROR.value,
                    error_details=f"Regression target '{params.target_column}' has zero variance (constant value). Target cannot be predicted."
                )

        # Inspect candidate feature columns
        feature_cols = [c for c in df.columns if c != params.target_column]
        leakage_candidates = []
        for c in feature_cols:
            if df[c].nunique() >= len(df) * 0.95 and not pd.api.types.is_numeric_dtype(df[c]):
                leakage_candidates.append(c)

        if leakage_candidates:
            warnings.append(
                f"Features {leakage_candidates} appear to be unique IDs or high-cardinality keys. Will be excluded to prevent data leakage."
            )

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result={
                "dataset_id": params.dataset_id,
                "target_column": params.target_column,
                "inferred_task_type": inferred_task,
                "valid_samples": len(target_series),
                "total_candidate_features": len(feature_cols),
                "class_distribution": class_distribution,
                "potential_leakage_features": leakage_candidates,
                "ready_for_training": True
            },
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            warnings=warnings,
            mutated_data=False,
            requires_approval=False
        )


class TrainAndCompareModelsTool(BaseTool):
    name = "train_and_compare_models"
    description = "Run leakage-free AutoML tournament evaluating candidate models on development data before final test evaluation."
    category = ToolCategory.ML
    permission = ToolPermission.READ_ONLY
    input_schema = TrainAndCompareModelsInput

    def execute(self, params: TrainAndCompareModelsInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found."
            )

        req = AutoMLRequest(
            dataset_id=params.dataset_id,
            target_column=params.target_column,
            task_type=params.task_type or "auto"
        )

        try:
            automl_response = train_automl_pipeline(df, req)
        except Exception as e:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"AutoML tournament failed: {str(e)}"
            )

        # Store in session storage
        ml_storage = context.get("last_ml_storage")
        if ml_storage is not None:
            ml_storage[params.dataset_id] = automl_response

        winner_item = next(
            (item for item in automl_response.leaderboard if item.model_name == automl_response.best_model_name),
            automl_response.leaderboard[0] if automl_response.leaderboard else None
        )
        val_score = winner_item.val_score if winner_item else 0.0
        test_score = winner_item.test_score if winner_item else None

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result=automl_response.dict() if hasattr(automl_response, "dict") else automl_response.model_dump(),
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            warnings=[f"Model selected strictly on validation score ({automl_response.best_model_name}: val_score={val_score:.3f})"],
            mutated_data=False,
            requires_approval=False,
            provenance={
                "best_model": automl_response.best_model_name,
                "validation_metric": automl_response.model_selection_metric,
                "val_score": val_score,
                "test_score": test_score
            }
        )


class EvaluateFinalModelTool(BaseTool):
    name = "evaluate_final_model"
    description = "Inspect the winning model's independent test set evaluation, strictly isolated from candidate selection."
    category = ToolCategory.ML
    permission = ToolPermission.READ_ONLY
    input_schema = EvaluateFinalModelInput

    def execute(self, params: EvaluateFinalModelInput, context: Dict[str, Any]) -> ToolExecutionResult:
        ml_storage = context.get("last_ml_storage", {})
        ml_res: Optional[AutoMLResponse] = ml_storage.get(params.dataset_id)

        if not ml_res:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"No trained model found for dataset '{params.dataset_id}'. Run 'train_and_compare_models' first."
            )

        winner_item = next(
            (item for item in ml_res.leaderboard if item.model_name == ml_res.best_model_name),
            ml_res.leaderboard[0] if ml_res.leaderboard else None
        )
        val_score = winner_item.val_score if winner_item else 0.0
        test_score = winner_item.test_score if winner_item else None

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result={
                "dataset_id": params.dataset_id,
                "best_model_name": ml_res.best_model_name,
                "task_type": ml_res.task_type,
                "target_column": ml_res.target_column,
                "selection_validation_score": val_score,
                "final_independent_test_score": test_score,
                "test_metrics": ml_res.best_model_metrics,
                "selection_criterion": ml_res.model_selection_metric,
                "isolation_guarantee": (
                    "Strict final test-set isolation enforced: Candidate models were ranked and selected exclusively "
                    "on training/validation data. The final test set was evaluated only once on the winning model."
                )
            },
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            warnings=[],
            mutated_data=False,
            requires_approval=False
        )


class AnalyzeModelErrorsTool(BaseTool):
    name = "analyze_model_errors"
    description = "Examine residual error distribution, error spread, and worst-case prediction failures."
    category = ToolCategory.ML
    permission = ToolPermission.READ_ONLY
    input_schema = AnalyzeModelErrorsInput

    def execute(self, params: AnalyzeModelErrorsInput, context: Dict[str, Any]) -> ToolExecutionResult:
        ml_storage = context.get("last_ml_storage", {})
        ml_res: Optional[AutoMLResponse] = ml_storage.get(params.dataset_id)

        if not ml_res or not ml_res.held_out_predictions:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"No trained model predictions available for dataset '{params.dataset_id}'. Run 'train_and_compare_models' first."
            )

        samples = ml_res.held_out_predictions
        errors = [abs(s.actual - s.predicted) for s in samples] if ml_res.task_type == "regression" else [1 if s.actual != s.predicted else 0 for s in samples]

        largest_errors = sorted(samples, key=lambda s: abs(s.actual - s.predicted) if ml_res.task_type == "regression" else (1 if s.actual != s.predicted else 0), reverse=True)[:5]

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result={
                "dataset_id": params.dataset_id,
                "task_type": ml_res.task_type,
                "total_held_out_samples_evaluated": len(samples),
                "mean_absolute_error_on_samples": round(float(np.mean(errors)), 3) if ml_res.task_type == "regression" else None,
                "misclassification_rate_on_samples": round(float(np.mean(errors)), 3) if ml_res.task_type == "classification" else None,
                "worst_prediction_cases": [
                    {
                        "row_index": s.row_index,
                        "actual": s.actual,
                        "predicted": s.predicted,
                        "discrepancy": round(abs(s.actual - s.predicted), 3) if ml_res.task_type == "regression" else ("MISCLASSIFIED" if s.actual != s.predicted else "CORRECT")
                    }
                    for s in largest_errors
                ]
            },
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            warnings=[],
            mutated_data=False,
            requires_approval=False
        )


class ExplainFeatureImportanceTool(BaseTool):
    name = "explain_feature_importance"
    description = "Rank and explain which input features drove the model's predictions."
    category = ToolCategory.ML
    permission = ToolPermission.READ_ONLY
    input_schema = ExplainFeatureImportanceInput

    def execute(self, params: ExplainFeatureImportanceInput, context: Dict[str, Any]) -> ToolExecutionResult:
        ml_storage = context.get("last_ml_storage", {})
        ml_res: Optional[AutoMLResponse] = ml_storage.get(params.dataset_id)

        if not ml_res or not ml_res.feature_importance:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"No feature importances found for dataset '{params.dataset_id}'. Run 'train_and_compare_models' first."
            )

        ranked_features = [
            {
                "feature": fi.feature,
                "importance": fi.importance,
                "percentage": fi.percentage
            }
            for fi in ml_res.feature_importance
        ]

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result={
                "dataset_id": params.dataset_id,
                "model_name": ml_res.best_model_name,
                "features_ranked": ranked_features
            },
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            warnings=[],
            mutated_data=False,
            requires_approval=False
        )
