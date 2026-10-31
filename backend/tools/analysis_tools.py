"""
DataSense AI Tool Layer — Analysis and Chart Tools (Phase 3)
Grounds chart interpretation, group comparison, and anomaly detection in the actual dataset.
"""
from typing import Dict, Any, List, Optional, Type, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolCategory, ToolPermission, ToolExecutionResult, ToolExecutionStatus
from models import CustomChartRequest, ChartSpec, GraphFindingItem, GraphIntelligenceResponse
from analysis_engine import generate_custom_chart, sanitize_val
from data_utils import normalize_dataframe_headers_and_columns


class ExploratoryAnalysisInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")
    columns: Optional[List[str]] = Field(default=None, description="Columns to analyze")


class GenerateChartInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")
    x_column: str = Field(description="Independent or category column")
    y_column: Optional[str] = Field(default=None, description="Dependent or numeric metric column")
    chart_type: str = Field(default="bar", description="'bar', 'line', 'scatter', 'histogram', or 'pie'")
    aggregation: Optional[str] = Field(default="mean", description="'mean', 'sum', 'count', 'max', 'min'")
    orientation: Optional[str] = Field(default="auto", description="'auto', 'h', or 'v'")
    sort_order: Optional[str] = Field(default="desc", description="'desc', 'asc', or 'none'")
    top_n: Optional[int] = Field(default=20, description="Top N categories to display (0 for All)")


class AnalyzeChartInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")
    x_column: str = Field(description="Plotted X column")
    y_column: Optional[str] = Field(default=None, description="Plotted Y column (if applicable)")
    chart_type: str = Field(default="bar", description="Chart type")
    aggregation: Optional[str] = Field(default="mean", description="Aggregation applied")
    chart_spec: Optional[Dict[str, Any]] = Field(default=None, description="Rendered chart specification")


class CompareGroupsInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")
    group_column: str = Field(description="Categorical grouping column")
    metric_column: str = Field(description="Numeric metric to compare across groups")


class InvestigateAnomaliesInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")
    column: str = Field(description="Target column to inspect for statistical outliers")
    method: Optional[str] = Field(default="iqr", description="'iqr' (1.5x) or 'zscore' (3.0 sigma)")


class RunExploratoryAnalysisTool(BaseTool):
    name = "run_exploratory_analysis"
    description = "Compute descriptive summary statistics, correlation pairs, and distributions across features."
    category = ToolCategory.ANALYSIS
    permission = ToolPermission.READ_ONLY
    input_schema = ExploratoryAnalysisInput

    def execute(self, params: ExploratoryAnalysisInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found."
            )

        target_cols = params.columns if params.columns else list(df.columns)
        unknown_cols = [c for c in target_cols if c not in df.columns]
        if unknown_cols:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Columns not found in dataset: {unknown_cols}"
            )

        summaries = {}
        for col in target_cols:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                clean_s = s.dropna()
                summaries[col] = {
                    "type": "numeric",
                    "count": int(clean_s.count()),
                    "missing": int(s.isna().sum()),
                    "mean": sanitize_val(clean_s.mean()),
                    "std": sanitize_val(clean_s.std()) if len(clean_s) > 1 else 0.0,
                    "median": sanitize_val(clean_s.median()),
                    "iqr": sanitize_val(clean_s.quantile(0.75) - clean_s.quantile(0.25)) if len(clean_s) > 0 else 0.0,
                    "min": sanitize_val(clean_s.min()),
                    "max": sanitize_val(clean_s.max())
                }
            else:
                vc = s.value_counts(dropna=True).head(5)
                summaries[col] = {
                    "type": "categorical",
                    "count": int(s.notna().sum()),
                    "missing": int(s.isna().sum()),
                    "unique_values": int(s.nunique(dropna=True)),
                    "top_categories": {str(k): int(v) for k, v in vc.items()}
                }

        # Numeric pairwise correlations
        correlations = []
        numeric_df = df[[c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]].dropna()
        if len(numeric_df.columns) >= 2 and len(numeric_df) >= 3:
            corr_mat = numeric_df.corr(method="pearson")
            for i, c1 in enumerate(corr_mat.columns):
                for j, c2 in enumerate(corr_mat.columns):
                    if i < j:
                        r_val = float(corr_mat.loc[c1, c2])
                        if not np.isnan(r_val):
                            correlations.append({
                                "feature_1": c1,
                                "feature_2": c2,
                                "pearson_r": round(r_val, 3),
                                "strength": "strong" if abs(r_val) >= 0.7 else ("moderate" if abs(r_val) >= 0.35 else "weak")
                            })
            correlations.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result={
                "dataset_id": params.dataset_id,
                "dataset_version": context.get("dataset_version", 0),
                "analyzed_columns": target_cols,
                "descriptive_statistics": summaries,
                "notable_correlations": correlations[:10]
            },
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            warnings=[],
            mutated_data=False,
            requires_approval=False
        )


class GenerateChartTool(BaseTool):
    name = "generate_chart"
    description = "Create interactive Plotly/ChartSpec aggregated directly from active dataset."
    category = ToolCategory.ANALYSIS
    permission = ToolPermission.READ_ONLY
    input_schema = GenerateChartInput

    def execute(self, params: GenerateChartInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found."
            )

        if params.x_column not in df.columns:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Column '{params.x_column}' does not exist in dataset."
            )

        if params.y_column and params.y_column not in df.columns:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Column '{params.y_column}' does not exist in dataset."
            )

        req = CustomChartRequest(
            dataset_id=params.dataset_id,
            x_column=params.x_column,
            y_column=params.y_column,
            chart_type=params.chart_type,
            aggregation=params.aggregation or "mean",
            orientation=params.orientation or "auto",
            sort_order=params.sort_order or "desc",
            top_n=params.top_n
        )

        try:
            spec: ChartSpec = generate_custom_chart(df, req)
        except Exception as e:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Chart generation error: {str(e)}"
            )

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result=spec.dict() if hasattr(spec, "dict") else spec.model_dump(),
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            warnings=[],
            mutated_data=False,
            requires_approval=False
        )


class AnalyzeChartTool(BaseTool):
    name = "analyze_chart"
    description = "Perform deep data-grounded intelligence on a plotted chart using the underlying observations."
    category = ToolCategory.ANALYSIS
    permission = ToolPermission.READ_ONLY
    input_schema = AnalyzeChartInput

    def execute(self, params: AnalyzeChartInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found."
            )

        # Defensive DataFrame normalization: flatten MultiIndex and deduplicate column names
        df = normalize_dataframe_headers_and_columns(df)

        x_col = params.x_column
        y_col = params.y_column

        col_map = {str(c).lower(): str(c) for c in df.columns}

        # Normalize x_col (handling " Bins" suffix and case-insensitivity)
        if x_col not in df.columns:
            clean_x = str(x_col).replace(" Bins", "").strip()
            if clean_x in df.columns:
                x_col = clean_x
            elif str(x_col).lower() in col_map:
                x_col = col_map[str(x_col).lower()]
            elif clean_x.lower() in col_map:
                x_col = col_map[clean_x.lower()]
            else:
                x_col = df.columns[0]

        # Normalize y_col (handling "Mean Sales", "Record Count", "Frequency Count", etc.)
        if y_col:
            raw_y = str(y_col).strip()
            if any(k in raw_y.lower() for k in ["record count", "frequency count", "count", "frequency"]) and raw_y not in df.columns:
                y_col = None
            elif raw_y in df.columns:
                y_col = raw_y
            elif raw_y.lower() in col_map:
                y_col = col_map[raw_y.lower()]
            else:
                parts = raw_y.split(" ", 1)
                if len(parts) > 1 and parts[1].lower() in col_map:
                    y_col = col_map[parts[1].lower()]
                else:
                    y_col = None

        # CRITICAL FIX: If y_col is identical to x_col, this is a univariate measure analysis,
        # not a group-by of a measure by itself.
        if y_col and y_col == x_col:
            y_col = None

        # Ensure single 1D Series for x and y
        raw_x = df[x_col]
        x_series = raw_x.iloc[:, 0] if isinstance(raw_x, pd.DataFrame) else raw_x

        y_series = None
        if y_col:
            raw_y = df[y_col]
            y_series = raw_y.iloc[:, 0] if isinstance(raw_y, pd.DataFrame) else raw_y

        v_idx = context.get("dataset_version", 0)
        findings: List[GraphFindingItem] = []
        anomalies: List[str] = []
        group_comparisons: List[str] = []
        hidden_patterns: List[str] = []
        quality_concerns: List[str] = []
        suggested_followups: List[str] = []

        try:
            # 1. Data Quality Check on Plotted Columns
            x_missing = int(x_series.isna().sum())
            if x_missing > 0:
                quality_concerns.append(f"Feature '{x_col}' contains {x_missing} missing values ({round(x_missing/len(df)*100, 1)}%) excluded from computation.")
            if y_series is not None:
                y_missing = int(y_series.isna().sum())
                if y_missing > 0:
                    quality_concerns.append(f"Feature '{y_col}' contains {y_missing} missing values ({round(y_missing/len(df)*100, 1)}%) excluded from computation.")

            # 2. Underlying Distribution Calculations
            if y_series is not None and pd.api.types.is_numeric_dtype(y_series):
                # Build isolated 2-column DataFrame with single 1D series
                sub_df = pd.DataFrame({"_group": x_series, "_measure": y_series}).dropna()
                total_obs = len(sub_df)
                if total_obs == 0:
                    return ToolExecutionResult(
                        tool_name=self.name,
                        status=ToolExecutionStatus.ERROR.value,
                        error_details=f"No overlapping non-null observations between '{x_col}' and '{y_col}'."
                    )

                # Check if x_series is continuous or scatter plot with sufficient cardinality
                is_x_continuous = (pd.api.types.is_float_dtype(x_series) or params.chart_type == "scatter") and sub_df["_group"].nunique() >= 15

                if is_x_continuous and (params.chart_type == "scatter" or sub_df["_group"].nunique() > 25):
                    # Bivariate correlation & regression analysis for continuous-vs-continuous measures
                    r = float(sub_df["_group"].corr(sub_df["_measure"]))
                    poly = np.polyfit(sub_df["_group"].astype(float), sub_df["_measure"].astype(float), 1)
                    slope, intercept = float(poly[0]), float(poly[1])
                    findings.append(GraphFindingItem(
                        category="Bivariate Relationship",
                        title="Correlation & Regression Fit",
                        detail=f"Bivariate relationship between '{x_col}' and '{y_col}': Pearson r = {r:.4f} across N = {total_obs} records (linear slope: {slope:+.4f}, intercept: {intercept:.4f}).",
                        confidence="High"
                    ))
                    group_comparisons.append(f"Linear regression fit: '{y_col}' shifts by {slope:+.4f} units per unit increase in '{x_col}'.")
                else:
                    agg_type = params.aggregation or "mean"
                    grouped = sub_df.groupby("_group")["_measure"]
                    g_means = grouped.mean()
                    g_medians = grouped.median()
                    g_counts = grouped.count()
                    g_stds = grouped.std().fillna(0)

                    # Sample size warnings
                    small_groups = g_counts[g_counts < 5]
                    if len(small_groups) > 0:
                        quality_concerns.append(
                            f"{len(small_groups)} group(s) have fewer than 5 observations (e.g. {', '.join(str(k) for k in small_groups.index[:3])}), making aggregated metrics volatile."
                        )

                    if len(g_means) > 0:
                        # Chronological or ordinal trend analysis across groups if x is ordered/temporal
                        try:
                            sorted_grps = g_means.index
                            num_grps = pd.to_numeric(pd.Series(sorted_grps), errors="coerce")
                            if num_grps.notna().all() and len(num_grps) >= 3:
                                grps_arr = num_grps.values.astype(float)
                                means_arr = g_means.values.astype(float)
                                sort_order = np.argsort(grps_arr)
                                grps_sorted = grps_arr[sort_order]
                                means_sorted = means_arr[sort_order]
                                slope, intercept = np.polyfit(grps_sorted, means_sorted, 1)
                                direction = "Upward" if slope > 0.0001 else ("Downward" if slope < -0.0001 else "Flat")
                                findings.append(GraphFindingItem(
                                    category="Trend Analysis",
                                    title=f"Chronological {direction} Trend",
                                    detail=f"Across chronological intervals '{x_col}' ({int(grps_sorted[0])} to {int(grps_sorted[-1])}), '{y_col}' exhibits an {direction.lower()} slope of {slope:+.4f} units per step.",
                                    confidence="High"
                                ))
                                hidden_patterns.append(
                                    f"Observed trend: '{y_col}' shifted from {means_sorted[0]:.2f} at {int(grps_sorted[0])} to {means_sorted[-1]:.2f} at {int(grps_sorted[-1])} (net change {means_sorted[-1] - means_sorted[0]:+.2f})."
                                )
                        except Exception as e:
                            logger.debug(f"Trend fitting note: {e}")

                        max_grp = g_means.idxmax()
                        min_grp = g_means.idxmin()
                        max_val = round(float(g_means[max_grp]), 2)
                        min_val = round(float(g_means[min_grp]), 2)
                        overall_avg = round(float(sub_df["_measure"].mean()), 2)
                        overall_med = round(float(sub_df["_measure"].median()), 2)

                        findings.append(GraphFindingItem(
                            category="Observed Fact",
                            title="Leading and Lagging Groups",
                            detail=f"The highest mean '{y_col}' is observed in '{max_grp}' ({max_val}), while the lowest is in '{min_grp}' ({min_val}). Active dataset benchmark mean is {overall_avg}.",
                            confidence="High"
                        ))

                        ratio = round(max_val / (min_val if min_val > 0 else 1.0), 2) if min_val > 0 else "N/A"
                        group_comparisons.append(
                            f"'{max_grp}' exceeds '{min_grp}' by {round(max_val - min_val, 2)} units (ratio: {ratio}x)."
                        )

                        # Within-group variance
                        highest_variance_grp = g_stds.idxmax()
                        highest_std = round(float(g_stds[highest_variance_grp]), 2)
                        if highest_std > 0:
                            hidden_patterns.append(
                                f"Substantial internal variance within '{highest_variance_grp}' (std = {highest_std}), indicating diverse sub-cohorts rather than a uniform effect."
                            )

                        # Check for outliers across observations
                        y_data = sub_df["_measure"]
                        q25, q75 = float(y_data.quantile(0.25)), float(y_data.quantile(0.75))
                        iqr = q75 - q25
                        iqr_outliers = sub_df[(sub_df["_measure"] < q25 - 1.5 * iqr) | (sub_df["_measure"] > q75 + 1.5 * iqr)]
                        if len(iqr_outliers) > 0:
                            anomalies.append(
                                f"Detected {len(iqr_outliers)} statistical outlier records beyond 1.5x IQR in '{y_col}' (range [{round(q25 - 1.5*iqr, 1)}, {round(q75 + 1.5*iqr, 1)}])."
                            )

                        suggested_followups.append(f"Run a two-sample t-test or Mann-Whitney U test between '{max_grp}' and '{min_grp}' to verify if the difference is statistically distinguishable.")
                        suggested_followups.append(f"Examine distribution of '{y_col}' across other potential confounding dimensions.")

            elif pd.api.types.is_numeric_dtype(x_series) or params.chart_type == "histogram":
                clean_s = pd.to_numeric(x_series, errors="coerce").dropna()
                n_obs = len(clean_s)
                if n_obs > 0:
                    mean_val = round(float(clean_s.mean()), 4)
                    med_val = round(float(clean_s.median()), 4)
                    std_val = round(float(clean_s.std()), 4) if n_obs > 1 else 0.0
                    skew_val = round(float(clean_s.skew()), 4) if n_obs > 2 else 0.0
                    min_val = round(float(clean_s.min()), 4)
                    max_val = round(float(clean_s.max()), 4)

                    # Look for temporal/dimension column for labeling min/max/top items
                    dim_col = None
                    for candidate in ["year", "date", "time", "period", "timestamp", "month"]:
                        for c in df.columns:
                            if candidate in str(c).lower() and c != x_col:
                                dim_col = c
                                break
                        if dim_col:
                            break

                    if dim_col and dim_col in df.columns:
                        raw_dim = df.loc[clean_s.index, dim_col]
                        dim_series = raw_dim.iloc[:, 0] if isinstance(raw_dim, pd.DataFrame) else raw_dim
                        idx_labels = [str(t) for t in dim_series]
                        dim_label = "Year" if "year" in str(dim_col).lower() else ("Date" if "date" in str(dim_col).lower() else str(dim_col))
                    else:
                        idx_labels = [str(x[0] if isinstance(x, tuple) else x) for x in clean_s.index]
                        dim_label = "Index"

                    indexed_s = pd.Series(clean_s.values, index=idx_labels)
                    min_id = str(indexed_s.idxmin())
                    max_id = str(indexed_s.idxmax())
                    top_items = {str(k): round(float(v), 4) for k, v in indexed_s.nlargest(min(3, n_obs)).items()}
                    bottom_items = {str(k): round(float(v), 4) for k, v in indexed_s.nsmallest(min(3, n_obs)).items()}

                    top_str = ", ".join(f"{k} ({v})" for k, v in top_items.items())
                    bottom_str = ", ".join(f"{k} ({v})" for k, v in bottom_items.items())

                    findings.append(GraphFindingItem(
                        category="Distribution Summary",
                        title="Distribution Shape & Spread",
                        detail=(
                            f"Sample size N = {n_obs}, mean = {mean_val:.4f}, median = {med_val:.4f}, "
                            f"standard deviation = {std_val:.4f}, skewness = {skew_val:.4f}. "
                            f"Values range from {min_val:.4f} ({dim_label} {min_id}) to {max_val:.4f} ({dim_label} {max_id}). "
                            f"Top values: {top_str}. Bottom values: {bottom_str}."
                        ),
                        confidence="High"
                    ))

                    group_comparisons.append(
                        f"Range spread: Maximum {max_val:.4f} ({dim_label} {max_id}) exceeds minimum {min_val:.4f} ({dim_label} {min_id}) by {round(max_val - min_val, 4)} units."
                    )

                    # Outlier check for distribution
                    q25, q75 = float(clean_s.quantile(0.25)), float(clean_s.quantile(0.75))
                    iqr = q75 - q25
                    iqr_outliers = clean_s[(clean_s < q25 - 1.5 * iqr) | (clean_s > q75 + 1.5 * iqr)]
                    if len(iqr_outliers) > 0:
                        anomalies.append(
                            f"Detected {len(iqr_outliers)} statistical outlier records beyond 1.5x IQR in '{x_col}' (IQR bounds: [{round(q25 - 1.5*iqr, 2)}, {round(q75 + 1.5*iqr, 2)}])."
                        )

                    skew_desc = "approximately symmetric" if abs(skew_val) < 0.5 else ("moderately right-skewed" if skew_val > 0 else "moderately left-skewed")
                    hidden_patterns.append(
                        f"Distribution shape: Skewness of {skew_val:.4f} indicates a {skew_desc} distribution where the median ({med_val:.4f}) reflects central tendency."
                    )

                    suggested_followups.append(f"Perform normality testing (e.g. Shapiro-Wilk) on '{x_col}' to evaluate parametric assumptions.")
                    suggested_followups.append(f"Evaluate chronological progression of '{x_col}' across decadal intervals.")

            else:
                # Single categorical column
                vc = x_series.value_counts(dropna=True)
                if len(vc) > 0:
                    top_cat = vc.index[0]
                    top_count = int(vc.iloc[0])
                    pct = round((top_count / len(df)) * 100, 1)
                    findings.append(GraphFindingItem(
                        category="Observed Fact",
                        title="Dominant Category",
                        detail=f"'{top_cat}' is the most frequent category with {top_count} observations ({pct}% of total records).",
                        confidence="High"
                    ))

                    # Low-frequency category analysis (e.g. restaurant names where most have count = 1)
                    num_ones = int((vc == 1).sum())
                    total_cats = len(vc)
                    if num_ones >= 3 and (num_ones / total_cats >= 0.35):
                        pct_ones = round((num_ones / total_cats) * 100, 1)
                        findings.append(GraphFindingItem(
                            category="Distribution Characteristic",
                            title="Low-Frequency Category Dominance",
                            detail=f"The distribution contains many low-frequency entities: {num_ones} of {total_cats} unique categories ({pct_ones}%) have a count of exactly 1. Differences among these entities do not represent meaningful volume variations.",
                            confidence="High"
                        ))
                        hidden_patterns.append(
                            f"Long-tail categorical spread: {pct_ones}% of categories appear only once (count = 1). Consider focusing on Top N categories for decision-making."
                        )

                    if len(vc) > 1:
                        least_cat = vc.index[-1]
                        least_count = int(vc.iloc[-1])
                        group_comparisons.append(f"Top category '{top_cat}' is {round(top_count / max(1, least_count), 1)}x more prevalent than '{least_cat}' ({least_count}).")

                    suggested_followups.append(f"Cross-tabulate '{x_col}' against target outcomes to test for categorical independence.")

        except Exception as e:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Analysis computation error on '{x_col}': {str(e)}"
            )

        summary_text = (
            f"Underlying analysis over {len(df)} total rows across feature(s) '{x_col}'"
            + (f" and '{y_col}'" if y_col else "")
            + f" (Dataset Version {v_idx})."
        )

        response_obj = GraphIntelligenceResponse(
            summary=summary_text,
            key_findings=findings,
            anomalies_detected=anomalies,
            group_comparisons=group_comparisons,
            hidden_patterns=hidden_patterns,
            data_quality_concerns=quality_concerns,
            suggested_followups=suggested_followups,
            recommended_alternative_charts=["box_plot", "violin_plot"] if y_col else ["horizontal_bar", "frequency_table"],
            causation_disclaimer="Patterns and correlations displayed represent observed sample associations and must not be interpreted as causal mechanisms without controlled experimental evidence."
        )

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result=response_obj.dict() if hasattr(response_obj, "dict") else response_obj.model_dump(),
            dataset_id=params.dataset_id,
            dataset_version=v_idx,
            warnings=quality_concerns,
            mutated_data=False,
            requires_approval=False
        )


class CompareGroupsTool(BaseTool):
    name = "compare_groups"
    description = "Compare a numerical metric across categorical cohorts using genuine dataset observations."
    category = ToolCategory.ANALYSIS
    permission = ToolPermission.READ_ONLY
    input_schema = CompareGroupsInput

    def execute(self, params: CompareGroupsInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found."
            )

        df = normalize_dataframe_headers_and_columns(df)

        if params.group_column not in df.columns:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Group column '{params.group_column}' not found."
            )
        if params.metric_column not in df.columns:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Metric column '{params.metric_column}' not found."
            )

        if params.group_column == params.metric_column:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details="Group column and metric column cannot be identical for group comparison."
            )

        grp_raw = df[params.group_column]
        metric_raw = df[params.metric_column]
        grp_series = grp_raw.iloc[:, 0] if isinstance(grp_raw, pd.DataFrame) else grp_raw
        metric_series = metric_raw.iloc[:, 0] if isinstance(metric_raw, pd.DataFrame) else metric_raw

        clean = pd.DataFrame({
            "_group": grp_series,
            "_metric": pd.to_numeric(metric_series, errors="coerce")
        }).dropna()

        if len(clean) == 0:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Metric column '{params.metric_column}' must contain valid numerical values."
            )

        overall_mean = float(clean["_metric"].mean())
        overall_median = float(clean["_metric"].median())

        group_stats = []
        for g_name, g_df in clean.groupby("_group"):
            vals = g_df["_metric"]
            g_mean = float(vals.mean())
            g_std = float(vals.std()) if len(vals) > 1 else 0.0
            g_median = float(vals.median())
            diff_from_overall_pct = round(((g_mean - overall_mean) / (overall_mean if overall_mean != 0 else 1.0)) * 100, 2)

            group_stats.append({
                "group": str(g_name),
                "count": len(vals),
                "mean": round(g_mean, 3),
                "median": round(g_median, 3),
                "std": round(g_std, 3),
                "min": round(float(vals.min()), 3),
                "max": round(float(vals.max()), 3),
                "diff_from_overall_mean_pct": diff_from_overall_pct
            })

        group_stats.sort(key=lambda x: x["mean"], reverse=True)

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result={
                "dataset_id": params.dataset_id,
                "group_column": params.group_column,
                "metric_column": params.metric_column,
                "overall_mean": round(overall_mean, 3),
                "overall_median": round(overall_median, 3),
                "total_groups": len(group_stats),
                "highest_group": group_stats[0]["group"] if group_stats else None,
                "lowest_group": group_stats[-1]["group"] if group_stats else None,
                "group_statistics": group_stats
            },
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            warnings=[f"{s['group']} has sample size < 5 ({s['count']})" for s in group_stats if s["count"] < 5],
            mutated_data=False,
            requires_approval=False
        )


class InvestigateAnomaliesTool(BaseTool):
    name = "investigate_anomalies"
    description = "Detect and isolate numerical anomalies in raw data using IQR or Z-score criteria."
    category = ToolCategory.ANALYSIS
    permission = ToolPermission.READ_ONLY
    input_schema = InvestigateAnomaliesInput

    def execute(self, params: InvestigateAnomaliesInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found."
            )

        if params.column not in df.columns:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Column '{params.column}' does not exist in dataset."
            )

        if not pd.api.types.is_numeric_dtype(df[params.column]):
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Anomaly detection requires a numerical column; '{params.column}' is non-numeric."
            )

        series = df[params.column].dropna()
        if len(series) < 4:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details="Insufficient non-null records (minimum 4 required for anomaly bounds)."
            )

        method = params.method.lower() if params.method else "iqr"

        if method == "zscore":
            mean = float(series.mean())
            std = float(series.std()) if len(series) > 1 else 0.0
            if std == 0:
                return ToolExecutionResult(
                    tool_name=self.name,
                    status=ToolExecutionStatus.SUCCESS.value,
                    result={"anomalies_found": 0, "outlier_indices": [], "method": "zscore", "notes": "Zero variance detected."},
                    dataset_id=params.dataset_id,
                    dataset_version=context.get("dataset_version", 0)
                )
            z_scores = (series - mean).abs() / std
            outliers = series[z_scores > 3.0]
            lower_bound = round(mean - 3.0 * std, 3)
            upper_bound = round(mean + 3.0 * std, 3)
        else:
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower_bound = round(q25 - 1.5 * iqr, 3)
            upper_bound = round(q75 + 1.5 * iqr, 3)
            outliers = series[(series < lower_bound) | (series > upper_bound)]

        sample_cases = [
            {"row_index": int(idx), "value": sanitize_val(val)}
            for idx, val in outliers.head(20).items()
        ]

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result={
                "dataset_id": params.dataset_id,
                "column": params.column,
                "method": method,
                "total_rows_inspected": len(series),
                "anomalies_found": len(outliers),
                "anomaly_percentage": round((len(outliers) / len(series)) * 100, 2),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "sample_outlier_cases": sample_cases
            },
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            warnings=[f"Identified {len(outliers)} statistical outlier records in '{params.column}'"] if len(outliers) > 0 else [],
            mutated_data=False,
            requires_approval=False
        )


class AnalyzeTimeSeriesInput(BaseModel):
    dataset_id: str = Field(description="Active session dataset identifier")
    time_column: str = Field(description="Date, datetime, or chronological timestamp column")
    value_column: str = Field(description="Numeric metric column to analyze over time")
    frequency: Optional[str] = Field(default=None, description="Optional resampling frequency: 'D' (daily), 'W' (weekly), 'M' (monthly), or None")


class AnalyzeTimeSeriesTool(BaseTool):
    name = "analyze_time_series"
    description = "Compute deterministic time-series trends, rate of change, moving averages, and linear trajectory over time."
    category = ToolCategory.ANALYSIS
    permission = ToolPermission.READ_ONLY
    input_schema = AnalyzeTimeSeriesInput

def extract_time_series_pair(df: pd.DataFrame, time_col: str, value_col: str) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[str]]:
    """Resolve time and numeric value series from DataFrame columns or MultiIndex/Index."""
    col_map = {str(c).lower(): c for c in df.columns}
    actual_val = value_col if value_col in df.columns else col_map.get(str(value_col).lower())
    if not actual_val:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            actual_val = num_cols[0]
        else:
            return None, None, f"Numeric value column '{value_col}' not found in dataset."

    val_s = df[actual_val]

    time_s = None
    if time_col in df.columns:
        time_s = df[time_col]
    elif str(time_col).lower() in col_map:
        time_s = df[col_map[str(time_col).lower()]]
    elif isinstance(df.index, pd.MultiIndex):
        level_0 = [x[0] if isinstance(x, tuple) else x for x in df.index]
        time_s = pd.Series(level_0, index=df.index)
    elif df.index.name and df.index.name.lower() == str(time_col).lower():
        time_s = pd.Series(df.index, index=df.index)
    elif str(time_col).lower() in ["year", "date", "time", "ds", "index"]:
        if isinstance(df.index, pd.MultiIndex):
            level_0 = [x[0] if isinstance(x, tuple) else x for x in df.index]
            time_s = pd.Series(level_0, index=df.index)
        else:
            time_s = pd.Series(df.index, index=df.index)
    else:
        return None, None, f"Time column '{time_col}' not found in dataset."

    return time_s, val_s, None


class AnalyzeTimeSeriesTool(BaseTool):
    name = "analyze_time_series"
    description = "Compute time-series trajectories, linear trends, annualized/decadal rates of change, and period comparisons."
    category = ToolCategory.ANALYSIS
    permission = ToolPermission.READ_ONLY
    input_schema = AnalyzeTimeSeriesInput

    def execute(self, params: AnalyzeTimeSeriesInput, context: Dict[str, Any]) -> ToolExecutionResult:
        from scipy import stats as sp_stats

        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found."
            )

        time_s, val_s, err = extract_time_series_pair(df, params.time_column, params.value_column)
        if err or time_s is None or val_s is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=err or f"Time column '{params.time_column}' not found in dataset."
            )

        if not pd.api.types.is_numeric_dtype(val_s):
            val_num = pd.to_numeric(val_s, errors="coerce")
            if val_num.notna().sum() < 3:
                return ToolExecutionResult(
                    tool_name=self.name,
                    status=ToolExecutionStatus.ERROR.value,
                    error_details=f"Value column '{params.value_column}' must be numeric for time-series analysis."
                )
            val_s = val_num

        # Check if time values are numeric integer years (e.g. 1880, 1881...)
        int_years = pd.to_numeric(time_s, errors="coerce")
        valid_int_mask = int_years.notna() & val_s.notna() & (int_years >= 1000) & (int_years <= 3000)
        is_year_series = bool(valid_int_mask.sum() >= 3 and (int_years[valid_int_mask] % 1 == 0).all())

        if is_year_series:
            clean_years = int_years[valid_int_mask].astype(int).values
            clean_vals = val_s[valid_int_mask].astype(float).values
            sort_order = np.argsort(clean_years)
            clean_years = clean_years[sort_order]
            clean_vals = clean_vals[sort_order]

            n = len(clean_years)
            start_year = int(clean_years[0])
            end_year = int(clean_years[-1])
            start_val = round(float(clean_vals[0]), 2)
            end_val = round(float(clean_vals[-1]), 2)

            lr_res = sp_stats.linregress(clean_years, clean_vals)
            slope_per_year = float(lr_res.slope)
            slope_per_decade = round(slope_per_year * 10.0, 4)
            slope_per_day = round(slope_per_year / 365.25, 6)
            intercept = float(lr_res.intercept)
            r_val = float(lr_res.rvalue)
            p_val = float(lr_res.pvalue)
            r_sq = float(r_val ** 2)
            is_significant = bool(p_val < 0.05)
            direction = "upward" if (is_significant and slope_per_year > 0) else ("downward" if (is_significant and slope_per_year < 0) else "stable")

            t_crit = sp_stats.t.ppf(0.975, n - 2) if n > 2 else 1.96
            ci_lower = round((slope_per_year - t_crit * lr_res.stderr) * 10.0, 4)
            ci_upper = round((slope_per_year + t_crit * lr_res.stderr) * 10.0, 4)

            abs_change = round(end_val - start_val, 3)
            pct_change = round(((end_val - start_val) / abs(start_val)) * 100.0, 2) if start_val != 0 else None
            rate_year = round(slope_per_year, 4)

            diffs = np.diff(clean_vals)
            max_c_idx = np.argmax(np.abs(diffs))
            largest_change = round(float(diffs[max_c_idx]), 2)
            largest_prev_year = int(clean_years[max_c_idx])
            largest_change_year = int(clean_years[max_c_idx + 1])

            rolling_s = pd.Series(clean_vals).rolling(window=5, min_periods=1).mean()
            latest_ma = round(float(rolling_s.iloc[-1]), 3)
            lag1_autocorr = round(float(pd.Series(clean_vals).autocorr(lag=1)), 3) if n >= 6 else None

            mid_idx = n // 2
            early_years, early_vals = clean_years[:mid_idx], clean_vals[:mid_idx]
            late_years, late_vals = clean_years[mid_idx:], clean_vals[mid_idx:]

            period_comp = {
                "early_half": {
                    "start_date": str(early_years[0]),
                    "end_date": str(early_years[-1]),
                    "mean": round(float(np.mean(early_vals)), 4),
                    "median": round(float(np.median(early_vals)), 4),
                    "std": round(float(np.std(early_vals, ddof=1)), 4),
                    "sample_size": len(early_vals),
                    "slope_per_day": round(float(sp_stats.linregress(early_years, early_vals).slope / 365.25), 6) if len(early_years) >= 2 else 0.0
                },
                "late_half": {
                    "start_date": str(late_years[0]),
                    "end_date": str(late_years[-1]),
                    "mean": round(float(np.mean(late_vals)), 4),
                    "median": round(float(np.median(late_vals)), 4),
                    "std": round(float(np.std(late_vals, ddof=1)), 4),
                    "sample_size": len(late_vals),
                    "slope_per_day": round(float(sp_stats.linregress(late_years, late_vals).slope / 365.25), 6) if len(late_years) >= 2 else 0.0
                },
                "early_period": {
                    "start": str(early_years[0]),
                    "end": str(early_years[-1]),
                    "mean": round(float(np.mean(early_vals)), 4),
                    "sample_size": len(early_years)
                },
                "late_period": {
                    "start": str(late_years[0]),
                    "end": str(late_years[-1]),
                    "mean": round(float(np.mean(late_vals)), 4),
                    "sample_size": len(late_years)
                },
                "mean_difference": round(float(np.mean(late_vals) - np.mean(early_vals)), 4),
                "delta_mean": round(float(np.mean(late_vals) - np.mean(early_vals)), 4),
                "difference_in_means": round(float(np.mean(late_vals) - np.mean(early_vals)), 4),
                "trend_acceleration": bool(np.mean(late_vals) > np.mean(early_vals)),
                "acceleration_status": "accelerating" if np.mean(late_vals) > np.mean(early_vals) else "steady"
            }

            steepest_segment = {
                "start_date": str(largest_prev_year),
                "end_date": str(largest_change_year),
                "days": 365.25,
                "change": largest_change,
                "slope_per_day": round(largest_change / 365.25, 4)
            }

            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.SUCCESS.value,
                result={
                    "dataset_id": params.dataset_id,
                    "time_column": params.time_column,
                    "value_column": params.value_column,
                    "sample_size": n,
                    "years_used": n,
                    "start_date": str(start_year),
                    "end_date": str(end_year),
                    "start_year": start_year,
                    "end_year": end_year,
                    "time_span_days": float((end_year - start_year) * 365.25),
                    "linear_trend": {
                        "slope_per_day": slope_per_day,
                        "slope_per_decade": slope_per_decade,
                        "confidence_interval_95": [ci_lower, ci_upper],
                        "intercept": round(intercept, 4),
                        "r_squared": round(r_sq, 4),
                        "p_value": round(p_val, 6),
                        "direction": direction,
                        "statistically_significant": is_significant
                    },
                    "rate_of_change": {
                        "start_value": start_val,
                        "end_value": end_val,
                        "absolute_change": abs_change,
                        "percentage_change": pct_change,
                        "rate_per_year": rate_year,
                        "rate_per_decade": slope_per_decade,
                        "confidence_interval_95": [ci_lower, ci_upper],
                        "largest_year_to_year_change": largest_change,
                        "largest_change_from": largest_prev_year,
                        "largest_change_to": largest_change_year
                    },
                    "period_comparison": period_comp,
                    "steepest_segment": steepest_segment,
                    "summary_statistics": {
                        "mean": round(float(np.mean(clean_vals)), 4),
                        "std": round(float(np.std(clean_vals, ddof=1)), 4),
                        "min": round(float(np.min(clean_vals)), 4),
                        "max": round(float(np.max(clean_vals)), 4)
                    },
                    "moving_average": {
                        "window_size": 5,
                        "latest_moving_average": latest_ma
                    },
                    "lag1_autocorrelation": lag1_autocorr
                },
                dataset_id=params.dataset_id,
                dataset_version=context.get("dataset_version", 0),
                warnings=[],
                mutated_data=False,
                requires_approval=False
            )

        # Attempt to parse time column into standard datetime
        parsed_dates = pd.to_datetime(time_s, errors="coerce")
        valid_mask = parsed_dates.notna() & val_s.notna()
        valid_count = int(valid_mask.sum())

        if valid_count < 3:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=(
                    f"Column '{params.time_column}' contains fewer than 3 valid datetime values. "
                    "Time-series trend analysis requires a chronological date or timestamp variable, "
                    "not unordered categorical data."
                )
            )

        # Construct clean sorted time-series DataFrame
        ts_df = pd.DataFrame({
            "ds": parsed_dates[valid_mask],
            "y": val_s[valid_mask].astype(float)
        }).sort_values("ds").reset_index(drop=True)

        n = len(ts_df)
        start_dt = ts_df["ds"].min()
        end_dt = ts_df["ds"].max()
        span_seconds = (end_dt - start_dt).total_seconds()
        span_days = span_seconds / 86400.0

        # Calculate linear trend via OLS
        x_days = (ts_df["ds"] - start_dt).dt.total_seconds() / 86400.0
        if span_days > 0 and len(x_days.unique()) > 1:
            lr_res = sp_stats.linregress(x_days, ts_df["y"])
            slope = float(lr_res.slope)
            intercept = float(lr_res.intercept)
            r_val = float(lr_res.rvalue)
            p_val = float(lr_res.pvalue)
            r_sq = float(r_val ** 2)
            is_significant = bool(p_val < 0.05)
            if is_significant and slope > 0:
                direction = "upward"
            elif is_significant and slope < 0:
                direction = "downward"
            else:
                direction = "stable / no statistically significant linear trend"
        else:
            slope, intercept, r_val, p_val, r_sq = 0.0, float(ts_df["y"].mean()), 0.0, 1.0, 0.0
            direction = "insufficient time variance to determine linear trend"
            is_significant = False

        # Rate of change
        y_start = float(ts_df["y"].iloc[0])
        y_end = float(ts_df["y"].iloc[-1])
        abs_change = round(y_end - y_start, 3)
        pct_change = round(((y_end - y_start) / abs(y_start)) * 100.0, 2) if y_start != 0 else None
        rate_year = round(slope * 365.25, 4)
        rate_decade = round(slope * 3652.5, 4)

        # Rolling moving average (window 3 or 7)
        win = 3 if n < 14 else 7
        rolling_series = ts_df["y"].rolling(window=win, min_periods=1).mean()
        latest_ma = round(float(rolling_series.iloc[-1]), 3)

        # Autocorrelation / Seasonality check if sufficient points
        lag1_autocorr = None
        if n >= 6:
            try:
                lag1_autocorr = round(float(ts_df["y"].autocorr(lag=1)), 3)
            except Exception:
                lag1_autocorr = None

        # Period comparison: Split chronologically into early and late halves
        mid_idx = max(1, n // 2)
        early_df = ts_df.iloc[:mid_idx]
        late_df = ts_df.iloc[mid_idx:]

        early_slope = 0.0
        late_slope = 0.0
        if len(early_df) >= 2:
            early_days = (early_df["ds"] - early_df["ds"].min()).dt.total_seconds() / 86400.0
            if early_days.max() > 0:
                early_slope = float(sp_stats.linregress(early_days, early_df["y"]).slope)

        if len(late_df) >= 2:
            late_days = (late_df["ds"] - late_df["ds"].min()).dt.total_seconds() / 86400.0
            if late_days.max() > 0:
                late_slope = float(sp_stats.linregress(late_days, late_df["y"]).slope)

        early_mean = round(float(early_df["y"].mean()), 3) if len(early_df) > 0 else 0.0
        late_mean = round(float(late_df["y"].mean()), 3) if len(late_df) > 0 else 0.0
        early_median = round(float(early_df["y"].median()), 3) if len(early_df) > 0 else 0.0
        late_median = round(float(late_df["y"].median()), 3) if len(late_df) > 0 else 0.0
        early_std = round(float(early_df["y"].std()), 3) if len(early_df) > 1 else 0.0
        late_std = round(float(late_df["y"].std()), 3) if len(late_df) > 1 else 0.0
        period_comp = {
            "early_period": {
                "start": str(early_df["ds"].min()) if len(early_df) > 0 else "",
                "end": str(early_df["ds"].max()) if len(early_df) > 0 else "",
                "sample_size": len(early_df),
                "mean": early_mean,
                "median": early_median,
                "std": early_std,
                "slope_per_day": round(early_slope, 6)
            },
            "late_period": {
                "start": str(late_df["ds"].min()) if len(late_df) > 0 else "",
                "end": str(late_df["ds"].max()) if len(late_df) > 0 else "",
                "sample_size": len(late_df),
                "mean": late_mean,
                "median": late_median,
                "std": late_std,
                "slope_per_day": round(late_slope, 6)
            },
            "early_half": {
                "start_date": str(early_df["ds"].min()) if len(early_df) > 0 else "",
                "end_date": str(early_df["ds"].max()) if len(early_df) > 0 else "",
                "mean": early_mean,
                "median": early_median,
                "std": early_std,
                "sample_size": len(early_df)
            },
            "late_half": {
                "start_date": str(late_df["ds"].min()) if len(late_df) > 0 else "",
                "end_date": str(late_df["ds"].max()) if len(late_df) > 0 else "",
                "mean": late_mean,
                "median": late_median,
                "std": late_std,
                "sample_size": len(late_df)
            },
            "difference_in_means": round(late_mean - early_mean, 3),
            "delta_mean": round(late_mean - early_mean, 3),
            "slope_change": round(late_slope - early_slope, 6),
            "acceleration_status": "accelerating" if late_slope > early_slope else ("decelerating" if late_slope < early_slope else "constant")
        }

        # Steepest segment detection
        steepest_segment = None
        if n >= 2:
            max_abs_slope = -1.0
            for i in range(len(ts_df) - 1):
                dt_days = (ts_df["ds"].iloc[i+1] - ts_df["ds"].iloc[i]).total_seconds() / 86400.0
                if dt_days > 0:
                    dy = ts_df["y"].iloc[i+1] - ts_df["y"].iloc[i]
                    cur_slope = dy / dt_days
                    if abs(cur_slope) > max_abs_slope:
                        max_abs_slope = abs(cur_slope)
                        steepest_segment = {
                            "start_date": str(ts_df["ds"].iloc[i]),
                            "end_date": str(ts_df["ds"].iloc[i+1]),
                            "days": round(dt_days, 2),
                            "change": round(float(dy), 3),
                            "slope_per_day": round(float(cur_slope), 4)
                        }

        warnings = []
        if n < 10:
            warnings.append(f"Small sample size (N={n}) makes trend projections sensitive to individual points.")
        if span_days == 0:
            warnings.append("All timestamps occur on the same date/time; cannot evaluate temporal progression.")

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result={
                "dataset_id": params.dataset_id,
                "time_column": params.time_column,
                "value_column": params.value_column,
                "sample_size": n,
                "years_used": n,
                "start_date": str(start_dt),
                "end_date": str(end_dt),
                "time_span_days": round(span_days, 2),
                "linear_trend": {
                    "slope_per_day": round(slope, 6),
                    "slope_per_decade": rate_decade,
                    "intercept": round(intercept, 4),
                    "r_squared": round(r_sq, 4),
                    "p_value": round(p_val, 6),
                    "direction": direction,
                    "statistically_significant": is_significant
                },
                "rate_of_change": {
                    "start_value": round(y_start, 3),
                    "end_value": round(y_end, 3),
                    "absolute_change": abs_change,
                    "percentage_change": pct_change,
                    "rate_per_year": rate_year,
                    "rate_per_decade": rate_decade
                },
                "period_comparison": period_comp,
                "steepest_segment": steepest_segment,
                "summary_statistics": {
                    "mean": round(float(ts_df["y"].mean()), 3),
                    "std": round(float(ts_df["y"].std()), 3) if n > 1 else 0.0,
                    "min": round(float(ts_df["y"].min()), 3),
                    "max": round(float(ts_df["y"].max()), 3)
                },
                "moving_average": {
                    "window_size": win,
                    "latest_moving_average": latest_ma
                },
                "lag1_autocorrelation": lag1_autocorr
            },
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            warnings=warnings,
            mutated_data=False,
            requires_approval=False
        )



# ==========================================
# Focused Rate of Change & Period Comparison Tools
# ==========================================

class ComputeRateOfChangeInput(BaseModel):
    time_column: str = Field(description="Name of the chronological date/time column")
    value_column: str = Field(description="Name of the numerical metric column")
    dataset_id: Optional[str] = Field(default=None, description="Dataset session identifier")
    unit: Optional[str] = Field(default="day", description="Unit for rate computation: 'day', 'year', 'decade'")


class ComputeRateOfChangeOutput(BaseModel):
    slope: float
    rate_per_day: float
    rate_per_year: float
    rate_per_decade: float
    direction: str
    p_value: float
    is_significant: bool
    steepest_segment: Optional[Dict[str, Any]] = None
    plain_english_finding: str


class ComputeRateOfChangeTool(BaseTool):
    """Calculates granular rate of change, linear slope, annualized/decadal trends, and steepest interval."""
    name: str = "compute_rate_of_change"
    description: str = "Computes rate of change per day, year, and decade, identifying linear trend slope and steepest acceleration interval."
    category: ToolCategory = ToolCategory.ANALYSIS
    permission: ToolPermission = ToolPermission.READ_ONLY
    input_schema: Type[BaseModel] = ComputeRateOfChangeInput
    output_schema: Optional[Type[BaseModel]] = ComputeRateOfChangeOutput
    timeout_seconds: float = 30.0

    def execute(self, params: ComputeRateOfChangeInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details="No active dataset available in execution context."
            )

        # Delegate core time-series math to AnalyzeTimeSeriesTool logic
        ts_tool = AnalyzeTimeSeriesTool()
        ts_res = ts_tool.execute(
            AnalyzeTimeSeriesInput(
                time_column=params.time_column,
                value_column=params.value_column,
                dataset_id=params.dataset_id
            ),
            context
        )

        if ts_res.status == ToolExecutionStatus.ERROR.value or not ts_res.result:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=ts_res.error_details or "Rate of change calculation failed."
            )

        res_data = ts_res.result
        lt = res_data.get("linear_trend", {})
        roc = res_data.get("rate_of_change", {})
        steepest = res_data.get("steepest_segment")

        slope_day = float(lt.get("slope_per_day", 0.0))
        rate_yr = float(roc.get("rate_per_year", slope_day * 365.25))
        rate_dec = float(roc.get("rate_per_decade", slope_day * 3652.5))
        direction = str(lt.get("direction", "flat"))
        p_val = float(lt.get("p_value", 1.0))
        is_sig = bool(lt.get("statistically_significant", False))

        finding = (
            f"The rate of change for '{params.value_column}' is {slope_day:+.4f} units/day "
            f"({rate_yr:+.2f} units/year, {rate_dec:+.2f} units/decade). "
            f"The trend is {direction} (p = {p_val:.4f}, {'statistically significant' if is_sig else 'not statistically significant'})."
        )
        if steepest:
            finding += f" The steepest rate of change occurred between {steepest.get('start_date')} and {steepest.get('end_date')} ({steepest.get('slope_per_day'):+.4f}/day)."

        output_payload = ComputeRateOfChangeOutput(
            slope=slope_day,
            rate_per_day=slope_day,
            rate_per_year=rate_yr,
            rate_per_decade=rate_dec,
            direction=direction,
            p_value=p_val,
            is_significant=is_sig,
            steepest_segment=steepest,
            plain_english_finding=finding
        )

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result=output_payload.model_dump() if hasattr(output_payload, "model_dump") else output_payload.dict(),
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            mutated_data=False,
            requires_approval=False
        )


class ComputePeriodComparisonInput(BaseModel):
    time_column: str = Field(description="Name of chronological datetime column")
    value_column: str = Field(description="Name of numerical metric column")
    dataset_id: Optional[str] = Field(default=None, description="Dataset session identifier")
    split_ratio: Optional[float] = Field(default=0.5, description="Fraction of observations allocated to early period")


class ComputePeriodComparisonOutput(BaseModel):
    early_period: Dict[str, Any]
    late_period: Dict[str, Any]
    delta_mean: float
    percent_change: float
    is_accelerating: bool
    plain_english_finding: str


class ComputePeriodComparisonTool(BaseTool):
    """Compares metrics between early and late chronological halves of a time series."""
    name: str = "compute_period_comparison"
    description: str = "Compares means, slopes, and growth rates between earlier and later chronological intervals."
    category: ToolCategory = ToolCategory.ANALYSIS
    permission: ToolPermission = ToolPermission.READ_ONLY
    input_schema: Type[BaseModel] = ComputePeriodComparisonInput
    output_schema: Optional[Type[BaseModel]] = ComputePeriodComparisonOutput
    timeout_seconds: float = 30.0

    def execute(self, params: ComputePeriodComparisonInput, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details="No active dataset available in execution context."
            )

        ts_tool = AnalyzeTimeSeriesTool()
        ts_res = ts_tool.execute(
            AnalyzeTimeSeriesInput(
                time_column=params.time_column,
                value_column=params.value_column,
                dataset_id=params.dataset_id
            ),
            context
        )

        if ts_res.status == ToolExecutionStatus.ERROR.value or not ts_res.result:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=ts_res.error_details or "Period comparison calculation failed."
            )

        period_comp = ts_res.result.get("period_comparison")
        if not period_comp:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details="Insufficient chronological records to partition into early and late periods."
            )

        early = period_comp.get("early_half", {})
        late = period_comp.get("late_half", {})
        delta = float(period_comp.get("mean_difference", 0.0))
        is_accel = bool(period_comp.get("trend_acceleration", False))
        early_mean = float(early.get("mean", 0.0))
        pct_change = round(((delta / early_mean) * 100.0), 2) if early_mean != 0 else 0.0

        finding = (
            f"Comparing the Early Period ({early.get('start_date')} to {early.get('end_date')}, mean = {early.get('mean')}) "
            f"to the Late Period ({late.get('start_date')} to {late.get('end_date')}, mean = {late.get('mean')}): "
            f"Mean metric delta is {delta:+.2f} ({pct_change:+.1f}%). "
            f"Trajectory is {'accelerating' if is_accel else 'decelerating or steady'}."
        )

        output_payload = ComputePeriodComparisonOutput(
            early_period=early,
            late_period=late,
            delta_mean=delta,
            percent_change=pct_change,
            is_accelerating=is_accel,
            plain_english_finding=finding
        )

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result=output_payload.model_dump() if hasattr(output_payload, "model_dump") else output_payload.dict(),
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            mutated_data=False,
            requires_approval=False
        )


