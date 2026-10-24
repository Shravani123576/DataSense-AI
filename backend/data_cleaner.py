import uuid
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from models import CleaningRecommendation, CleaningActionRequest, CleaningAuditRecord


def generate_cleaning_recommendations(df: pd.DataFrame) -> List[CleaningRecommendation]:
    """
    Examine DataFrame anomalies and return evidence-backed cleaning recommendations.
    Every recommendation specifies exact affected row count, percentage, rule, and strategies.
    """
    recommendations: List[CleaningRecommendation] = []
    total_rows = len(df)
    if total_rows == 0:
        return recommendations

    # 1. Duplicate Rows Check
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = round((dup_count / total_rows) * 100, 2)
        recommendations.append(CleaningRecommendation(
            id=f"rec-dup-{uuid.uuid4().hex[:6]}",
            title="Remove Exact Duplicate Rows",
            column=None,
            rule="deduplicate",
            severity="high" if dup_pct > 5 else "medium",
            evidence=f"Detected {dup_count:,} duplicate record(s) ({dup_pct}% of dataset). Exact duplicate records inflate summary statistics and bias model training.",
            affected_rows=dup_count,
            affected_percentage=dup_pct,
            recommended_action="Deduplicate keeping the first occurrence of each unique record.",
            strategy_options=["keep_first", "drop_all_duplicates"]
        ))

    # 2. Missing Values Check by Column
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            null_pct = round((null_count / total_rows) * 100, 2)
            is_num = pd.api.types.is_numeric_dtype(df[col])
            
            if is_num:
                median_val = round(float(df[col].dropna().median()), 2)
                mean_val = round(float(df[col].dropna().mean()), 2)
                recommendations.append(CleaningRecommendation(
                    id=f"rec-null-{col.lower()[:8]}-{uuid.uuid4().hex[:4]}",
                    title=f"Impute Missing Values in '{col}'",
                    column=col,
                    rule="impute_missing",
                    severity="high" if null_pct > 20 else "medium",
                    evidence=f"Column has {null_count:,} missing value(s) ({null_pct}%). Median is {median_val:,}; Mean is {mean_val:,}.",
                    affected_rows=null_count,
                    affected_percentage=null_pct,
                    recommended_action="Impute missing numeric values using the column median to preserve robust distribution.",
                    strategy_options=["median", "mean", "zero", "drop_rows"]
                ))
            else:
                top_mode = df[col].dropna().mode()
                mode_val = str(top_mode.iloc[0]) if len(top_mode) > 0 else "Unknown"
                recommendations.append(CleaningRecommendation(
                    id=f"rec-null-{col.lower()[:8]}-{uuid.uuid4().hex[:4]}",
                    title=f"Impute Missing Categories in '{col}'",
                    column=col,
                    rule="impute_missing",
                    severity="high" if null_pct > 20 else "low",
                    evidence=f"Column has {null_count:,} missing value(s) ({null_pct}%). Most frequent category is '{mode_val}'.",
                    affected_rows=null_count,
                    affected_percentage=null_pct,
                    recommended_action=f"Impute missing categorical cells with '{mode_val}' or label as 'Unknown'.",
                    strategy_options=["mode", "fill_unknown", "drop_rows"]
                ))

    # 3. Inconsistent Category Normalization Check (whitespace / mixed casing)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
            non_null = df[col].dropna().astype(str)
            if len(non_null) > 0:
                has_whitespace = non_null.str.strip().ne(non_null).any()
                # Check casing differences: e.g. 'Yes' vs 'yes'
                unique_raw = set(non_null.unique())
                unique_lower = set(non_null.str.lower().unique())
                has_casing_conflict = len(unique_raw) > len(unique_lower)

                if has_whitespace or has_casing_conflict:
                    affected = int((non_null.str.strip() != non_null).sum() if has_whitespace else len(non_null))
                    aff_pct = round((affected / total_rows) * 100, 2)
                    recommendations.append(CleaningRecommendation(
                        id=f"rec-norm-{col.lower()[:8]}-{uuid.uuid4().hex[:4]}",
                        title=f"Normalize String Formatting in '{col}'",
                        column=col,
                        rule="normalize_categories",
                        severity="low",
                        evidence=f"Found {'whitespace padding' if has_whitespace else ''}{' and ' if has_whitespace and has_casing_conflict else ''}{'casing inconsistency' if has_casing_conflict else ''} in {len(unique_raw)} raw categories.",
                        affected_rows=affected,
                        affected_percentage=aff_pct,
                        recommended_action="Trim leading/trailing whitespace and normalize text to clean Title Case.",
                        strategy_options=["strip_and_titlecase", "lowercase", "strip_only"]
                    ))

    # 4. Outlier Capping Check (1.5x IQR Rule)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_series = df[col].dropna()
            if len(num_series) > 10:
                q25 = num_series.quantile(0.25)
                q75 = num_series.quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outliers = num_series[(num_series < lower_bound) | (num_series > upper_bound)]
                    outlier_count = len(outliers)
                    if outlier_count > 0:
                        out_pct = round((outlier_count / len(num_series)) * 100, 2)
                        recommendations.append(CleaningRecommendation(
                            id=f"rec-out-{col.lower()[:8]}-{uuid.uuid4().hex[:4]}",
                            title=f"Cap Statistical Outliers in '{col}'",
                            column=col,
                            rule="cap_outliers",
                            severity="medium" if out_pct > 2 else "low",
                            evidence=f"Detected {outlier_count:,} statistical outlier(s) ({out_pct}%) outside IQR bounds [{round(lower_bound, 2)}, {round(upper_bound, 2)}].",
                            affected_rows=outlier_count,
                            affected_percentage=out_pct,
                            recommended_action=f"Winsorize outliers to upper limit ({round(upper_bound, 2)}) and lower limit ({round(lower_bound, 2)}).",
                            strategy_options=["cap_iqr", "drop_outliers"]
                        ))

    return recommendations


def preview_cleaning_action(df: pd.DataFrame, req: CleaningActionRequest) -> Dict[str, Any]:
    """
    Generate an uncommitted preview of a proposed cleaning action with diff sample.
    """
    rule = req.rule
    col = req.column
    strategy = req.strategy or "median"

    if rule == "deduplicate":
        dup_mask = df.duplicated()
        affected = int(dup_mask.sum())
        sample_diff = df[dup_mask].head(5).to_dict(orient="records")
        return {
            "rule": rule,
            "column": None,
            "strategy": strategy,
            "affected_rows": affected,
            "affected_percentage": round(affected / len(df) * 100, 2) if len(df) > 0 else 0.0,
            "summary": f"Will remove {affected:,} duplicate rows from the dataset.",
            "preview_diff": sample_diff
        }

    if not col or col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")

    series = df[col]
    if rule == "impute_missing":
        null_mask = series.isna()
        affected = int(null_mask.sum())
        if affected == 0:
            return {"affected_rows": 0, "summary": "No missing values present."}

        fill_val = None
        if strategy == "median" and pd.api.types.is_numeric_dtype(series):
            fill_val = round(float(series.dropna().median()), 2)
        elif strategy == "mean" and pd.api.types.is_numeric_dtype(series):
            fill_val = round(float(series.dropna().mean()), 2)
        elif strategy == "zero":
            fill_val = 0
        elif strategy == "mode":
            mode_s = series.dropna().mode()
            fill_val = str(mode_s.iloc[0]) if len(mode_s) > 0 else "Unknown"
        elif strategy == "fill_unknown":
            fill_val = "Unknown"
        elif strategy == "drop_rows":
            fill_val = "DROP_ROWS"

        return {
            "rule": rule,
            "column": col,
            "strategy": strategy,
            "fill_value": fill_val,
            "affected_rows": affected,
            "affected_percentage": round(affected / len(df) * 100, 2),
            "summary": f"Will fill {affected:,} null cell(s) in '{col}' with {fill_val}."
        }

    if rule == "normalize_categories":
        non_null = series.dropna().astype(str)
        affected = int(non_null.str.strip().ne(non_null).sum())
        return {
            "rule": rule,
            "column": col,
            "strategy": strategy,
            "affected_rows": affected,
            "summary": f"Will normalize formatting and remove whitespace padding in '{col}' across {affected:,} cells."
        }

    if rule == "cap_outliers":
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"Column '{col}' is not numeric.")
        num_s = series.dropna()
        q25 = num_s.quantile(0.25)
        q75 = num_s.quantile(0.75)
        iqr = q75 - q25
        lower_bound = round(q25 - 1.5 * iqr, 2)
        upper_bound = round(q75 + 1.5 * iqr, 2)
        outliers = num_s[(num_s < lower_bound) | (num_s > upper_bound)]
        affected = len(outliers)
        return {
            "rule": rule,
            "column": col,
            "strategy": strategy,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "affected_rows": affected,
            "summary": f"Will cap {affected:,} outlier value(s) in '{col}' outside [{lower_bound}, {upper_bound}]."
        }

    return {"summary": "Preview generated."}


def apply_cleaning_action(df: pd.DataFrame, req: CleaningActionRequest) -> Tuple[pd.DataFrame, CleaningAuditRecord]:
    """
    Execute an approved cleaning action on a copy of df, returning the cleaned df and an audit record.
    """
    clean_df = df.copy()
    rule = req.rule
    col = req.column
    strategy = req.strategy or "median"
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 1. Deduplication
    if rule == "deduplicate":
        dup_mask = clean_df.duplicated()
        affected = int(dup_mask.sum())
        clean_df = clean_df.drop_duplicates(keep="first" if strategy != "drop_all_duplicates" else False).reset_index(drop=True)
        
        audit = CleaningAuditRecord(
            id=f"audit-dup-{uuid.uuid4().hex[:6]}",
            timestamp=now_str,
            rule="deduplicate",
            column=None,
            affected_rows=affected,
            affected_percentage=round(affected / len(df) * 100, 2) if len(df) > 0 else 0.0,
            strategy_used=strategy,
            transformation_summary=f"Deduplicated dataset: removed {affected:,} exact duplicate row(s).",
            original_sample=[],
            cleaned_sample=[],
            reason="Eliminates redundant records to prevent inflated sums, counts, and biased model training."
        )
        return clean_df, audit

    # Column-level validation
    if not col or col not in clean_df.columns:
        raise ValueError(f"Column '{col}' does not exist in dataset.")

    orig_series = clean_df[col]
    
    # 2. Impute Missing Values
    if rule == "impute_missing":
        null_mask = orig_series.isna()
        affected = int(null_mask.sum())
        if affected == 0:
            audit = CleaningAuditRecord(
                id=f"audit-noop-{uuid.uuid4().hex[:6]}",
                timestamp=now_str,
                rule=rule,
                column=col,
                affected_rows=0,
                affected_percentage=0.0,
                strategy_used=strategy,
                transformation_summary=f"No missing values in '{col}'. Operation skipped.",
                reason="Verification"
            )
            return clean_df, audit

        fill_val = None
        if strategy == "drop_rows":
            clean_df = clean_df.dropna(subset=[col]).reset_index(drop=True)
            summary = f"Dropped {affected:,} row(s) containing missing values in '{col}'."
        else:
            if strategy == "median" and pd.api.types.is_numeric_dtype(orig_series):
                fill_val = round(float(orig_series.dropna().median()), 2)
            elif strategy == "mean" and pd.api.types.is_numeric_dtype(orig_series):
                fill_val = round(float(orig_series.dropna().mean()), 2)
            elif strategy == "zero":
                fill_val = 0.0 if pd.api.types.is_numeric_dtype(orig_series) else "0"
            elif strategy == "mode":
                mode_s = orig_series.dropna().mode()
                fill_val = mode_s.iloc[0] if len(mode_s) > 0 else "Unknown"
            else:  # fill_unknown
                fill_val = "Unknown"

            clean_df[col] = orig_series.fillna(fill_val)
            summary = f"Imputed {affected:,} missing cell(s) in '{col}' with {fill_val} ({strategy})."

        audit = CleaningAuditRecord(
            id=f"audit-impute-{uuid.uuid4().hex[:6]}",
            timestamp=now_str,
            rule=rule,
            column=col,
            affected_rows=affected,
            affected_percentage=round(affected / len(df) * 100, 2),
            strategy_used=strategy,
            transformation_summary=summary,
            original_sample=[None for _ in range(min(affected, 3))],
            cleaned_sample=[fill_val for _ in range(min(affected, 3))],
            reason=f"Imputation prevents loss of observation rows while maintaining valid input shapes for algorithms."
        )
        return clean_df, audit

    # 3. Normalize Categories
    if rule == "normalize_categories":
        non_null = orig_series.dropna().astype(str)
        affected = int(non_null.str.strip().ne(non_null).sum())
        
        if strategy == "lowercase":
            clean_df[col] = orig_series.astype(str).str.strip().str.lower()
        elif strategy == "strip_only":
            clean_df[col] = orig_series.astype(str).str.strip()
        else:  # strip_and_titlecase
            clean_df[col] = orig_series.astype(str).str.strip().str.title()

        orig_sample = non_null.head(3).tolist()
        cleaned_sample = clean_df[col].dropna().head(3).tolist()

        audit = CleaningAuditRecord(
            id=f"audit-norm-{uuid.uuid4().hex[:6]}",
            timestamp=now_str,
            rule=rule,
            column=col,
            affected_rows=affected,
            affected_percentage=round(affected / len(df) * 100, 2),
            strategy_used=strategy,
            transformation_summary=f"Normalized '{col}' formatting using {strategy}.",
            original_sample=orig_sample,
            cleaned_sample=cleaned_sample,
            reason="Harmonizes casing and removes padding so identical categories group and encode cleanly."
        )
        return clean_df, audit

    # 4. Cap Outliers (1.5x IQR Rule)
    if rule == "cap_outliers":
        if not pd.api.types.is_numeric_dtype(orig_series):
            raise ValueError(f"Column '{col}' is not numeric.")

        num_s = orig_series.dropna()
        q25 = num_s.quantile(0.25)
        q75 = num_s.quantile(0.75)
        iqr = q75 - q25
        lower_bound = round(q25 - 1.5 * iqr, 3)
        upper_bound = round(q75 + 1.5 * iqr, 3)

        outlier_mask = (orig_series < lower_bound) | (orig_series > upper_bound)
        affected = int(outlier_mask.sum())

        if strategy == "drop_outliers":
            clean_df = clean_df[~outlier_mask].reset_index(drop=True)
            summary = f"Dropped {affected:,} outlier row(s) in '{col}' beyond [{lower_bound}, {upper_bound}]."
        else:  # cap_iqr
            clean_df[col] = orig_series.clip(lower=lower_bound, upper=upper_bound)
            summary = f"Winsorized/capped {affected:,} outlier(s) in '{col}' to bounds [{lower_bound}, {upper_bound}]."

        audit = CleaningAuditRecord(
            id=f"audit-out-{uuid.uuid4().hex[:6]}",
            timestamp=now_str,
            rule=rule,
            column=col,
            affected_rows=affected,
            affected_percentage=round(affected / len(df) * 100, 2),
            strategy_used=strategy,
            transformation_summary=summary,
            original_sample=orig_series[outlier_mask].head(3).tolist(),
            cleaned_sample=[upper_bound if v > upper_bound else lower_bound for v in orig_series[outlier_mask].head(3)],
            reason="Protects linear models, regressions, and means from extreme statistical leverage."
        )
        return clean_df, audit

    raise ValueError(f"Unsupported cleaning rule '{rule}'.")
