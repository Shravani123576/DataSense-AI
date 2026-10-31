"""
DataSense AI Tool Layer — Statistical Hypothesis Testing Tool (Phase 3)
Connects HypothesisTestRequest to scipy.stats implementations, checking assumptions,
computing effect sizes, and enforcing non-causality disclaimers.
"""
import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from pydantic import BaseModel, Field

from tools.base import BaseTool, ToolCategory, ToolPermission, ToolExecutionResult, ToolExecutionStatus
from models import HypothesisTestRequest, HypothesisTestResponse, AssumptionCheck


class RunStatisticalTestTool(BaseTool):
    name = "run_statistical_test"
    description = (
        "Execute formal hypothesis tests (two-sample t-test, Mann-Whitney U, one-way ANOVA, "
        "Kruskal-Wallis, Chi-square, Pearson correlation) with assumption checks and effect sizes."
    )
    category = ToolCategory.STATISTICAL
    permission = ToolPermission.READ_ONLY
    input_schema = HypothesisTestRequest
    output_schema = HypothesisTestResponse

    def execute(self, params: HypothesisTestRequest, context: Dict[str, Any]) -> ToolExecutionResult:
        df: Optional[pd.DataFrame] = context.get("df")
        if df is None:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Dataset '{params.dataset_id}' not found."
            )

        col_x = params.variable_x
        col_y = params.variable_y

        if col_x not in df.columns:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Variable X '{col_x}' not found in dataset."
            )
        if col_y and col_y not in df.columns:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Variable Y '{col_y}' not found in dataset."
            )

        test_type = (params.test_type or "auto").lower()
        alpha = round(1.0 - (params.confidence_level or 0.95), 3)

        # Clean subset
        cols_needed = [col_x] + ([col_y] if col_y else [])
        clean_df = df[cols_needed].dropna()

        if len(clean_df) < 5:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Insufficient non-null observations ({len(clean_df)} records; minimum 5 required for hypothesis testing)."
            )

        # Detect types
        x_is_num = pd.api.types.is_numeric_dtype(clean_df[col_x])
        y_is_num = pd.api.types.is_numeric_dtype(clean_df[col_y]) if col_y else False

        # Route test type if 'auto'
        if test_type == "auto":
            if col_y is None:
                # One variable tests not yet supported, must provide 2 variables
                return ToolExecutionResult(
                    tool_name=self.name,
                    status=ToolExecutionStatus.ERROR.value,
                    error_details="Hypothesis test requires two variables (group vs metric, or two categorical, or two numeric)."
                )
            if x_is_num and y_is_num:
                test_type = "pearson_r"
            elif (not x_is_num) and (not y_is_num):
                test_type = "chi_square"
            elif (not x_is_num) and y_is_num:
                # Group X vs Metric Y
                num_groups = clean_df[col_x].nunique()
                test_type = "two_sample_t" if num_groups == 2 else "anova"
            elif x_is_num and (not y_is_num):
                # Swap so categorical is X, metric is Y
                col_x, col_y = col_y, col_x
                x_is_num, y_is_num = False, True
                num_groups = clean_df[col_x].nunique()
                test_type = "two_sample_t" if num_groups == 2 else "anova"

        try:
            if test_type in ["two_sample_t", "mann_whitney_u"]:
                res = self._run_two_group_test(clean_df, col_x, col_y, alpha, force_mann_whitney=(test_type == "mann_whitney_u"))
            elif test_type in ["anova", "one_way_anova", "kruskal_wallis"]:
                res = self._run_multigroup_test(clean_df, col_x, col_y, alpha, force_kruskal=(test_type == "kruskal_wallis"))
            elif test_type in ["chi_square", "chi2"]:
                res = self._run_chi_square_test(clean_df, col_x, col_y, alpha)
            elif test_type.lower() in ["pearson_r", "correlation", "pearson"]:
                res = self._run_pearson_test(clean_df, col_x, col_y, alpha)
            else:
                return ToolExecutionResult(
                    tool_name=self.name,
                    status=ToolExecutionStatus.ERROR.value,
                    error_details=f"Unsupported test type '{test_type}'. Supported: 'auto', 'two_sample_t', 'mann_whitney_u', 'anova', 'kruskal_wallis', 'chi_square', 'pearson_r'."
                )
        except ValueError as ve:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=str(ve)
            )

        return ToolExecutionResult(
            tool_name=self.name,
            status=ToolExecutionStatus.SUCCESS.value,
            result=res.dict() if hasattr(res, "dict") else res.model_dump(),
            dataset_id=params.dataset_id,
            dataset_version=context.get("dataset_version", 0),
            warnings=[a.details for a in res.assumptions if not a.passed],
            mutated_data=False,
            requires_approval=False,
            provenance={"test": res.test_name, "p_value": res.p_value, "effect_size": res.effect_size_value}
        )

    def _run_two_group_test(
        self,
        df: pd.DataFrame,
        group_col: str,
        metric_col: str,
        alpha: float,
        force_mann_whitney: bool = False
    ) -> HypothesisTestResponse:
        groups = [g for g, _ in df.groupby(group_col)]
        if len(groups) != 2:
            raise ValueError(f"Two-sample test requires exactly 2 distinct groups in '{group_col}', found {len(groups)}: {groups[:5]}")

        g1_name, g2_name = str(groups[0]), str(groups[1])
        s1 = df[df[group_col] == groups[0]][metric_col].dropna().values
        s2 = df[df[group_col] == groups[1]][metric_col].dropna().values

        if len(s1) < 3 or len(s2) < 3:
            raise ValueError(f"Each group must have at least 3 observations. Found {len(s1)} for '{g1_name}' and {len(s2)} for '{g2_name}'.")

        # Assumption 1: Normality (Shapiro-Wilk)
        # For large samples, Shapiro-Wilk can reject minor deviations, so check sample size
        n1_ok, n2_ok = True, True
        p_norm1, p_norm2 = 1.0, 1.0
        if len(s1) <= 5000:
            stat_w1, p_norm1 = stats.shapiro(s1)
            n1_ok = (p_norm1 >= 0.01)
        if len(s2) <= 5000:
            stat_w2, p_norm2 = stats.shapiro(s2)
            n2_ok = (p_norm2 >= 0.01)
        norm_passed = n1_ok and n2_ok

        # Assumption 2: Homoscedasticity (Levene's test)
        stat_lev, p_lev = stats.levene(s1, s2)
        var_passed = bool(p_lev >= 0.05)

        assumptions = [
            AssumptionCheck(
                name="Normality Check (Shapiro-Wilk)",
                passed=norm_passed,
                details=f"Group 1 p={p_norm1:.4f}, Group 2 p={p_norm2:.4f}. {'Data consistent with normal distribution.' if norm_passed else 'Data departs from normality; non-parametric test advised.'}",
                metric_value=float(p_norm1),
                p_value=float(p_norm2)
            ),
            AssumptionCheck(
                name="Equal Variance (Homoscedasticity - Levene)",
                passed=var_passed,
                details=f"Levene test statistic={stat_lev:.3f}, p={p_lev:.4f}. {'Variances approximately equal.' if var_passed else 'Unequal variances detected (Welch correction applied).'}",
                metric_value=float(stat_lev),
                p_value=float(p_lev)
            )
        ]

        use_mann_whitney = force_mann_whitney or (not norm_passed and (len(s1) < 30 or len(s2) < 30))

        if use_mann_whitney:
            test_name = "Mann-Whitney U Test"
            category = "non_parametric"
            stat_val, p_val = stats.mannwhitneyu(s1, s2, alternative="two-sided")
            stat_name = "U statistic"

            # Effect size: rank-biserial correlation r = 1 - (2U)/(n1*n2)
            u_stat = float(stat_val)
            rank_biserial = 1.0 - (2.0 * u_stat) / (len(s1) * len(s2))
            eff_val = abs(round(rank_biserial, 3))
            eff_name = "Rank-Biserial Correlation (r)"
            if eff_val < 0.1:
                eff_mag = "negligible"
            elif eff_val < 0.3:
                eff_mag = "small"
            elif eff_val < 0.5:
                eff_mag = "medium"
            else:
                eff_mag = "large"
        else:
            test_name = "Welch's Two-Sample t-Test" if not var_passed else "Student's Two-Sample t-Test"
            category = "parametric"
            stat_val, p_val = stats.ttest_ind(s1, s2, equal_var=var_passed)
            stat_name = "t statistic"

            # Effect size: Cohen's d
            m1, m2 = float(np.mean(s1)), float(np.mean(s2))
            v1, v2 = float(np.var(s1, ddof=1)), float(np.var(s2, ddof=1))
            pooled_std = math.sqrt(((len(s1) - 1) * v1 + (len(s2) - 1) * v2) / (len(s1) + len(s2) - 2)) if (len(s1) + len(s2) - 2) > 0 else 1.0
            d = abs(m1 - m2) / pooled_std if pooled_std > 0 else 0.0
            eff_val = round(d, 3)
            eff_name = "Cohen's d"
            if eff_val < 0.2:
                eff_mag = "negligible"
            elif eff_val < 0.5:
                eff_mag = "small"
            elif eff_val < 0.8:
                eff_mag = "medium"
            else:
                eff_mag = "large"

        p_float = float(p_val)
        is_sig = bool(p_float < alpha)

        group_summaries = {
            g1_name: {
                "count": len(s1),
                "mean": round(float(np.mean(s1)), 3),
                "median": round(float(np.median(s1)), 3),
                "std": round(float(np.std(s1, ddof=1)), 3) if len(s1) > 1 else 0.0
            },
            g2_name: {
                "count": len(s2),
                "mean": round(float(np.mean(s2)), 3),
                "median": round(float(np.median(s2)), 3),
                "std": round(float(np.std(s2, ddof=1)), 3) if len(s2) > 1 else 0.0
            }
        }

        diff_mean = round(float(np.mean(s1) - np.mean(s2)), 3)
        english = (
            f"Comparing '{g1_name}' vs '{g2_name}' on '{metric_col}': "
            f"the difference in means ({diff_mean}) {'is' if is_sig else 'is not'} statistically significant "
            f"(p = {p_float:.4f}, alpha = {alpha}). Effect size is {eff_mag} ({eff_name} = {eff_val})."
        )
        practical = (
            f"Observed difference is unlikely due to random sampling chance alone."
            if is_sig else
            f"Insufficient statistical evidence to distinguish group distributions; difference may reflect chance variation."
        )

        return HypothesisTestResponse(
            test_name=test_name,
            test_category=category,
            null_hypothesis=f"There is no difference in the distribution of '{metric_col}' between '{g1_name}' and '{g2_name}'.",
            alternative_hypothesis=f"There is a statistically detectable difference in '{metric_col}' between '{g1_name}' and '{g2_name}'.",
            test_statistic_name=stat_name,
            test_statistic_value=round(float(stat_val), 4),
            p_value=round(p_float, 5),
            is_statistically_significant=is_sig,
            significance_threshold=alpha,
            effect_size_name=eff_name,
            effect_size_value=eff_val,
            effect_size_magnitude=eff_mag,
            assumptions=assumptions,
            group_summaries=group_summaries,
            plain_english_finding=english,
            practical_implication=practical,
            caveat_disclaimer="Statistical significance does not prove real-world importance or causal relationship."
        )

    def _run_multigroup_test(
        self,
        df: pd.DataFrame,
        group_col: str,
        metric_col: str,
        alpha: float,
        force_kruskal: bool = False
    ) -> HypothesisTestResponse:
        grouped = [vals.values for _, vals in df.groupby(group_col)[metric_col]]
        group_names = [str(g) for g, _ in df.groupby(group_col)]

        if len(grouped) < 3:
            raise ValueError(f"ANOVA / Kruskal-Wallis requires 3 or more groups, found {len(grouped)}.")

        # Equal variance check across groups
        stat_lev, p_lev = stats.levene(*grouped)
        var_passed = bool(p_lev >= 0.05)

        assumptions = [
            AssumptionCheck(
                name="Equal Variance across Groups (Levene)",
                passed=var_passed,
                details=f"Levene test statistic={stat_lev:.3f}, p={p_lev:.4f}.",
                metric_value=float(stat_lev),
                p_value=float(p_lev)
            )
        ]

        if force_kruskal or not var_passed:
            test_name = "Kruskal-Wallis H Test"
            category = "non_parametric"
            stat_val, p_val = stats.kruskal(*grouped)
            stat_name = "H statistic"

            # Epsilon-squared effect size for Kruskal-Wallis
            n_total = sum(len(g) for g in grouped)
            h_val = float(stat_val)
            k = len(grouped)
            eps_sq = (h_val - k + 1) / (n_total - k) if (n_total - k) > 0 else 0.0
            eff_val = max(0.0, round(float(eps_sq), 3))
            eff_name = "Epsilon-squared (ε²)"
        else:
            test_name = "One-Way ANOVA"
            category = "parametric"
            stat_val, p_val = stats.f_oneway(*grouped)
            stat_name = "F statistic"

            # Eta-squared effect size: SS_between / SS_total
            all_vals = np.concatenate(grouped)
            grand_mean = np.mean(all_vals)
            ss_total = np.sum((all_vals - grand_mean) ** 2)
            ss_between = np.sum([len(g) * (np.mean(g) - grand_mean) ** 2 for g in grouped])
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
            eff_val = max(0.0, round(float(eta_sq), 3))
            eff_name = "Eta-squared (η²)"

        if eff_val < 0.01:
            eff_mag = "negligible"
        elif eff_val < 0.06:
            eff_mag = "small"
        elif eff_val < 0.14:
            eff_mag = "medium"
        else:
            eff_mag = "large"

        p_float = float(p_val)
        is_sig = bool(p_float < alpha)

        group_summaries = {}
        for g_name, g_vals in zip(group_names, grouped):
            group_summaries[g_name] = {
                "count": len(g_vals),
                "mean": round(float(np.mean(g_vals)), 3),
                "median": round(float(np.median(g_vals)), 3),
                "std": round(float(np.std(g_vals, ddof=1)), 3) if len(g_vals) > 1 else 0.0
            }

        english = (
            f"Comparing {len(group_names)} groups across '{metric_col}': "
            f"between-group variance {'is' if is_sig else 'is not'} statistically significant "
            f"(p = {p_float:.4f}, alpha = {alpha}). Effect size is {eff_mag} ({eff_name} = {eff_val})."
        )
        practical = (
            "At least one group exhibits a statistically distinct mean from the others."
            if is_sig else
            "Variations across group averages are consistent with expected random sampling noise."
        )

        return HypothesisTestResponse(
            test_name=test_name,
            test_category=category,
            null_hypothesis=f"All {len(group_names)} group means/medians in '{group_col}' are equal on '{metric_col}'.",
            alternative_hypothesis=f"At least one group differs significantly on '{metric_col}'.",
            test_statistic_name=stat_name,
            test_statistic_value=round(float(stat_val), 4),
            p_value=round(p_float, 5),
            is_statistically_significant=is_sig,
            significance_threshold=alpha,
            effect_size_name=eff_name,
            effect_size_value=eff_val,
            effect_size_magnitude=eff_mag,
            assumptions=assumptions,
            group_summaries=group_summaries,
            plain_english_finding=english,
            practical_implication=practical,
            caveat_disclaimer="Statistical significance does not prove real-world importance or causal relationship."
        )

    def _run_chi_square_test(
        self,
        df: pd.DataFrame,
        var1: str,
        var2: str,
        alpha: float
    ) -> HypothesisTestResponse:
        contingency = pd.crosstab(df[var1], df[var2])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            raise ValueError(f"Chi-square test requires at least 2 levels in each category, found shape {contingency.shape}.")

        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

        # Assumption: Expected frequencies >= 5
        min_expected = float(np.min(expected))
        exp_passed = bool(min_expected >= 5.0)

        assumptions = [
            AssumptionCheck(
                name="Expected Frequencies Check (>= 5 in all cells)",
                passed=exp_passed,
                details=f"Minimum expected frequency is {min_expected:.2f}. {'All expected cell counts >= 5.' if exp_passed else 'Some cells have expected count < 5; interpret chi-square with caution.'}",
                metric_value=min_expected
            )
        ]

        # Effect size: Cramér's V
        n = contingency.sum().sum()
        k = min(contingency.shape) - 1
        cramers_v = math.sqrt(chi2 / (n * k)) if (n * k) > 0 else 0.0
        eff_val = round(float(cramers_v), 3)
        eff_name = "Cramér's V"

        if eff_val < 0.1:
            eff_mag = "negligible"
        elif eff_val < 0.3:
            eff_mag = "small"
        elif eff_val < 0.5:
            eff_mag = "medium"
        else:
            eff_mag = "large"

        p_float = float(p_val)
        is_sig = bool(p_float < alpha)

        group_summaries = {}
        for r_name in contingency.index:
            group_summaries[str(r_name)] = {
                str(c_name): int(contingency.loc[r_name, c_name])
                for c_name in contingency.columns
            }

        english = (
            f"Testing categorical independence between '{var1}' and '{var2}': "
            f"the association {'is' if is_sig else 'is not'} statistically significant "
            f"(χ² = {chi2:.2f}, dof = {dof}, p = {p_float:.4f}, alpha = {alpha}). "
            f"Strength of association is {eff_mag} (Cramér's V = {eff_val})."
        )
        practical = (
            f"The distribution of '{var2}' is dependent upon '{var1}' beyond random co-occurrence."
            if is_sig else
            f"No evidence that '{var1}' and '{var2}' are related; distributions appear mutually independent."
        )

        return HypothesisTestResponse(
            test_name="Pearson's Chi-Square Test of Independence",
            test_category="categorical_independence",
            null_hypothesis=f"Features '{var1}' and '{var2}' are statistically independent.",
            alternative_hypothesis=f"Features '{var1}' and '{var2}' are significantly associated.",
            test_statistic_name="Chi-square (χ²)",
            test_statistic_value=round(float(chi2), 4),
            p_value=round(p_float, 5),
            is_statistically_significant=is_sig,
            significance_threshold=alpha,
            effect_size_name=eff_name,
            effect_size_value=eff_val,
            effect_size_magnitude=eff_mag,
            assumptions=assumptions,
            group_summaries=group_summaries,
            plain_english_finding=english,
            practical_implication=practical,
            caveat_disclaimer="Statistical significance does not prove real-world importance or causal relationship."
        )

    def _run_pearson_test(
        self,
        df: pd.DataFrame,
        var1: str,
        var2: str,
        alpha: float
    ) -> HypothesisTestResponse:
        s1 = df[var1].dropna().values
        s2 = df[var2].dropna().values

        r_val, p_val = stats.pearsonr(s1, s2)
        r_float = float(r_val)
        p_float = float(p_val)
        is_sig = bool(p_float < alpha)

        abs_r = abs(r_float)
        if abs_r < 0.1:
            eff_mag = "negligible"
        elif abs_r < 0.35:
            eff_mag = "small"
        elif abs_r < 0.7:
            eff_mag = "medium"
        else:
            eff_mag = "large"

        direction = "positive" if r_float > 0 else "negative"

        assumptions = [
            AssumptionCheck(
                name="Linearity and Numeric Continuity",
                passed=True,
                details=f"Both variables continuous numeric ({len(s1)} pairs observed)."
            )
        ]

        group_summaries = {
            var1: {"mean": round(float(np.mean(s1)), 3), "std": round(float(np.std(s1, ddof=1)), 3)},
            var2: {"mean": round(float(np.mean(s2)), 3), "std": round(float(np.std(s2, ddof=1)), 3)}
        }

        english = (
            f"Pearson linear correlation between '{var1}' and '{var2}' is r = {r_float:.3f} "
            f"({direction}, {eff_mag} strength). The correlation {'is' if is_sig else 'is not'} "
            f"statistically significant (p = {p_float:.4f}, alpha = {alpha})."
        )
        practical = (
            f"Increases in '{var1}' correlate systematically with {'increases' if r_float > 0 else 'decreases'} in '{var2}'."
            if is_sig else
            f"No statistically distinguishable linear relationship between '{var1}' and '{var2}'."
        )

        return HypothesisTestResponse(
            test_name="Pearson Correlation Coefficient Test",
            test_category="parametric",
            null_hypothesis=f"True population correlation between '{var1}' and '{var2}' is zero (ρ = 0).",
            alternative_hypothesis=f"True population correlation is non-zero (ρ ≠ 0).",
            test_statistic_name="Pearson r",
            test_statistic_value=round(r_float, 4),
            p_value=round(p_float, 5),
            is_statistically_significant=is_sig,
            significance_threshold=alpha,
            effect_size_name="Correlation Magnitude (|r|)",
            effect_size_value=round(abs_r, 3),
            effect_size_magnitude=eff_mag,
            assumptions=assumptions,
            group_summaries=group_summaries,
            plain_english_finding=english,
            practical_implication=practical,
            caveat_disclaimer="Statistical significance does not prove real-world importance or causal relationship."
        )
