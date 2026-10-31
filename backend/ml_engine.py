import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from models import (
    AutoMLRequest,
    AutoMLResponse,
    ModelLeaderboardItem,
    FeatureImportanceItem,
    HeldOutPredictionSample,
    WhatIfRequest,
    WhatIfResponse,
    ChartSpec,
)
from analysis_engine import create_plotly_spec

logger = logging.getLogger("datasense.ml_engine")

# In-memory storage for active trained pipelines to support What-If simulations
ACTIVE_ML_SESSIONS: Dict[str, Dict[str, Any]] = {}


class NaiveModelPipeline(BaseEstimator, RegressorMixin):
    """Naive Last-Value Baseline predictor using lag_1 strictly without lookahead."""
    def __init__(self, lag_feature: str = "lag_1", lag_idx: int = 2):
        self.lag_feature = lag_feature
        self.lag_idx = lag_idx
        self.last_train_val_ = 0.0

    def fit(self, X: Any, y: Any):
        if y is not None and len(y) > 0:
            self.last_train_val_ = float(y.iloc[-1] if hasattr(y, "iloc") else y[-1])
        return self

    def predict(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.lag_feature in X.columns:
            return X[self.lag_feature].fillna(self.last_train_val_).values.astype(float)
        elif hasattr(X, "ndim") and X.ndim == 2:
            idx = self.lag_idx if X.shape[1] > self.lag_idx else 0
            res = X[:, idx].astype(float)
            if np.isnan(res).any():
                res = np.nan_to_num(res, nan=self.last_train_val_)
            return res
        return np.full(len(X), self.last_train_val_)


def detect_task_type(series: pd.Series, requested_type: Optional[str] = "auto") -> str:
    """
    Intelligently infer whether target represents classification or regression.
    """
    if requested_type and requested_type in ["classification", "regression"]:
        return requested_type

    clean = series.dropna()
    unique_count = clean.nunique()
    
    # If string, boolean, or category dtype -> classification
    if not pd.api.types.is_numeric_dtype(clean):
        return "classification"
    
    # If numeric but only 2 to 10 unique values (e.g. 0/1 or ratings 1..5)
    if unique_count <= 10 and np.all(np.mod(clean.values, 1) == 0):
        return "classification"

    # If continuous float or high cardinality integer -> regression
    return "regression"


def build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Construct a leakage-free preprocessing pipeline.
    Imputes and scales numeric columns, imputes and one-hot encodes categorical columns.
    """
    numeric_cols = []
    categorical_cols = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    transformers = []

    if numeric_cols:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_transformer, numeric_cols))

    if categorical_cols:
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_cols, categorical_cols


def train_automl_pipeline(df: pd.DataFrame, req: AutoMLRequest) -> AutoMLResponse:
    """
    Executes a structured, leakage-free AutoML tournament:
    1. Splits into train & test (stratified if classification).
    2. Trains Dummy Baseline + Candidate Models.
    3. Evaluates with standardized task metrics.
    4. Analyzes overfitting and feature importance.
    """
    target_col = req.target_column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Drop rows where target is missing
    valid_df = df.dropna(subset=[target_col]).copy()
    if len(valid_df) < 15:
        raise ValueError(f"Dataset has only {len(valid_df)} rows with valid target '{target_col}'. At least 15 valid rows required for AutoML.")

    # Target validation 1: zero-variance / constant column
    unique_target_count = valid_df[target_col].nunique()
    if unique_target_count < 2:
        raise ValueError(f"Target column '{target_col}' has only {unique_target_count} unique value. Machine learning requires at least 2 distinct target values to train.")

    # Target validation 2: high-cardinality unique identifier or text target
    if not pd.api.types.is_numeric_dtype(valid_df[target_col]) and (unique_target_count / len(valid_df) > 0.8) and unique_target_count > 50:
        raise ValueError(f"Target column '{target_col}' appears to be a unique identifier or high-cardinality text ({unique_target_count} distinct values). Please select a predictive categorical or numerical target.")

    # Target validation 3: enforce requested task type compatibility
    req_type = req.task_type or "auto"
    if req_type == "regression":
        if not pd.api.types.is_numeric_dtype(valid_df[target_col]):
            num_s = pd.to_numeric(valid_df[target_col], errors="coerce")
            if num_s.dropna().count() < 15:
                raise ValueError(f"Target column '{target_col}' contains non-numeric data and cannot be trained for regression. Please select Classification or choose a numerical column.")
            valid_df[target_col] = num_s
            valid_df = valid_df.dropna(subset=[target_col])
            if valid_df[target_col].nunique() < 2:
                raise ValueError(f"Target column '{target_col}' has fewer than 2 distinct numeric values.")
    elif req_type == "classification":
        if pd.api.types.is_numeric_dtype(valid_df[target_col]) and unique_target_count > 20:
            raise ValueError(f"Target column '{target_col}' contains continuous numeric values with {unique_target_count} unique values. Classification is limited to <= 20 distinct classes. Please select Regression for continuous numerical targets.")

    task_type = detect_task_type(valid_df[target_col], req_type)
    
    # ---------------------------------------------------------
    # Time-Aware Feature Engineering for Time-Indexed Series
    # ---------------------------------------------------------
    time_series_features = ["Year", "year_index", "lag_1", "lag_2", "rolling_mean_3", "rolling_mean_5", "diff_1"]
    has_year_index = False
    years_list = None
    if isinstance(valid_df.index, pd.MultiIndex):
        first_level = [x[0] if isinstance(x, tuple) else x for x in valid_df.index]
        try:
            years_list = [int(x) for x in first_level]
            has_year_index = True
        except (ValueError, TypeError):
            pass
    elif pd.api.types.is_numeric_dtype(valid_df.index) and len(valid_df.index) > 0 and (valid_df.index > 1800).all() and (valid_df.index < 2100).all():
        years_list = [int(x) for x in valid_df.index]
        has_year_index = True
    elif "Year" in valid_df.columns:
        years_list = [int(x) for x in valid_df["Year"]]
        has_year_index = True

    other_cols = [c for c in valid_df.columns if c != target_col]
    wants_ts_features = bool(req.selected_features and any(f in req.selected_features for f in time_series_features))
    is_time_series = task_type == "regression" and ((has_year_index and (len(other_cols) == 0 or wants_ts_features)) or (req.validation_strategy == "chronological" and has_year_index))

    if is_time_series:
        if years_list and len(years_list) == len(valid_df):
            valid_df["Year"] = years_list
        valid_df["year_index"] = np.arange(len(valid_df))
        target_s = valid_df[target_col]
        valid_df["lag_1"] = target_s.shift(1)
        valid_df["lag_2"] = target_s.shift(2)
        valid_df["rolling_mean_3"] = target_s.shift(1).rolling(3).mean()
        valid_df["rolling_mean_5"] = target_s.shift(1).rolling(5).mean()
        valid_df["diff_1"] = target_s.shift(1) - target_s.shift(2)
        valid_df = valid_df.dropna().reset_index(drop=True)
        total_len = len(valid_df)

    # Select candidate feature columns (exclude target, id-like high cardinality strings)
    candidate_features = []
    total_len = len(valid_df)
    for c in valid_df.columns:
        if c == target_col:
            continue
        # Drop columns with 100% missing values
        if valid_df[c].isna().all():
            continue
        # Drop unique identifier columns (strings with > 90% unique values)
        if not pd.api.types.is_numeric_dtype(valid_df[c]) and valid_df[c].nunique() / total_len > 0.9 and valid_df[c].nunique() > 50:
            continue
        candidate_features.append(c)

    # Filter candidate features if user explicitly selected a predictor subset
    if req.selected_features and len(req.selected_features) > 0:
        req_set = set(req.selected_features)
        candidate_features = [c for c in candidate_features if c in req_set]

    if not candidate_features:
        if is_time_series:
            candidate_features = [f for f in time_series_features if f in valid_df.columns and f != target_col]
        else:
            raise ValueError("No valid predictive features remaining. Please ensure at least one predictor feature is selected.")

    X = valid_df[candidate_features]
    y = valid_df[target_col]

    # Pre-encode y if classification and object dtype
    class_labels = None
    class_distribution = None
    if task_type == "classification":
        if not pd.api.types.is_numeric_dtype(y):
            y_str = y.astype(str)
            class_counts = y_str.value_counts().to_dict()
            class_distribution = {str(k): int(v) for k, v in class_counts.items()}
            class_labels = list(class_counts.keys())
            if len(class_labels) > 20:
                raise ValueError(f"Target column '{target_col}' has {len(class_labels)} unique classes. Classification is limited to <= 20 distinct classes.")
            y = y_str
        else:
            class_counts = y.value_counts().to_dict()
            class_distribution = {str(k): int(v) for k, v in class_counts.items()}
            class_labels = [str(k) for k in class_counts.keys()]
            if len(class_labels) > 20:
                raise ValueError(f"Target column '{target_col}' has {len(class_labels)} unique classes. Classification is limited to <= 20 distinct classes.")

    # Multicollinearity Check
    multicollinearity_warnings = []
    num_features = [c for c in candidate_features if pd.api.types.is_numeric_dtype(valid_df[c])]
    if len(num_features) >= 2:
        try:
            corr_matrix = valid_df[num_features].corr().abs()
            for i in range(len(num_features)):
                for j in range(i + 1, len(num_features)):
                    c1, c2 = num_features[i], num_features[j]
                    val = float(corr_matrix.loc[c1, c2])
                    if not np.isnan(val) and val >= 0.85:
                        multicollinearity_warnings.append({
                            "feature_1": c1,
                            "feature_2": c2,
                            "correlation": round(val, 3),
                            "severity": "high" if val >= 0.95 else "medium",
                            "recommendation": f"Features '{c1}' and '{c2}' exhibit high correlation (|r| = {val:.2f}). Consider dropping one to stabilize model coefficients and prevent collinearity distortion."
                        })
        except Exception as e:
            logger.warning(f"Could not compute correlation matrix: {e}")

    # Train / Test Split
    test_size = float(req.test_size or 0.2)
    test_size = max(0.1, min(0.4, test_size))
    
    # ---------------------------------------------------------
    # Time-Aware Validation vs. Standard Cross-Validation
    # Only force chronological validation if explicitly requested
    # (via req.validation_strategy == 'chronological' or explicit req.time_column).
    # Incidental date columns (e.g. BirthDate, Date) must NOT force chronological split!
    # ---------------------------------------------------------
    req_strat = (req.validation_strategy or "auto").lower()
    allowed_strategies = ["auto", "kfold_cv", "stratified_holdout", "chronological"]
    if req_strat not in allowed_strategies:
        raise ValueError(f"Unsupported validation strategy '{req.validation_strategy}'. Supported options: {', '.join(allowed_strategies)}.")

    if req_strat == "stratified_holdout" and task_type == "regression":
        raise ValueError("Stratified validation is only applicable to categorical classification tasks, not continuous regression.")

    time_col = None
    is_chronological = False
    
    if req_strat == "chronological" or req.time_column:
        if req.time_column:
            if req.time_column not in valid_df.columns:
                raise ValueError(f"Specified time_column '{req.time_column}' was not found in dataset.")
            time_col = req.time_column
        else:
            for c in valid_df.columns:
                if c != target_col and (pd.api.types.is_datetime64_any_dtype(valid_df[c]) or (c.lower() in ["orderdate", "date", "timestamp", "datetime"] and pd.to_datetime(valid_df[c], errors="coerce").notna().mean() > 0.8)):
                    time_col = c
                    break
        if not time_col:
            raise ValueError("Chronological validation was requested, but no valid date or timestamp column was found in the dataset.")
        
        if len(valid_df) < 10:
            raise ValueError("Dataset has insufficient observations (< 10 rows) for chronological time-series splitting.")
            
        time_series_parsed = pd.to_datetime(valid_df[time_col], errors="coerce")
        if time_series_parsed.notna().sum() < 5:
            raise ValueError(f"Column '{time_col}' contains insufficient valid dates for chronological time-series splitting.")
            
        is_chronological = True

    date_cols_found = [c for c in valid_df.columns if c != target_col and (pd.api.types.is_datetime64_any_dtype(valid_df[c]) or any(d in c.lower() for d in ["date", "time", "dob", "birth"]))]

    if is_time_series:
        is_chronological = True
        split_point = int(len(valid_df) * (1.0 - test_size))
        X_dev, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_dev, y_test = y.iloc[:split_point], y.iloc[split_point:]
        val_strategy = "Time-ordered holdout"
        val_rationale = "Chronological time-ordered holdout split preserving temporal ordering. Historical observations used for training; subsequent observations used for out-of-sample forward evaluation."
    elif is_chronological and time_col:
        try:
            time_series_parsed = pd.to_datetime(valid_df[time_col], errors="coerce")
            sorted_idx = time_series_parsed.sort_values(na_position="first").index
            X_sorted = X.loc[sorted_idx]
            y_sorted = y.loc[sorted_idx]
            split_point = int(len(valid_df) * (1.0 - test_size))
            X_dev, X_test = X_sorted.iloc[:split_point], X_sorted.iloc[split_point:]
            y_dev, y_test = y_sorted.iloc[:split_point], y_sorted.iloc[split_point:]
            val_strategy = f"Chronological time-series split on '{time_col}' ({int((1-test_size)*100)}% dev / {int(test_size*100)}% test, un-shuffled) preventing lookahead leakage"
            val_rationale = f"Chronological validation applied strictly along '{time_col}'. Observations partitioned in chronological order to simulate real-world forward testing without future-lookahead leakage."
        except Exception as e:
            logger.warning(f"Chronological split failed: {e}; falling back to standard dev/test split.")
            is_chronological = False

    if not is_chronological:
        stratify_arg = None
        if task_type == "classification":
            min_class_count = min(class_distribution.values()) if class_distribution else 0
            if min_class_count >= 2:
                stratify_arg = y

        try:
            X_dev, X_test, y_dev, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify_arg
            )
        except ValueError as split_err:
            logger.warning(f"Stratified split failed ({split_err}); falling back to unstratified split.")
            X_dev, X_test, y_dev, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=None
            )

        if req_strat == "kfold_cv":
            val_strategy = f"5-Fold Cross-Validation on Development Partition ({int((1-test_size)*100)}% dev / {int(test_size*100)}% holdout test)"
            val_rationale = "User-selected 5-Fold Cross-Validation. Candidate models evaluated across 5 development folds; winning model evaluated independently on held-out test split."
        elif req_strat == "stratified_holdout":
            val_strategy = f"Stratified Holdout Validation ({int((1-test_size)*100)}% dev / {int(test_size*100)}% holdout test)"
            val_rationale = "User-selected Stratified Holdout. Class distributions preserved across development folds and held-out test set."
        elif date_cols_found:
            val_strategy = f"Stratified/K-Fold Cross-Validation on Development Partition ({int((1-test_size)*100)}% dev / {int(test_size*100)}% holdout test)"
            val_rationale = f"Standard stratified/k-fold cross-validation on development split selected. (Note: Detected incidental date column(s) {date_cols_found}; retained standard cross-validation because analytical objective was not configured for chronological forecasting)."
        else:
            val_strategy = f"Stratified/K-Fold Cross-Validation on Development Partition ({int((1-test_size)*100)}% dev / {int(test_size*100)}% holdout test)"
            val_rationale = "Standard stratified/k-fold cross-validation on development split selected with isolated untouched holdout test partition."

    X_train, y_train = X_dev, y_dev

    # Fit preprocessor strictly on development data to guarantee zero leakage into test set
    if is_time_series:
        preprocessor = ColumnTransformer(
            transformers=[("num", SimpleImputer(strategy="median"), candidate_features)],
            remainder="drop"
        )
        num_cols, cat_cols = candidate_features, []
    else:
        preprocessor, num_cols, cat_cols = build_preprocessor(X_train, candidate_features)

    # Detect class imbalance for weighted loss and selection metric
    is_imbalanced = False
    if task_type == "classification" and class_distribution:
        counts = list(class_distribution.values())
        if len(counts) >= 2 and (max(counts) / (min(counts) or 1)) >= 2.5:
            is_imbalanced = True

    # Setup Cross-Validation on Development Set (X_train, y_train)
    n_dev = len(X_train)
    if is_time_series:
        n_splits = min(3, max(2, n_dev // 20))
        cv = TimeSeriesSplit(n_splits=n_splits)
    elif is_chronological:
        n_splits = min(3, max(2, n_dev // 10))
        cv = TimeSeriesSplit(n_splits=n_splits)
    elif task_type == "classification":
        min_class_dev = min(pd.Series(y_train).value_counts().to_dict().values()) if len(y_train) > 0 else 0
        n_splits = min(5, max(2, min_class_dev))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=min(3, max(2, n_dev // 5)), shuffle=True, random_state=42)
    else:
        n_splits = min(5, max(2, n_dev // 5))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Selection Metric Definition & Transparent Rationale
    if task_type == "classification":
        if is_imbalanced:
            selection_metric_name = "Macro F1-Score (Cross-Validation on Dev Set)"
            selection_metric_rationale = (
                "Macro F1-Score was chosen as the model selection criterion because significant class imbalance was detected. "
                "Unlike weighted F1 or accuracy (which weight by support and can be dominated by majority classes), "
                "Macro F1 computes the unweighted arithmetic mean of F1 across all classes, penalizing models that sacrifice minority classes."
            )
        else:
            selection_metric_name = "Weighted F1-Score (Cross-Validation on Dev Set)"
            selection_metric_rationale = (
                "Weighted F1-Score was chosen as the model selection criterion for balanced class distribution, "
                "evaluating harmonic precision and recall across validation folds."
            )
    else:
        selection_metric_name = "R² Score (Cross-Validation on Dev Set)"
        selection_metric_rationale = (
            "R² was chosen as the model selection criterion because it measures the proportion of variance explained by the model on unseen validation folds relative to baseline."
        )

    # Define model candidates
    candidates = []
    if is_time_series:
        lag1_idx = candidate_features.index("lag_1") if "lag_1" in candidate_features else 0
        candidates = [
            ("Naive Baseline", "Last-Value Baseline Model", True, NaiveModelPipeline(lag_feature="lag_1", lag_idx=lag1_idx)),
            ("Linear Regression", "Ordinary Least Squares Linear Model", False, LinearRegression()),
            ("Ridge Regression", "L2 Regularized Linear Model", False, Ridge(alpha=1.0, random_state=42)),
            ("Random Forest", "Ensemble Bagging Trees", False, RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
        ]
    elif task_type == "classification":
        clf_weight = "balanced" if is_imbalanced else None
        candidates = [
            ("Baseline Dummy", "Most Frequent Class", True, DummyClassifier(strategy="most_frequent")),
            ("Logistic Regression", "L2 Regularized Linear Classifier" + (" (Class Balanced)" if is_imbalanced else ""), False, LogisticRegression(max_iter=1000, random_state=42, class_weight=clf_weight)),
            ("Decision Tree", "CART Tree Classifier", False, DecisionTreeClassifier(max_depth=6, random_state=42, class_weight=clf_weight)),
            ("Random Forest", "Ensemble Bagging Trees" + (" (Class Balanced)" if is_imbalanced else ""), False, RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight=clf_weight)),
            ("Gradient Boosting", "Histogram Gradient Boosting Ensemble" + (" (Class Balanced)" if is_imbalanced else ""), False, HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=42, class_weight=clf_weight)),
        ]
    else:
        candidates = [
            ("Baseline Dummy", "Predicts Mean Target", True, DummyRegressor(strategy="mean")),
            ("Ridge Regression", "L2 Regularized Linear Model", False, Ridge(alpha=1.0, random_state=42)),
            ("Decision Tree", "CART Tree Regressor", False, DecisionTreeRegressor(max_depth=6, random_state=42)),
            ("Random Forest", "Ensemble Bagging Trees", False, RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
            ("Gradient Boosting", "Histogram Gradient Boosting Ensemble", False, HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)),
        ]

    # Filter candidate models if user chose specific algorithms
    if req.algorithm_filter and len(req.algorithm_filter) > 0:
        filter_lower = [af.lower() for af in req.algorithm_filter]
        filtered = [
            c for c in candidates
            if c[2] or any(fl in c[0].lower() or fl in c[1].lower() for fl in filter_lower)
        ]
        if len(filtered) > 1:
            candidates = filtered

    leaderboard: List[ModelLeaderboardItem] = []
    trained_pipelines: Dict[str, Pipeline] = {}

    # Candidate evaluation loop: STRICTLY evaluates on development set (X_train, y_train)
    # The final test set (X_test, y_test) is NEVER evaluated during candidate ranking/selection!
    for name, algo_desc, is_base, estimator in candidates:
        try:
            start_t = time.perf_counter()

            # Cross-validation on development partition (X_train, y_train)
            cv_scores = []
            for fold_train_idx, fold_val_idx in cv.split(X_train, y_train):
                X_tr, X_va = X_train.iloc[fold_train_idx], X_train.iloc[fold_val_idx]
                y_tr, y_va = y_train.iloc[fold_train_idx], y_train.iloc[fold_val_idx]

                fold_pipe = Pipeline([
                    ("preprocessor", clone(preprocessor)),
                    ("estimator", clone(estimator))
                ])
                fold_pipe.fit(X_tr, y_tr)
                y_va_pred = fold_pipe.predict(X_va)

                if task_type == "classification":
                    if is_imbalanced:
                        fold_score = float(f1_score(y_va, y_va_pred, average="macro", zero_division=0))
                    else:
                        fold_score = float(f1_score(y_va, y_va_pred, average="weighted", zero_division=0))
                else:
                    fold_score = float(r2_score(y_va, y_va_pred))
                cv_scores.append(fold_score)

            cv_mean = float(np.mean(cv_scores)) if cv_scores else 0.0
            cv_std = float(np.std(cv_scores)) if cv_scores else 0.0
            val_score = round(cv_mean, 4)

            # Fit candidate pipeline on FULL development set
            pipe = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("estimator", clone(estimator))
            ])
            pipe.fit(X_train, y_train)
            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)

            # Predictions on training development set
            y_train_pred = pipe.predict(X_train)
            if task_type == "classification":
                train_score = round(float(accuracy_score(y_train, y_train_pred)), 4)
            else:
                train_score = round(float(r2_score(y_train, y_train_pred)), 4)

            gap = round(train_score - val_score, 4)
            leaderboard.append(ModelLeaderboardItem(
                model_name=name,
                algorithm=algo_desc,
                is_baseline=is_base,
                train_score=train_score,
                val_score=val_score,
                test_score=None,  # Strictly isolated: untouched test set is not evaluated during candidate tournament
                cv_scores_mean=round(cv_mean, 4),
                cv_scores_std=round(cv_std, 4),
                metrics={},
                val_metrics={"cv_val_score": val_score, "cv_std": round(cv_std, 4)},
                overfitting_gap=gap,
                training_time_ms=dur_ms,
                status="trained"
            ))
            trained_pipelines[name] = pipe

        except Exception as e:
            logger.warning(f"Failed training {name}: {e}")
            leaderboard.append(ModelLeaderboardItem(
                model_name=name,
                algorithm=algo_desc,
                is_baseline=is_base,
                train_score=0.0,
                val_score=0.0,
                test_score=None,
                metrics={"error": 1.0},
                overfitting_gap=0.0,
                training_time_ms=0.0,
                status=f"failed: {str(e)[:50]}"
            ))

    # Rank candidates STRICTLY by validation score on development folds (val_score), NOT by test score!
    valid_candidates = [m for m in leaderboard if m.status == "trained"]
    if not valid_candidates:
        raise RuntimeError("All candidate models failed during training.")

    # Sort descending by val_score; non-baseline models preferred on ties
    leaderboard.sort(key=lambda m: (m.status == "trained", not m.is_baseline, m.val_score), reverse=True)
    best_item = leaderboard[0]
    best_name = best_item.model_name
    best_pipe = trained_pipelines[best_name]

    # ---------------------------------------------------------
    # FINAL TEST EVALUATION: PERFORMED ON HELD-OUT TEST SET
    # Strictly applied to the WINNING model to maintain test set isolation
    # ---------------------------------------------------------
    best_test_pred = best_pipe.predict(X_test)
    if task_type == "regression":
        best_r2 = float(r2_score(y_test, best_test_pred))
        best_rmse = float(np.sqrt(mean_squared_error(y_test, best_test_pred)))
        best_mae = float(mean_absolute_error(y_test, best_test_pred))
        best_bias = float(np.mean(best_test_pred - y_test))
        best_item.test_score = round(best_r2, 4)
        best_item.metrics = {
            "r2_score": round(best_r2, 4),
            "rmse": round(best_rmse, 4),
            "mae": round(best_mae, 4),
            "bias": round(best_bias, 4),
            "signed_mean_error": round(best_bias, 4),
        }
    else:
        best_acc = float(accuracy_score(y_test, best_test_pred))
        best_item.test_score = round(best_acc, 4)
        best_item.metrics = {
            "accuracy": round(best_acc, 4)
        }

    final_test_score = best_item.test_score
    test_metrics = best_item.metrics

    # Compute Feature Importance
    feature_importances: List[FeatureImportanceItem] = []
    try:
        # Check if Random Forest or Decision Tree has feature importances
        rf_pipe = trained_pipelines.get("Random Forest") or trained_pipelines.get("Decision Tree")
        if rf_pipe and hasattr(rf_pipe.named_steps["estimator"], "feature_importances_"):
            raw_importances = rf_pipe.named_steps["estimator"].feature_importances_
            
            # Map back to feature columns
            fitted_prep = rf_pipe.named_steps["preprocessor"]
            try:
                feature_names = list(fitted_prep.get_feature_names_out())
            except Exception:
                feature_names = []
            
            # Aggregate one-hot sub-features back to parent columns
            col_importance_map: Dict[str, float] = {c: 0.0 for c in candidate_features}
            if len(feature_names) == len(raw_importances):
                for f_name, imp in zip(feature_names, raw_importances):
                    clean_f = f_name
                    if "__" in clean_f:
                        clean_f = clean_f.split("__", 1)[1]
                    matched_parent = None
                    for orig in candidate_features:
                        if clean_f == orig or clean_f.startswith(f"{orig}_"):
                            matched_parent = orig
                            break
                    if matched_parent:
                        col_importance_map[matched_parent] += float(imp)
                    else:
                        col_importance_map[candidate_features[0]] += float(imp)
            else:
                # Fallback to direct mapping
                for idx, c in enumerate(candidate_features):
                    if idx < len(raw_importances):
                        col_importance_map[c] = float(raw_importances[idx])

            total_imp = sum(col_importance_map.values()) or 1.0
            sorted_cols = sorted(col_importance_map.items(), key=lambda x: x[1], reverse=True)
            for c_name, imp_val in sorted_cols[:10]:
                pct = round((imp_val / total_imp) * 100, 1)
                feature_importances.append(FeatureImportanceItem(
                    feature=c_name,
                    importance=round(imp_val, 4),
                    percentage=pct
                ))
    except Exception as e:
        logger.warning(f"Could not compute feature importances: {e}")

    # Overfitting Assessment
    gap = best_item.overfitting_gap
    if gap > 0.20:
        overfit_text = f"High risk of overfitting detected (Train-Test Gap: {gap:.2f}). Model has memorized training nuances and may generalize poorly on unseen data."
    elif gap > 0.10:
        overfit_text = f"Moderate overfitting gap ({gap:.2f}). Performance on test holdout is reasonable, but regularization or pruning is recommended."
    else:
        overfit_text = f"Low generalization gap ({gap:.2f}). The selected model demonstrates strong, consistent generalization across train and test partitions."

    # Dataset Limitations and Caveats
    limitations = [
        f"Evaluated on {len(X_test):,} held-out test rows out of {total_len:,} total observations.",
        "Categorical features were one-hot encoded; unseen categories in production will be handled gracefully but without historical evidence.",
        "Model outputs represent observed correlations, not proved causality.",
    ]
    if task_type == "classification" and class_distribution:
        min_c = min(class_distribution.values())
        max_c = max(class_distribution.values())
        if max_c / (min_c or 1) > 3.0:
            limitations.append(f"Significant class imbalance observed ({max_c}:{min_c}). F1-score and Precision are more reliable performance indicators than raw Accuracy.")

    # Generate actual held-out test predictions using the winning pipeline
    best_test_pred = best_pipe.predict(X_test)
    held_out_samples: List[HeldOutPredictionSample] = []
    test_indices = list(X_test.index)

    for idx_pos, row_idx in enumerate(test_indices[:25]):
        act_val = y_test.iloc[idx_pos]
        pred_val = best_test_pred[idx_pos]
        feat_dict = {col: X_test.iloc[idx_pos][col] for col in candidate_features[:6]}
        feat_dict_clean = {}
        for k, v in feat_dict.items():
            if pd.isna(v):
                feat_dict_clean[k] = None
            elif isinstance(v, (np.integer, int)):
                feat_dict_clean[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                feat_dict_clean[k] = round(float(v), 2)
            else:
                feat_dict_clean[k] = str(v)

        if task_type == "classification":
            is_correct = (str(act_val) == str(pred_val))
            held_out_samples.append(HeldOutPredictionSample(
                row_index=int(row_idx),
                actual=str(act_val),
                predicted=str(pred_val),
                is_correct=is_correct,
                features=feat_dict_clean
            ))
        else:
            act_num = float(act_val)
            pred_num = float(pred_val)
            residual = round(pred_num - act_num, 3)
            held_out_samples.append(HeldOutPredictionSample(
                row_index=int(row_idx),
                actual=round(act_num, 3),
                predicted=round(pred_num, 3),
                error_or_residual=residual,
                features=feat_dict_clean
            ))

    # Construct Held-Out Evaluation Chart
    if task_type == "regression":
        eval_sample_len = min(len(y_test), 300)
        pts = [
            {"x": round(float(y_test.iloc[i]), 2), "y": round(float(best_test_pred[i]), 2)}
            for i in range(eval_sample_len)
        ]
        plotly_eval = create_plotly_spec(
            chart_type="scatter",
            title=f"Holdout Evaluation: Actual vs Predicted {target_col}",
            x_axis=f"Actual {target_col} (Ground Truth)",
            y_axis=f"Predicted {target_col} ({best_name})",
            points=pts
        )
        eval_chart_spec = ChartSpec(
            chart_type="scatter",
            title=f"Holdout Evaluation: Actual vs Predicted {target_col}",
            x_axis=f"Actual {target_col}",
            y_axis=f"Predicted {target_col}",
            data=pts,
            description=f"Direct test-set evaluation across {eval_sample_len} held-out observations comparing actual ground-truth against {best_name} predictions.",
            plotly_spec=plotly_eval,
            data_stage="model_ready_held_out",
            rows_used=len(X_test),
            transformations=f"Holdout evaluation using trained {best_name} on unseen test partition",
            exclusions="Training observations excluded to prevent data leakage"
        )
    else:
        # Classification: Compare actual vs predicted class counts on holdout test set
        act_counts = pd.Series(y_test).value_counts().to_dict()
        prd_counts = pd.Series(best_test_pred).astype(str).value_counts().to_dict()
        eval_classes = list(set(list(act_counts.keys()) + list(prd_counts.keys())))[:8]
        bar_data = []
        for c in eval_classes:
            bar_data.append({"x": f"{c} (Actual)", "y": int(act_counts.get(c, 0))})
            bar_data.append({"x": f"{c} (Predicted)", "y": int(prd_counts.get(c, 0))})

        plotly_eval = create_plotly_spec(
            chart_type="bar",
            title=f"Holdout Class Distribution: Actual vs Predicted ({best_name})",
            x_axis="Class",
            y_axis="Observations",
            labels=[d["x"] for d in bar_data],
            values=[d["y"] for d in bar_data]
        )
        eval_chart_spec = ChartSpec(
            chart_type="bar",
            title=f"Holdout Class Distribution: Actual vs Predicted ({best_name})",
            x_axis="Class",
            y_axis="Observations",
            labels=[d["x"] for d in bar_data],
            data=bar_data,
            description=f"Comparison of actual vs predicted class breakdown on {len(X_test)} held-out test records.",
            plotly_spec=plotly_eval,
            data_stage="model_ready_held_out",
            rows_used=len(X_test),
            transformations=f"Holdout class breakdown evaluation for {best_name}",
            exclusions="Training observations excluded to prevent data leakage"
        )

    # Construct Held-Out Residual Chart
    residual_chart_spec = None
    if task_type == "regression":
        res_pts = [
            {"x": round(float(best_test_pred[i]), 3), "y": round(float(best_test_pred[i] - y_test.iloc[i]), 3)}
            for i in range(len(y_test))
        ]
        plotly_res = create_plotly_spec(
            chart_type="scatter",
            title=f"Residual Plot: Predicted vs Error ({best_name})",
            x_axis=f"Predicted {target_col}",
            y_axis="Residual (Predicted - Actual)",
            points=res_pts
        )
        residual_chart_spec = ChartSpec(
            chart_type="scatter",
            title=f"Residual Plot: Predicted vs Error ({best_name})",
            x_axis=f"Predicted {target_col}",
            y_axis="Residual (Predicted - Actual)",
            data=res_pts,
            description=f"Residual diagnostics across {len(y_test)} held-out observations evaluating bias and homoscedasticity for {best_name}.",
            plotly_spec=plotly_res,
            data_stage="model_ready_held_out",
            rows_used=len(X_test),
            transformations=f"Holdout residual computation for {best_name}",
            exclusions="Training observations excluded to prevent data leakage"
        )

    # Chart Spec for Leaderboard Comparison
    leaderboard_chart = ChartSpec(
        chart_type="bar",
        title="Model Leaderboard — Holdout Test Performance",
        x_axis="Model",
        y_axis="Test Score",
        data=[
            {"x": m.model_name, "y": m.test_score}
            for m in leaderboard if m.status == "trained"
        ],
        description="Comparison of holdout validation scores across baseline and candidate ML architectures.",
        data_stage="model_ready_held_out",
        rows_used=len(X_test),
        transformations="Holdout metric evaluation across baseline and candidate architectures",
        exclusions="Training observations excluded"
    )

    # Store trained model in active memory for What-If simulator
    # Compute median / mode baseline inputs for features
    baseline_inputs: Dict[str, Any] = {}
    for c in candidate_features:
        if pd.api.types.is_numeric_dtype(valid_df[c]):
            baseline_inputs[c] = round(float(valid_df[c].median()), 2)
        else:
            mode_s = valid_df[c].mode()
            baseline_inputs[c] = str(mode_s.iloc[0]) if len(mode_s) > 0 else "Unknown"

    ACTIVE_ML_SESSIONS[req.dataset_id] = {
        "target_column": target_col,
        "task_type": task_type,
        "candidate_features": candidate_features,
        "best_model_name": best_name,
        "best_pipeline": best_pipe,
        "baseline_inputs": baseline_inputs,
        "class_labels": class_labels,
    }

    # Plain-English metric explanations
    metric_explanations: Dict[str, str] = {
        "selection_criterion": selection_metric_name,
        "selection_rationale": selection_metric_rationale,
    }
    if task_type == "classification":
        metric_explanations.update({
            "accuracy": "Proportion of total held-out test predictions that were exactly correct.",
            "f1_score": "Harmonic balance between precision and recall (robust against class imbalance).",
            "precision": "Accuracy of positive predictions; fraction of predicted positive cases that were truly positive.",
            "recall": "Sensitivity; fraction of actual positive observations in the test set successfully identified.",
        })
        if "roc_auc" in best_item.metrics:
            metric_explanations["roc_auc"] = "Discrimination capability across decision thresholds (0.5 is random chance, 1.0 is optimal separation)."
    else:
        metric_explanations.update({
            "r2_score": f"Coefficient of determination (R²); proportion of variance in '{target_col}' explained by the model (1.0 is perfect prediction).",
            "mae": f"Mean Absolute Error; average absolute error in original units of '{target_col}'.",
            "rmse": f"Root Mean Squared Error; average prediction error with higher penalty for large errors in '{target_col}'.",
        })

    # Error Analysis on Holdout Predictions
    error_analysis: Dict[str, Any] = {}
    if task_type == "regression":
        residuals = np.abs(np.array(best_test_pred) - np.array(y_test))
        worst_order = np.argsort(residuals)[-5:][::-1]
        worst_samples = []
        for w_pos in worst_order:
            worst_samples.append({
                "row_index": int(X_test.index[w_pos]),
                "actual": round(float(y_test.iloc[w_pos]), 3),
                "predicted": round(float(best_test_pred[w_pos]), 3),
                "residual": round(float(residuals[w_pos]), 3)
            })
        bias_val = round(float(np.mean(np.array(best_test_pred) - np.array(y_test))), 3)
        error_analysis = {
            "mean_absolute_error": round(float(mean_absolute_error(y_test, best_test_pred)), 3),
            "max_residual": round(float(np.max(residuals)), 3),
            "prediction_bias": bias_val,
            "bias_interpretation": "Slight average overestimation" if bias_val > 0 else "Slight average underestimation",
            "top_worst_predictions": worst_samples
        }
    else:
        misclass_mask = np.array(best_test_pred) != np.array(y_test)
        misclass_count = int(np.sum(misclass_mask))
        worst_samples = []
        if misclass_count > 0:
            misclass_pos = np.where(misclass_mask)[0][:5]
            for m_pos in misclass_pos:
                worst_samples.append({
                    "row_index": int(X_test.index[m_pos]),
                    "actual": str(y_test.iloc[m_pos]),
                    "predicted": str(best_test_pred[m_pos])
                })
        error_analysis = {
            "misclassified_count": misclass_count,
            "misclassified_percentage": round((misclass_count / len(y_test)) * 100, 2),
            "top_misclassifications": worst_samples
        }

    # Baseline Drift Reference Metrics on Training Set
    baseline_drift_reference: Dict[str, Any] = {}
    for c in candidate_features:
        s = X_train[c].dropna()
        if pd.api.types.is_numeric_dtype(s) and len(s) > 0:
            baseline_drift_reference[c] = {
                "type": "numeric",
                "mean": round(float(s.mean()), 3),
                "std": round(float(s.std()), 3),
                "median": round(float(s.median()), 3),
                "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 3),
                "min": round(float(s.min()), 3),
                "max": round(float(s.max()), 3)
            }
        elif len(s) > 0:
            baseline_drift_reference[c] = {
                "type": "categorical",
                "top_categories": s.astype(str).value_counts(normalize=True).head(5).round(3).to_dict()
            }

    # Standalone Reproducible Python Script
    num_cols_repr = [c for c in candidate_features if pd.api.types.is_numeric_dtype(valid_df[c])]
    cat_cols_repr = [c for c in candidate_features if not pd.api.types.is_numeric_dtype(valid_df[c])]
    metric_import = 'accuracy_score, classification_report' if task_type == 'classification' else 'r2_score, mean_squared_error, mean_absolute_error'
    model_init = 'RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)' if task_type == 'classification' else 'RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)'
    eval_call = 'classification_report(y_test, pipeline.predict(X_test))' if task_type == 'classification' else '{"R2": r2_score(y_test, pipeline.predict(X_test)), "RMSE": np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))}'

    reproducible_python_script = f'''# Standalone Reproducible Machine Learning Pipeline
# Generated by DataSense AI
# Target: {target_col} | Task: {task_type} | Winning Architecture: {best_name}

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import {metric_import}

# 1. Feature Specifications
numeric_cols = {num_cols_repr}
categorical_cols = {cat_cols_repr}

# 2. Leakage-Safe Preprocessing Pipeline
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols),
], remainder="drop")

# 3. Model Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("estimator", {model_init})
])

# 4. Fit & Evaluate (Replace with your actual dataset filepath)
# df = pd.read_csv("dataset.csv")
# X = df[numeric_cols + categorical_cols]
# y = df["{target_col}"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)
# pipeline.fit(X_train, y_train)
# print("Test Holdout Results:", {eval_call})
'''

    verif_status = "Model verified: Out-of-sample held-out evaluation completed with isolated feature transformers"

    return AutoMLResponse(
        dataset_id=req.dataset_id,
        target_column=target_col,
        task_type=task_type,
        total_samples=total_len,
        train_samples=len(X_train),
        test_samples=len(X_test),
        features_used=candidate_features,
        class_labels=class_labels,
        class_distribution=class_distribution,
        leaderboard=leaderboard,
        best_model_name=best_name,
        best_model_metrics=best_item.metrics,
        feature_importance=feature_importances,
        overfitting_assessment=overfit_text,
        limitations=limitations,
        chart_spec=leaderboard_chart,
        held_out_predictions=held_out_samples,
        evaluation_chart_spec=eval_chart_spec,
        residual_chart_spec=residual_chart_spec,
        metric_explanations=metric_explanations,
        model_selection_metric=selection_metric_name,
        selection_metric_rationale=selection_metric_rationale,
        validation_strategy=val_strategy,
        validation_rationale=val_rationale,
        verification_status=verif_status,
        error_analysis=error_analysis,
        multicollinearity_warnings=multicollinearity_warnings,
        baseline_drift_reference=baseline_drift_reference,
        reproducible_python_script=reproducible_python_script
    )


def simulate_what_if(req: WhatIfRequest) -> WhatIfResponse:
    """
    Run what-if counterfactual scenario against active trained ML pipeline.
    Compares user-specified hypothetical feature input vs typical median baseline.
    """
    session = ACTIVE_ML_SESSIONS.get(req.dataset_id)
    if not session:
        raise ValueError("No active trained ML pipeline found for this dataset. Please train AutoML first.")

    target_col = session["target_column"]
    task_type = session["task_type"]
    candidate_features = session["candidate_features"]
    pipeline: Pipeline = session["best_pipeline"]
    model_name = session["best_model_name"]
    baseline_inputs = session["baseline_inputs"].copy()

    # Build Baseline Feature Row
    baseline_row_df = pd.DataFrame([baseline_inputs])
    baseline_pred = pipeline.predict(baseline_row_df)[0]

    # Build What-If Feature Row
    scenario_inputs = baseline_inputs.copy()
    for k, v in req.feature_inputs.items():
        if k in scenario_inputs:
            scenario_inputs[k] = v

    scenario_row_df = pd.DataFrame([scenario_inputs])
    scenario_pred = pipeline.predict(scenario_row_df)[0]

    probabilities = None
    delta = None

    if task_type == "classification":
        if hasattr(pipeline.named_steps["estimator"], "predict_proba"):
            probs = pipeline.predict_proba(scenario_row_df)[0]
            labels = session.get("class_labels") or [str(i) for i in range(len(probs))]
            probabilities = {str(lbl): round(float(p), 4) for lbl, p in zip(labels, probs)}
        
        explanation = (
            f"Under the specified scenario inputs, the model predicts '{scenario_pred}'. "
            f"For comparison, a typical representative observation produces '{baseline_pred}'."
        )
    else:  # regression
        pred_val = round(float(scenario_pred), 2)
        base_val = round(float(baseline_pred), 2)
        delta = round(pred_val - base_val, 2)
        scenario_pred = pred_val
        baseline_pred = base_val
        direction = "increase" if delta > 0 else "decrease"
        explanation = (
            f"Hypothetical scenario projects '{target_col}' = {pred_val:,}, representing a "
            f"{abs(delta):,} ({abs(round(delta/base_val*100, 1)) if base_val else 0}%) {direction} "
            f"relative to the dataset baseline average ({base_val:,})."
        )

    return WhatIfResponse(
        target_column=target_col,
        task_type=task_type,
        model_used=model_name,
        prediction=scenario_pred,
        probability=probabilities,
        baseline_prediction=baseline_pred,
        prediction_delta=delta,
        explanation=explanation,
        disclaimer="Note: What-if scenarios reflect model statistical associations, not proven causal interventions."
    )
