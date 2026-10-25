from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    id: str
    name: str
    filename: str
    rows: int
    columns: int
    memory_kb: float
    duplicate_rows: int
    missing_cells: int
    missing_percentage: float
    column_names: List[str]


class TopCategory(BaseModel):
    value: str
    count: int
    percentage: float


class ColumnStat(BaseModel):
    column: str
    dtype: str
    inferred_type: str  # 'numeric', 'categorical', 'datetime', 'boolean', 'text'
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: List[Any]
    # Numeric stats
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    outlier_count: Optional[int] = None
    outlier_percentage: Optional[float] = None
    # Categorical stats
    top_categories: Optional[List[TopCategory]] = None


class CorrelationPair(BaseModel):
    col1: str
    col2: str
    correlation: float


class HealthScoreBreakdown(BaseModel):
    overall_score: int  # 0 to 100
    completeness_score: int
    uniqueness_score: int
    outlier_score: int
    consistency_score: int
    grade: str  # 'A+', 'A', 'B', 'C', 'Needs Attention'
    badges: List[str]
    recommendations: List[str]


class RecommendedNextStep(BaseModel):
    action_type: str  # 'clean', 'analyze', 'predict', 'export'
    title: str
    description: str
    reason: str
    target_tab: str  # 'clean', 'charts', 'predict', 'export'
    priority: str  # 'high', 'medium', 'low'
    target_column: Optional[str] = None


class ProfileReport(BaseModel):
    metadata: DatasetMetadata
    columns: List[ColumnStat]
    health: HealthScoreBreakdown
    correlation_matrix: Dict[str, Dict[str, float]]
    top_correlations: List[CorrelationPair]
    preview_rows: List[Dict[str, Any]]
    suggested_questions: List[str]
    is_restaurant_dataset: bool = False
    restaurant_insights: Optional[Dict[str, Any]] = None
    recommended_next_step: Optional[RecommendedNextStep] = None


class SampleDatasetInfo(BaseModel):
    id: str
    name: str
    description: str
    category: str
    rows: int
    columns: int
    tags: List[str]


class QueryRequest(BaseModel):
    question: str
    dataset_id: str
    api_key: Optional[str] = None


class ChartSpec(BaseModel):
    chart_type: str  # 'bar', 'line', 'scatter', 'pie', 'histogram', 'box'
    title: str
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    labels: Optional[List[str]] = None
    series: Optional[List[Dict[str, Any]]] = None
    data: Optional[List[Dict[str, Any]]] = None
    description: Optional[str] = None
    plotly_spec: Optional[Dict[str, Any]] = None
    data_stage: Optional[str] = 'analysis_ready'  # 'raw', 'cleaned', 'analysis_ready', 'model_ready_held_out'
    rows_used: Optional[int] = None
    transformations: Optional[str] = None
    exclusions: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer_text: str
    code_executed: str
    result_type: str  # 'scalar', 'table', 'chart', 'summary'
    summary_figures: Dict[str, Any]
    table_result: Optional[List[Dict[str, Any]]] = None
    table_columns: Optional[List[str]] = None
    chart_spec: Optional[ChartSpec] = None
    follow_up_suggestions: List[str]
    execution_time_ms: float
    grounded: bool = True
    error: Optional[str] = None


class CustomChartRequest(BaseModel):
    dataset_id: str
    x_column: str
    y_column: Optional[str] = None
    chart_type: str  # 'bar', 'line', 'scatter', 'histogram', 'box', 'pie'
    aggregation: Optional[str] = 'mean'  # 'mean', 'sum', 'count', 'min', 'max'
    limit: Optional[int] = 20


# ==========================================
# Auditable Data Cleaning Models (Stage 2)
# ==========================================

class CleaningRecommendation(BaseModel):
    id: str
    title: str
    column: Optional[str] = None
    rule: str  # 'deduplicate', 'impute_missing', 'normalize_categories', 'cap_outliers'
    severity: str  # 'high', 'medium', 'low'
    evidence: str
    affected_rows: int
    affected_percentage: float
    recommended_action: str
    strategy_options: List[str] = Field(default_factory=list)


class CleaningActionRequest(BaseModel):
    dataset_id: str
    rule: str  # 'deduplicate', 'impute_missing', 'normalize_categories', 'cap_outliers'
    column: Optional[str] = None
    strategy: Optional[str] = 'median'  # 'median', 'mean', 'mode', 'drop_rows', 'cap_iqr', 'strip_whitespace'
    fill_value: Optional[Any] = None


class CleaningAuditRecord(BaseModel):
    id: str
    timestamp: str
    rule: str
    column: Optional[str]
    affected_rows: int
    affected_percentage: float
    strategy_used: str
    transformation_summary: str
    original_sample: List[Any] = Field(default_factory=list)
    cleaned_sample: List[Any] = Field(default_factory=list)
    reason: str


class CleaningSummaryResponse(BaseModel):
    dataset_id: str
    total_records: int
    total_cleaned_actions_applied: int
    audit_history: List[CleaningAuditRecord]
    current_health: HealthScoreBreakdown
    remaining_recommendations: List[CleaningRecommendation]
    cleaned_filename: str


# ==========================================
# Automated Machine Learning Models (Stage 4)
# ==========================================

class AutoMLRequest(BaseModel):
    dataset_id: str
    target_column: str
    task_type: Optional[str] = 'auto'  # 'auto', 'classification', 'regression'
    test_size: Optional[float] = 0.2
    metric: Optional[str] = 'auto'


class ModelLeaderboardItem(BaseModel):
    model_name: str
    algorithm: str
    is_baseline: bool
    train_score: float
    test_score: float
    metrics: Dict[str, float]
    overfitting_gap: float
    training_time_ms: float
    status: str = "trained"


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    percentage: float


class HeldOutPredictionSample(BaseModel):
    row_index: int
    actual: Any
    predicted: Any
    error_or_residual: Optional[float] = None
    is_correct: Optional[bool] = None
    features: Dict[str, Any] = Field(default_factory=dict)


class AutoMLResponse(BaseModel):
    dataset_id: str
    target_column: str
    task_type: str  # 'classification' or 'regression'
    total_samples: int
    train_samples: int
    test_samples: int
    features_used: List[str]
    class_labels: Optional[List[str]] = None
    class_distribution: Optional[Dict[str, int]] = None
    leaderboard: List[ModelLeaderboardItem]
    best_model_name: str
    best_model_metrics: Dict[str, float]
    feature_importance: List[FeatureImportanceItem]
    overfitting_assessment: str
    limitations: List[str]
    chart_spec: Optional[ChartSpec] = None
    held_out_predictions: Optional[List[HeldOutPredictionSample]] = None
    evaluation_chart_spec: Optional[ChartSpec] = None
    metric_explanations: Optional[Dict[str, str]] = None
    validation_strategy: Optional[str] = "Held-out test split (isolated preprocessing)"
    verification_status: Optional[str] = "Model evaluated on unseen holdout test split"
    permutation_feature_importance: Optional[List[Dict[str, Any]]] = None
    error_analysis: Optional[Dict[str, Any]] = None
    multicollinearity_warnings: Optional[List[Dict[str, Any]]] = None
    baseline_drift_reference: Optional[Dict[str, Any]] = None
    reproducible_python_script: Optional[str] = None


class WhatIfRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_inputs: Dict[str, Any]
    model_name: Optional[str] = None


class WhatIfResponse(BaseModel):
    target_column: str
    task_type: str
    model_used: str
    prediction: Any
    probability: Optional[Dict[str, float]] = None
    baseline_prediction: Any
    prediction_delta: Optional[float] = None
    explanation: str
    disclaimer: str = "Note: What-if scenarios reflect model statistical associations, not proven causal interventions."


# ==========================================
# Agent Orchestration Models (Stage 3)
# ==========================================

class PlanStep(BaseModel):
    id: str
    step_number: int
    title: str
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"  # 'pending', 'running', 'completed', 'failed', 'clarification_needed'
    result_summary: Optional[str] = None
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None


class AgentPlan(BaseModel):
    goal: str
    steps: List[PlanStep]
    current_step_index: int = 0
    is_completed: bool = False
    status: str = "created"  # 'created', 'in_progress', 'completed', 'paused_for_clarification', 'failed'


class AgentGoalRequest(BaseModel):
    dataset_id: str
    goal: str
    api_key: Optional[str] = None
    auto_execute: bool = True
    clarification_response: Optional[str] = None


class AgentGoalResponse(BaseModel):
    goal: str
    dataset_id: str
    plan: AgentPlan
    summary_explanation: str
    findings: List[str] = Field(default_factory=list)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_options: Optional[List[str]] = None
    follow_up_suggestions: List[str] = Field(default_factory=list)
    execution_time_ms: float


# ==========================================
# Executive Report Models (Stage 6)
# ==========================================

class ExecutiveReportRequest(BaseModel):
    dataset_id: str
    objective: Optional[str] = None
    include_audit: bool = True
    include_ml: bool = True


class ExecutiveReportResponse(BaseModel):
    dataset_id: str
    dataset_name: str
    objective: str
    generated_at: str
    markdown_content: str
    html_content: str
    executive_summary: str
    recommendations: List[str]
    limitations: List[str]


# ==========================================
# Statistical Hypothesis Testing Models (V4)
# ==========================================

class HypothesisTestRequest(BaseModel):
    dataset_id: str
    test_type: Optional[str] = "auto"  # 'auto', 'two_sample_t', 'anova', 'chi_square'
    variable_x: str  # group column or first category
    variable_y: Optional[str] = None  # metric column or second category
    confidence_level: Optional[float] = 0.95


class AssumptionCheck(BaseModel):
    name: str  # 'Normality Check', 'Equal Variance (Homoscedasticity)', 'Expected Frequencies'
    passed: bool
    details: str
    metric_value: Optional[float] = None
    p_value: Optional[float] = None


class HypothesisTestResponse(BaseModel):
    test_name: str
    test_category: str  # 'parametric', 'non_parametric', 'categorical_independence'
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic_name: str
    test_statistic_value: float
    p_value: float
    is_statistically_significant: bool
    significance_threshold: float = 0.05
    effect_size_name: str
    effect_size_value: float
    effect_size_magnitude: str  # 'negligible', 'small', 'medium', 'large'
    assumptions: List[AssumptionCheck]
    group_summaries: Dict[str, Dict[str, Any]]
    plain_english_finding: str
    practical_implication: str
    caveat_disclaimer: str = "Statistical significance does not prove real-world importance or causal relationship."


# ==========================================
# Graph Intelligence Agent Models (V4)
# ==========================================

class GraphIntelligenceAnalyzeRequest(BaseModel):
    dataset_id: str
    chart_spec: Dict[str, Any]
    x_column: str
    y_column: Optional[str] = None
    chart_type: str
    aggregation: Optional[str] = "mean"
    filters: Optional[Dict[str, Any]] = None


class GraphFindingItem(BaseModel):
    category: str  # 'Observed Fact', 'Statistical Finding', 'Anomaly', 'Data Quality Note'
    title: str
    detail: str
    confidence: Optional[str] = "High"  # 'High', 'Moderate', 'Hypothesis'


class GraphIntelligenceResponse(BaseModel):
    summary: str
    key_findings: List[GraphFindingItem]
    anomalies_detected: List[str]
    group_comparisons: List[str]
    hidden_patterns: List[str]
    data_quality_concerns: List[str]
    suggested_followups: List[str]
    recommended_alternative_charts: List[str]
    causation_disclaimer: str = "Patterns and correlations displayed represent observed sample associations and must not be interpreted as causal mechanisms without controlled experimental evidence."


class GraphIntelligenceAskRequest(BaseModel):
    dataset_id: str
    chart_spec: Dict[str, Any]
    x_column: str
    y_column: Optional[str] = None
    chart_type: str
    question: str
    api_key: Optional[str] = None
