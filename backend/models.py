import uuid
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
    record_count: Optional[int] = None
    record_count_label: Optional[str] = None
    partial_rows_excluded: Optional[int] = 0
    total_raw_rows: Optional[int] = None
    missing_monthly_cells: Optional[int] = None


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
    orientation: Optional[str] = 'v'  # 'v' (vertical) or 'h' (horizontal)
    is_subset: Optional[bool] = False
    total_categories: Optional[int] = None
    displayed_categories: Optional[int] = None
    sort_order: Optional[str] = 'desc'  # 'desc', 'asc', 'none'
    low_frequency_note: Optional[str] = None


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
    orientation: Optional[str] = 'auto'  # 'auto', 'h', 'v'
    sort_order: Optional[str] = 'desc'  # 'desc', 'asc', 'none'
    top_n: Optional[int] = 20  # 10, 20, 50, 0 (0 means All)


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
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str
    rule: str
    column: Optional[str] = None
    affected_rows: int = 0
    affected_percentage: float = 0.0
    strategy_used: str = "filter"
    transformation_summary: str = ""
    original_sample: List[Any] = Field(default_factory=list)
    cleaned_sample: List[Any] = Field(default_factory=list)
    reason: str = ""


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
    validation_strategy: Optional[str] = 'auto'  # 'auto', 'kfold_cv', 'stratified_holdout', 'chronological'
    time_column: Optional[str] = None  # Chronological holdout validation column
    selected_features: Optional[List[str]] = None  # User-selected predictor features
    algorithm_filter: Optional[List[str]] = None   # Selected algorithms to benchmark


class ModelLeaderboardItem(BaseModel):
    model_name: str
    algorithm: str
    is_baseline: bool
    train_score: float
    val_score: float = 0.0  # Validation/Cross-Validation score used for candidate selection
    test_score: Optional[float] = None  # Final evaluation score on untouched held-out test split (selected champion model only)
    cv_scores_mean: Optional[float] = None
    cv_scores_std: Optional[float] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    val_metrics: Optional[Dict[str, float]] = None
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
    residual_chart_spec: Optional[ChartSpec] = None
    metric_explanations: Optional[Dict[str, str]] = None
    model_selection_metric: Optional[str] = "Validation Cross-Validation Score (Development Set)"
    selection_metric_rationale: Optional[str] = None
    validation_strategy: Optional[str] = "Held-out test split (isolated preprocessing)"
    validation_rationale: Optional[str] = None
    verification_status: Optional[str] = "Model evaluated on unseen holdout test split"
    permutation_feature_importance: Optional[List[Dict[str, Any]]] = None
    error_analysis: Optional[Dict[str, Any]] = None
    multicollinearity_warnings: Optional[List[Dict[str, Any]]] = None
    baseline_drift_reference: Optional[Dict[str, Any]] = None
    reproducible_python_script: Optional[str] = None


# ==========================================
# Versioned Transformation & Proposal Schemas
# ==========================================

class DatasetVersionInfo(BaseModel):
    version_id: int
    parent_version_id: Optional[int]
    timestamp: str
    operation: str
    description: str
    affected_rows: int
    affected_cols: int
    total_rows: int
    total_cols: int


class CleaningProposal(BaseModel):
    proposal_id: str
    dataset_id: str
    base_version_id: Optional[int] = None  # Active dataset version when proposal was generated
    rule: str
    column: Optional[str] = None
    strategy: Optional[str] = 'median'
    fill_value: Optional[Any] = None
    reason: str
    is_consequential: bool = False
    affected_rows: int
    affected_percentage: float
    transformation_summary: str
    diff_preview: Optional[Dict[str, Any]] = None
    created_at: str
    status: str = "pending"  # "pending", "approved", "rejected"


class ProposalResponse(BaseModel):
    status: str
    proposal: CleaningProposal
    message: str


class CoercionAuditItem(BaseModel):
    column: str
    original_dtype: str
    coerced_dtype: str
    affected_rows: int
    failed_coercions: int
    sample_original: List[Any] = Field(default_factory=list)
    sample_coerced: List[Any] = Field(default_factory=list)



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


class AgentTraceStep(BaseModel):
    step: int
    tool: str
    input: Dict[str, Any] = Field(default_factory=dict)
    output_summary: str = ""
    duration_ms: float = 0.0
    error: Optional[str] = None


class StructuredAnalystAnswer(BaseModel):
    finding: str
    evidence: List[str] = Field(default_factory=list)
    method: str = ""
    limitations: List[str] = Field(default_factory=list)
    next_step: Optional[str] = None


class GraphIntelligenceAskRequest(BaseModel):
    dataset_id: str
    chart_spec: Optional[Dict[str, Any]] = None
    x_column: str
    y_column: Optional[str] = None
    chart_type: str
    question: str
    aggregation: Optional[str] = "mean"
    filters: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    conversation_id: Optional[str] = None
    chart_signature: Optional[str] = None
    dataset_version: Optional[int] = None
    api_key: Optional[str] = None


class GraphIntelligenceAskResponse(BaseModel):
    answer: StructuredAnalystAnswer
    trace: List[AgentTraceStep] = Field(default_factory=list)
    mode: str = Field(default="fallback", description="'llm' or 'fallback'")
    grounded: bool = Field(default=True)
    warnings: List[str] = Field(default_factory=list)

    # Backward-compatible fields
    findings: List[GraphFindingItem] = Field(default_factory=list)
    computed_evidence: Dict[str, Any] = Field(default_factory=dict)
    methodology: str = ""
    limitations: List[str] = Field(default_factory=list)
    suggested_next_step: Optional[str] = None
    tools_invoked: List[str] = Field(default_factory=list)
    trace_id: Optional[str] = None
    execution_mode: str = Field(default="deterministic_fallback", description="'llm' or 'deterministic_fallback'")
    disclaimer: str = "Patterns and correlations displayed represent observed sample associations and must not be interpreted as causal mechanisms without controlled experimental evidence."


class ToolExecutionApiRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    dataset_id: Optional[str] = None


# ==========================================
# Multi-File Ingestion & Table Inventory Models
# ==========================================

class WorksheetInfo(BaseModel):
    sheet_name: str
    rows: int
    columns: int
    column_names: List[str]
    is_empty: bool
    preview_rows: List[Dict[str, Any]] = Field(default_factory=list)


class WorkbookInspectResponse(BaseModel):
    filename: str
    total_sheets: int
    sheets: List[WorksheetInfo]


class TableInventoryItem(BaseModel):
    table_id: str
    filename: str
    sheet_name: Optional[str] = None
    rows: int
    columns: int
    column_names: List[str]
    inferred_types: Dict[str, str] = Field(default_factory=dict)
    parsing_status: str  # "success" or error message
    preview_rows: List[Dict[str, Any]] = Field(default_factory=list)


class MultiFileUploadResponse(BaseModel):
    session_id: str
    total_tables: int
    tables: List[TableInventoryItem]
    warnings: List[str] = Field(default_factory=list)


# ==========================================
# Related Tables & Join Preview Models
# ==========================================

class CandidateJoinKey(BaseModel):
    table_a_column: str
    table_b_column: str
    data_type_a: str
    data_type_b: str
    is_unique_a: bool
    is_unique_b: bool
    overlap_count: int
    overlap_percentage_a: float
    overlap_percentage_b: float
    relationship_type: str  # "one-to-one", "one-to-many", "many-to-one", "many-to-many"
    confidence: str  # "high", "moderate", "low"
    explanation: str


class TableRelationshipItem(BaseModel):
    table_a_id: str
    table_a_name: str
    table_b_id: str
    table_b_name: str
    status: str  # "related", "independent", "ambiguous"
    candidate_keys: List[CandidateJoinKey] = Field(default_factory=list)
    evidence_summary: str


class RelationshipDetectionResponse(BaseModel):
    relationships: List[TableRelationshipItem]


class JoinPreviewRequest(BaseModel):
    table_a_id: str
    table_b_id: str
    join_column_a: str
    join_column_b: str
    join_type: str = "inner"  # "inner", "left", "right", "outer"


class JoinPreviewResponse(BaseModel):
    table_a_name: str
    table_b_name: str
    join_column_a: str
    join_column_b: str
    join_type: str
    table_a_rows: int
    table_b_rows: int
    expected_rows: int
    unmatched_rows_a: int
    unmatched_rows_b: int
    cartesian_multiplication_risk: bool
    risk_warning: Optional[str] = None
    columns_before_join: List[str]
    columns_after_join: List[str]
    preview_rows: List[Dict[str, Any]] = Field(default_factory=list)


class JoinApplyRequest(BaseModel):
    table_a_id: str
    table_b_id: str
    join_column_a: str
    join_column_b: str
    join_type: str = "inner"
    new_dataset_name: Optional[str] = None


class AgentChatRequest(BaseModel):
    dataset_id: Optional[str] = None
    dataset: Optional[str] = None
    message: Optional[str] = None
    question: Optional[str] = None
    chart_type: Optional[str] = "line"
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    api_key: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

