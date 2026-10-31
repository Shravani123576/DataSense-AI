export interface DatasetMetadata {
  id: string;
  name: string;
  filename: string;
  rows: number;
  columns: number;
  memory_kb: number;
  duplicate_rows: number;
  missing_cells: number;
  missing_percentage: number;
  column_names: string[];
}

export interface TopCategory {
  value: string;
  count: number;
  percentage: number;
}

export interface ColumnStat {
  column: string;
  dtype: string;
  inferred_type: 'numeric' | 'categorical' | 'datetime' | 'boolean' | 'text';
  null_count: number;
  null_percentage: number;
  unique_count: number;
  unique_percentage: number;
  sample_values: (string | number | boolean | null)[];
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  median?: number | null;
  std?: number | null;
  q25?: number | null;
  q75?: number | null;
  iqr?: number | null;
  skewness?: number | null;
  outlier_count?: number | null;
  outlier_percentage?: number | null;
  top_categories?: TopCategory[] | null;
}

export interface CorrelationPair {
  col1: string;
  col2: string;
  correlation: number;
}

export interface HealthScoreBreakdown {
  overall_score: number;
  completeness_score: number;
  uniqueness_score: number;
  outlier_score: number;
  consistency_score: number;
  grade: string;
  badges: string[];
  recommendations: string[];
}

export interface RecommendedNextStep {
  action_type: 'clean' | 'analyze' | 'predict' | 'export';
  title: string;
  description: string;
  reason: string;
  target_tab: 'clean' | 'charts' | 'predict' | 'export';
  priority: 'high' | 'medium' | 'low';
  target_column?: string | null;
}

export interface ProfileReport {
  metadata: DatasetMetadata;
  columns: ColumnStat[];
  health: HealthScoreBreakdown;
  correlation_matrix: Record<string, Record<string, number>>;
  top_correlations: CorrelationPair[];
  preview_rows: Record<string, any>[];
  suggested_questions: string[];
  is_restaurant_dataset?: boolean;
  restaurant_insights?: Record<string, any> | null;
  recommended_next_step?: RecommendedNextStep | null;
}

export interface SampleDatasetInfo {
  id: string;
  name: string;
  description: string;
  category: string;
  rows: number;
  columns: number;
  tags: string[];
}

export interface ChartDataPoint {
  x: string | number;
  y: number;
}

export interface ChartSpec {
  chart_type: 'bar' | 'line' | 'scatter' | 'pie' | 'histogram' | 'box';
  title: string;
  x_axis?: string | null;
  y_axis?: string | null;
  labels?: string[] | null;
  series?: any[] | null;
  data?: ChartDataPoint[] | null;
  description?: string | null;
  plotly_spec?: Record<string, any> | null;
  data_stage?: 'raw' | 'cleaned' | 'analysis_ready' | 'model_ready_held_out';
  rows_used?: number | null;
  transformations?: string | null;
  exclusions?: string | null;
  orientation?: 'v' | 'h' | null;
  is_subset?: boolean;
  total_categories?: number | null;
  displayed_categories?: number | null;
  sort_order?: 'desc' | 'asc' | 'none' | null;
  low_frequency_note?: string | null;
  aggregation?: string | null;
}


export interface QueryResponse {
  question: string;
  answer_text: string;
  code_executed: string;
  result_type: string;
  summary_figures: Record<string, any>;
  table_result?: Record<string, any>[] | null;
  table_columns?: string[] | null;
  chart_spec?: ChartSpec | null;
  follow_up_suggestions: string[];
  execution_time_ms: number;
  grounded: boolean;
  error?: string | null;
}

export interface DatasetPreviewData {
  dataset_id: string;
  total_rows: number;
  total_columns: number;
  page: number;
  page_size: number;
  columns: string[];
  rows: Record<string, any>[];
}

// ==========================================
// Stage 2: Auditable Cleaning Types
// ==========================================

export interface CleaningRecommendation {
  id: string;
  title: string;
  column?: string | null;
  rule: 'deduplicate' | 'impute_missing' | 'normalize_categories' | 'cap_outliers';
  severity: 'high' | 'medium' | 'low';
  evidence: string;
  affected_rows: number;
  affected_percentage: number;
  recommended_action: string;
  strategy_options: string[];
}

export interface CleaningActionRequest {
  dataset_id: string;
  rule: string;
  column?: string | null;
  strategy?: string;
  fill_value?: any;
}

export interface CleaningAuditRecord {
  id: string;
  timestamp: string;
  rule: string;
  column?: string | null;
  affected_rows: number;
  affected_percentage: number;
  strategy_used: string;
  transformation_summary: string;
  original_sample: any[];
  cleaned_sample: any[];
  reason: string;
}

export interface CleaningSummaryResponse {
  dataset_id: string;
  total_records: number;
  total_cleaned_actions_applied: number;
  audit_history: CleaningAuditRecord[];
  current_health: HealthScoreBreakdown;
  remaining_recommendations: CleaningRecommendation[];
  cleaned_filename: string;
}

// ==========================================
export interface DatasetVersionInfo {
  version_id: number;
  parent_version_id: number;
  timestamp: string;
  operation: string;
  description: string;
  affected_rows: number;
  affected_cols: number;
  total_rows: number;
  total_cols: number;
}

export interface CleaningProposal {
  proposal_id: string;
  dataset_id: string;
  base_version_id?: number | null;
  rule: string;
  column?: string | null;
  strategy?: string;
  fill_value?: any;
  reason: string;
  is_consequential: boolean;
  affected_rows: number;
  affected_percentage: number;
  transformation_summary: string;
  diff_preview?: any;
  created_at: string;
  status: 'pending' | 'approved' | 'rejected';
}

export interface ProposalResponse {
  status: string;
  proposal: CleaningProposal;
  message: string;
}

// ==========================================
// Stage 4: AutoML & What-If Types
// ==========================================

export interface AutoMLRequest {
  dataset_id: string;
  target_column: string;
  task_type?: 'auto' | 'classification' | 'regression';
  test_size?: number;
  metric?: string;
  validation_strategy?: 'auto' | 'kfold_cv' | 'stratified_holdout' | 'chronological';
  time_column?: string | null;
  selected_features?: string[];
  algorithm_filter?: string[];
}

export interface ModelLeaderboardItem {
  model_name: string;
  algorithm: string;
  is_baseline: boolean;
  train_score: number;
  val_score?: number;
  test_score?: number | null;
  cv_scores_mean?: number | null;
  cv_scores_std?: number | null;
  metrics: Record<string, number>;
  overfitting_gap: number;
  training_time_ms: number;
  status: string;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
  percentage: number;
}

export interface HeldOutPredictionSample {
  row_index: number;
  actual: any;
  predicted: any;
  error_or_residual?: number | null;
  is_correct?: boolean | null;
  features: Record<string, any>;
}

export interface AutoMLResponse {
  dataset_id: string;
  target_column: string;
  task_type: 'classification' | 'regression';
  total_samples: number;
  train_samples: number;
  test_samples: number;
  features_used: string[];
  class_labels?: string[] | null;
  class_distribution?: Record<string, number> | null;
  leaderboard: ModelLeaderboardItem[];
  best_model_name: string;
  best_model_metrics: Record<string, number>;
  feature_importance: FeatureImportanceItem[];
  overfitting_assessment: string;
  limitations: string[];
  chart_spec?: ChartSpec | null;
  held_out_predictions?: HeldOutPredictionSample[] | null;
  evaluation_chart_spec?: ChartSpec | null;
  residual_chart_spec?: ChartSpec | null;
  metric_explanations?: Record<string, string> | null;
  validation_strategy?: string | null;
  verification_status?: string | null;
}

export interface WhatIfRequest {
  dataset_id: string;
  target_column: string;
  feature_inputs: Record<string, any>;
  model_name?: string | null;
}

export interface WhatIfResponse {
  target_column: string;
  task_type: string;
  model_used: string;
  prediction: any;
  probability?: Record<string, number> | null;
  baseline_prediction: any;
  prediction_delta?: number | null;
  explanation: string;
  disclaimer: string;
}

// ==========================================
// Stage 3: Agent Orchestration Types
// ==========================================

export interface PlanStep {
  id: string;
  step_number: number;
  title: string;
  tool_name: string;
  parameters: Record<string, any>;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'clarification_needed';
  result_summary?: string | null;
  execution_time_ms?: number | null;
  error?: string | null;
}

export interface AgentPlan {
  goal: string;
  steps: PlanStep[];
  current_step_index: number;
  is_completed: boolean;
  status: 'created' | 'in_progress' | 'completed' | 'paused_for_clarification' | 'failed';
}

export interface AgentGoalRequest {
  dataset_id: string;
  goal: string;
  api_key?: string | null;
  auto_execute?: boolean;
  clarification_response?: string | null;
}

export interface AgentGoalResponse {
  goal: string;
  dataset_id: string;
  plan: AgentPlan;
  summary_explanation: string;
  findings: string[];
  artifacts: Record<string, any>[];
  requires_clarification: boolean;
  clarification_question?: string | null;
  clarification_options?: string[] | null;
  follow_up_suggestions: string[];
  execution_time_ms: number;
}

// ==========================================
// Stage 6: Executive Reporting Types
// ==========================================

export interface ExecutiveReportRequest {
  dataset_id: string;
  objective?: string | null;
  include_audit?: boolean;
  include_ml?: boolean;
}

export interface ExecutiveReportResponse {
  dataset_id: string;
  dataset_name: string;
  objective: string;
  generated_at: string;
  markdown_content: string;
  html_content: string;
  executive_summary: string;
  recommendations: string[];
  limitations: string[];
}

// ==========================================
// Multi-File Ingestion & Table Relationships
// ==========================================

export interface WorksheetInfo {
  sheet_name: string;
  rows: number;
  columns: number;
  column_names: string[];
  is_empty: boolean;
  preview_rows: Record<string, any>[];
}

export interface WorkbookInspectResponse {
  filename: string;
  total_sheets: number;
  sheets: WorksheetInfo[];
}

export interface TableInventoryItem {
  table_id: string;
  filename: string;
  sheet_name?: string | null;
  rows: number;
  columns: number;
  column_names: string[];
  inferred_types: Record<string, string>;
  parsing_status: string;
  preview_rows: Record<string, any>[];
}

export interface MultiFileUploadResponse {
  session_id: string;
  total_tables: number;
  tables: TableInventoryItem[];
  warnings: string[];
}

export interface CandidateJoinKey {
  table_a_column: string;
  table_b_column: string;
  data_type_a: string;
  data_type_b: string;
  is_unique_a: boolean;
  is_unique_b: boolean;
  overlap_count: number;
  overlap_percentage_a: number;
  overlap_percentage_b: number;
  relationship_type: string;
  confidence: string;
  explanation: string;
}

export interface TableRelationshipItem {
  table_a_id: string;
  table_a_name: string;
  table_b_id: string;
  table_b_name: string;
  status: 'related' | 'independent' | 'ambiguous';
  candidate_keys: CandidateJoinKey[];
  evidence_summary: string;
}

export interface RelationshipDetectionResponse {
  relationships: TableRelationshipItem[];
}

export interface JoinPreviewRequest {
  table_a_id: string;
  table_b_id: string;
  join_column_a: string;
  join_column_b: string;
  join_type?: string;
}

export interface JoinPreviewResponse {
  table_a_name: string;
  table_b_name: string;
  join_column_a: string;
  join_column_b: string;
  join_type: string;
  table_a_rows: number;
  table_b_rows: number;
  expected_rows: number;
  unmatched_rows_a: number;
  unmatched_rows_b: number;
  cartesian_multiplication_risk: boolean;
  risk_warning?: string | null;
  columns_before_join: string[];
  columns_after_join: string[];
  preview_rows: Record<string, any>[];
}

export interface JoinApplyRequest {
  table_a_id: string;
  table_b_id: string;
  join_column_a: string;
  join_column_b: string;
  join_type?: string;
  new_dataset_name?: string | null;
}

