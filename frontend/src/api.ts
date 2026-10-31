import type {
  ProfileReport,
  SampleDatasetInfo,
  QueryResponse,
  ChartSpec,
  DatasetPreviewData,
  CleaningRecommendation,
  CleaningActionRequest,
  CleaningSummaryResponse,
  ProposalResponse,
  DatasetVersionInfo,
  AutoMLRequest,
  AutoMLResponse,
  WhatIfRequest,
  WhatIfResponse,
  AgentGoalRequest,
  AgentGoalResponse,
  ExecutiveReportRequest,
  ExecutiveReportResponse,
  WorkbookInspectResponse,
  MultiFileUploadResponse,
  RelationshipDetectionResponse,
  JoinPreviewRequest,
  JoinPreviewResponse,
  JoinApplyRequest,
} from './types';

const API_BASE = '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function fetchSamples(): Promise<SampleDatasetInfo[]> {
  const res = await fetch(`${API_BASE}/samples`);
  if (!res.ok) throw new Error('Failed to load sample datasets');
  return res.json();
}

export async function loadSampleDataset(sampleId: string): Promise<ProfileReport> {
  const res = await fetch(`${API_BASE}/samples/load?sample_id=${encodeURIComponent(sampleId)}`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to load sample dataset');
  }
  return res.json();
}

export async function uploadDatasetFile(file: File, sheetName?: string): Promise<ProfileReport> {
  const formData = new FormData();
  formData.append('file', file);
  if (sheetName) {
    formData.append('sheet_name', sheetName);
  }

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to upload dataset');
  }
  return res.json();
}

export async function inspectExcelWorksheets(file: File): Promise<WorkbookInspectResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/upload/inspect-sheets`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to inspect Excel worksheets');
  }
  return res.json();
}

export async function uploadMultiDatasetFiles(files: File[]): Promise<MultiFileUploadResponse> {
  const formData = new FormData();
  for (const f of files) {
    formData.append('files', f);
  }

  const res = await fetch(`${API_BASE}/upload/multi`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to upload multiple datasets');
  }
  return res.json();
}

export async function fetchTableRelationships(
  sessionId?: string,
  tableIds?: string[]
): Promise<RelationshipDetectionResponse> {
  let url = `${API_BASE}/tables/relationships?`;
  const params: string[] = [];
  if (sessionId) params.push(`session_id=${encodeURIComponent(sessionId)}`);
  if (tableIds && tableIds.length > 0) {
    tableIds.forEach((tid) => params.push(`table_ids=${encodeURIComponent(tid)}`));
  }
  url += params.join('&');

  const res = await fetch(url, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to detect table relationships');
  }
  return res.json();
}

export async function previewTableJoin(request: JoinPreviewRequest): Promise<JoinPreviewResponse> {
  const res = await fetch(`${API_BASE}/tables/join-preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to preview table join');
  }
  return res.json();
}

export async function applyTableJoin(request: JoinApplyRequest): Promise<ProfileReport> {
  const res = await fetch(`${API_BASE}/tables/join-apply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to execute table join');
  }
  return res.json();
}

export async function fetchDatasetPreview(
  datasetId: string,
  page: number = 1,
  pageSize: number = 20,
  search?: string
): Promise<DatasetPreviewData> {
  let url = `${API_BASE}/dataset/${encodeURIComponent(datasetId)}/preview?page=${page}&page_size=${pageSize}`;
  if (search) {
    url += `&search=${encodeURIComponent(search)}`;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch dataset preview');
  return res.json();
}

export async function fetchDatasetProfile(datasetId: string): Promise<ProfileReport> {
  const res = await fetch(`${API_BASE}/dataset/${encodeURIComponent(datasetId)}/profile`);
  if (!res.ok) throw new Error('Failed to fetch dataset profile');
  return res.json();
}

export async function queryDataset(
  datasetId: string,
  question: string,
  apiKey?: string
): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_id: datasetId,
      question: question,
      api_key: apiKey || undefined,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Query execution failed');
  }
  return res.json();
}

export async function generateCustomChart(params: {
  dataset_id: string;
  x_column: string;
  y_column?: string;
  chart_type: string;
  aggregation?: string;
  limit?: number;
  orientation?: 'auto' | 'h' | 'v';
  sort_order?: 'desc' | 'asc' | 'none';
  top_n?: number;
}): Promise<ChartSpec> {
  const res = await fetch(`${API_BASE}/custom-chart`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Chart generation failed');
  }
  return res.json();
}

// ==========================================
// Stage 2: Auditable Cleaning Endpoints
// ==========================================

export async function fetchCleaningRecommendations(datasetId: string): Promise<CleaningRecommendation[]> {
  const res = await fetch(`${API_BASE}/clean/${encodeURIComponent(datasetId)}/recommendations`);
  if (!res.ok) throw new Error('Failed to fetch cleaning recommendations');
  return res.json();
}

export async function previewCleaningAction(request: CleaningActionRequest): Promise<any> {
  const res = await fetch(`${API_BASE}/clean/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Preview generation failed');
  }
  return res.json();
}

export async function applyCleaningAction(request: CleaningActionRequest): Promise<CleaningSummaryResponse> {
  const res = await fetch(`${API_BASE}/clean/apply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Cleaning action failed');
  }
  return res.json();
}

export async function fetchCleaningAudit(datasetId: string): Promise<CleaningSummaryResponse> {
  const res = await fetch(`${API_BASE}/clean/${encodeURIComponent(datasetId)}/audit`);
  if (!res.ok) throw new Error('Failed to fetch cleaning audit');
  return res.json();
}

export async function resetToRawDataset(datasetId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/clean/${encodeURIComponent(datasetId)}/reset-to-raw`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to reset dataset to raw');
  return res.json();
}

export async function proposeCleaningAction(request: CleaningActionRequest): Promise<ProposalResponse> {
  const res = await fetch(`${API_BASE}/clean/propose`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to propose cleaning transformation');
  }
  return res.json();
}

export async function approveCleaningProposal(proposalId: string): Promise<CleaningSummaryResponse> {
  const res = await fetch(`${API_BASE}/clean/approve/${encodeURIComponent(proposalId)}`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to approve cleaning proposal');
  }
  return res.json();
}

export async function rejectCleaningProposal(proposalId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/clean/reject/${encodeURIComponent(proposalId)}`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to reject cleaning proposal');
  }
  return res.json();
}

export async function fetchDatasetVersions(datasetId: string): Promise<DatasetVersionInfo[]> {
  const res = await fetch(`${API_BASE}/clean/${encodeURIComponent(datasetId)}/versions`);
  if (!res.ok) throw new Error('Failed to fetch dataset versions');
  return res.json();
}

// ==========================================
// Stage 4: AutoML & What-If Endpoints
// ==========================================

export async function trainAutoML(request: AutoMLRequest): Promise<AutoMLResponse> {
  const res = await fetch(`${API_BASE}/automl/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'AutoML tournament failed');
  }
  return res.json();
}

export async function simulateWhatIf(request: WhatIfRequest): Promise<WhatIfResponse> {
  const res = await fetch(`${API_BASE}/automl/what-if`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'What-if scenario failed');
  }
  return res.json();
}

// ==========================================
// Stage 3: Agent Orchestrator Endpoint
// ==========================================

export async function executeAgentGoal(request: AgentGoalRequest): Promise<AgentGoalResponse> {
  const res = await fetch(`${API_BASE}/agent/goal`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Agent workflow failed');
  }
  return res.json();
}

// ==========================================
// Stage 6: Executive Reporting Endpoints
// ==========================================

export async function generateExecutiveReport(request: ExecutiveReportRequest): Promise<ExecutiveReportResponse> {
  const res = await fetch(`${API_BASE}/report/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Report synthesis failed');
  }
  return res.json();
}

export function getSummaryExportUrl(datasetId: string): string {
  return `${API_BASE}/dataset/${encodeURIComponent(datasetId)}/export/summary`;
}

export function getCleanedExportUrl(datasetId: string): string {
  return `${API_BASE}/dataset/${encodeURIComponent(datasetId)}/export/cleaned`;
}

export function getRawExportUrl(datasetId: string): string {
  return `${API_BASE}/dataset/${encodeURIComponent(datasetId)}/export/raw`;
}

export function getCsvReportExportUrl(datasetId: string): string {
  return `${API_BASE}/dataset/${encodeURIComponent(datasetId)}/export/csv-report`;
}

export function getReportHtmlExportUrl(datasetId: string): string {
  return `${API_BASE}/report/${encodeURIComponent(datasetId)}/download/html`;
}

export function getReportMarkdownExportUrl(datasetId: string): string {
  return `${API_BASE}/report/${encodeURIComponent(datasetId)}/download/markdown`;
}

// ==========================================
// Phase 3: Graph Intelligence & Hypothesis Testing
// ==========================================

export interface GraphFindingItem {
  category: string;
  title: string;
  detail: string;
  confidence?: string;
}

export interface GraphIntelligenceResponse {
  summary: string;
  key_findings: GraphFindingItem[];
  anomalies_detected: string[];
  group_comparisons: string[];
  hidden_patterns: string[];
  data_quality_concerns: string[];
  suggested_followups: string[];
  recommended_alternative_charts: string[];
  causation_disclaimer: string;
}

export interface AgentTraceStep {
  step: number;
  tool: string;
  input: Record<string, any>;
  output_summary: string;
  duration_ms: number;
  error?: string | null;
}

export interface StructuredAnalystAnswer {
  finding: string;
  evidence: string[];
  method: string;
  limitations: string[];
  next_step?: string | null;
}

export interface GraphIntelligenceAskRequest {
  dataset_id: string;
  chart_spec: Record<string, any>;
  x_column: string;
  y_column?: string | null;
  chart_type: string;
  question: string;
  aggregation?: string;
  filters?: Record<string, any> | null;
  conversation_history?: Array<{ role: string; content: string }>;
  conversation_id?: string | null;
  chart_signature?: string | null;
  dataset_version?: number;
  api_key?: string | null;
}

export interface GraphIntelligenceAskResponse {
  answer: StructuredAnalystAnswer | string;
  trace: AgentTraceStep[];
  mode: 'llm' | 'fallback';
  grounded: boolean;
  warnings: string[];
  findings?: GraphFindingItem[];
  computed_evidence?: Record<string, any>;
  methodology?: string;
  limitations?: string[];
  suggested_next_step?: string | null;
  tools_invoked?: string[];
  trace_id?: string | null;
  execution_mode?: 'llm' | 'fallback' | 'deterministic_fallback';
  disclaimer?: string;
}

export interface HypothesisTestRequest {
  dataset_id: string;
  test_type?: string;
  variable_x: string;
  variable_y?: string | null;
  confidence_level?: number;
}

export interface AssumptionCheck {
  name: string;
  passed: boolean;
  details: string;
  metric_value?: number;
  p_value?: number;
}

export interface HypothesisTestResponse {
  test_name: string;
  test_category: string;
  null_hypothesis: string;
  alternative_hypothesis: string;
  test_statistic_name: string;
  test_statistic_value: number;
  p_value: number;
  is_statistically_significant: boolean;
  significance_threshold: number;
  effect_size_name: string;
  effect_size_value: number;
  effect_size_magnitude: string;
  assumptions: AssumptionCheck[];
  group_summaries: Record<string, Record<string, any>>;
  plain_english_finding: string;
  practical_implication: string;
  caveat_disclaimer: string;
}

export async function askGraphAi(request: GraphIntelligenceAskRequest): Promise<GraphIntelligenceAskResponse> {
  const resolvedKey =
    request.api_key ||
    localStorage.getItem('datasense_gemini_key') ||
    localStorage.getItem('datasense_gemini_api_key') ||
    undefined;

  const payload = {
    ...request,
    api_key: resolvedKey,
  };

  const res = await fetch(`${API_BASE}/graph/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Graph AI request failed');
  }
  return res.json();
}

export async function analyzeGraph(request: {
  dataset_id: string;
  chart_spec: Record<string, any>;
  x_column: string;
  y_column?: string | null;
  chart_type: string;
  aggregation?: string;
}): Promise<GraphIntelligenceResponse> {
  const res = await fetch(`${API_BASE}/graph/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Graph analysis failed');
  }
  return res.json();
}

export async function runHypothesisTest(request: HypothesisTestRequest): Promise<HypothesisTestResponse> {
  const res = await fetch(`${API_BASE}/statistical/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Statistical hypothesis test failed');
  }
  return res.json();
}



