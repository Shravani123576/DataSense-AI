import type {
  ProfileReport,
  SampleDatasetInfo,
  QueryResponse,
  ChartSpec,
  DatasetPreviewData,
  CleaningRecommendation,
  CleaningActionRequest,
  CleaningSummaryResponse,
  AutoMLRequest,
  AutoMLResponse,
  WhatIfRequest,
  WhatIfResponse,
  AgentGoalRequest,
  AgentGoalResponse,
  ExecutiveReportRequest,
  ExecutiveReportResponse,
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

export async function uploadDatasetFile(file: File): Promise<ProfileReport> {
  const formData = new FormData();
  formData.append('file', file);

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
  return `${API_BASE}/dataset/${encodeURIComponent(datasetId)}/export/cleaned?deduplicate=true&impute_numeric=true`;
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


