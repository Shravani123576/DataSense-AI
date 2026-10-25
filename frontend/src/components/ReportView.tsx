import React, { useState, useEffect } from 'react';
import type { ProfileReport, ExecutiveReportResponse } from '../types';
import {
  generateExecutiveReport,
  getReportHtmlExportUrl,
  getReportMarkdownExportUrl,
  getCsvReportExportUrl,
} from '../api';

interface ReportViewProps {
  dataset: ProfileReport;
}

export const ReportView: React.FC<ReportViewProps> = ({ dataset }) => {
  const [objective, setObjective] = useState<string>('');
  const [report, setReport] = useState<ExecutiveReportResponse | null>(null);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const datasetId = dataset.metadata.id;

  const handleGenerate = async (customObj?: string) => {
    setIsGenerating(true);
    setErrorMsg(null);
    try {
      const res = await generateExecutiveReport({
        dataset_id: datasetId,
        objective: customObj || (objective.trim() ? objective : undefined),
        include_audit: true,
        include_ml: true,
      });
      setReport(res);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to synthesize report.');
    } finally {
      setIsGenerating(false);
    }
  };

  useEffect(() => {
    handleGenerate();
  }, [datasetId]);

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '16px 20px 48px' }}>
      {/* Header Banner */}
      <div
        style={{
          background: '#FFFFFF',
          padding: '24px 28px',
          borderRadius: '20px',
          border: '1px solid #EAE6DF',
          boxShadow: '0 2px 10px rgba(15, 61, 62, 0.04)',
          marginBottom: '24px',
        }}
      >
        <h2 style={{ fontSize: '24px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 16px' }}>
          Reports
        </h2>

        {/* Objective Input & Export Action Bar */}
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
          <input
            type="text"
            placeholder="Customize business objective (optional)..."
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
            style={{
              flex: 1,
              minWidth: '260px',
              padding: '11px 16px',
              borderRadius: '12px',
              border: '1px solid #CBD5E1',
              fontSize: '14px',
              background: '#FAF8F5',
              color: '#0F3D3E',
            }}
          />
          <button
            onClick={() => handleGenerate()}
            disabled={isGenerating}
            style={{
              background: '#0F3D3E',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '12px',
              padding: '11px 22px',
              fontSize: '15px',
              fontWeight: 600,
              cursor: isGenerating ? 'not-allowed' : 'pointer',
              opacity: isGenerating ? 0.7 : 1,
            }}
          >
            {isGenerating ? 'Synthesizing...' : 'Update Report'}
          </button>

          <div style={{ display: 'flex', gap: '10px', marginLeft: 'auto', flexWrap: 'wrap' }}>
            <a
              href={getReportHtmlExportUrl(datasetId)}
              download
              style={{
                background: '#EA580C',
                color: '#FFFFFF',
                borderRadius: '12px',
                padding: '11px 20px',
                fontSize: '15px',
                fontWeight: 600,
                textDecoration: 'none',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                boxShadow: '0 2px 8px rgba(234, 88, 12, 0.25)',
              }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              Download HTML Brief
            </a>

            <a
              href={getReportMarkdownExportUrl(datasetId)}
              download
              style={{
                background: '#FFFFFF',
                color: '#0F3D3E',
                border: '1.5px solid #0F3D3E',
                borderRadius: '12px',
                padding: '11px 20px',
                fontSize: '15px',
                fontWeight: 600,
                textDecoration: 'none',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              Download Markdown
            </a>

            <a
              href={getCsvReportExportUrl(datasetId)}
              download
              style={{
                background: '#FAF8F5',
                color: '#476D6D',
                border: '1px solid #CBD5E1',
                borderRadius: '12px',
                padding: '11px 18px',
                fontSize: '15px',
                fontWeight: 600,
                textDecoration: 'none',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              Export CSV
            </a>
          </div>
        </div>
      </div>

      {errorMsg && (
        <div
          style={{
            background: '#FEF2F2',
            color: '#991B1B',
            border: '1px solid #FECACA',
            padding: '14px 20px',
            borderRadius: '12px',
            marginBottom: '20px',
            fontSize: '15px',
          }}
        >
          {errorMsg}
        </div>
      )}

      {/* Report Preview */}
      {report && (
        <div
          style={{
            background: '#FFFFFF',
            borderRadius: '20px',
            padding: '36px',
            border: '1px solid #EAE6DF',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.03)',
          }}
        >
          {/* Executive Header */}
          <div style={{ borderBottom: '2px solid #EA580C', paddingBottom: '20px', marginBottom: '28px' }}>
            <span
              style={{
                fontSize: '12px',
                fontWeight: 700,
                color: '#EA580C',
                textTransform: 'uppercase',
                letterSpacing: '0.8px',
              }}
            >
              Executive Brief
            </span>
            <h3 style={{ fontSize: '26px', fontWeight: 800, color: '#0F3D3E', margin: '8px 0 6px' }}>
              {report.dataset_name} — Intelligence Report
            </h3>
            <div style={{ fontSize: '14px', color: '#648484' }}>
              Generated on {report.generated_at} &bull; Objective: {report.objective}
            </div>
          </div>

          {/* Quick Metrics */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: '16px',
              marginBottom: '32px',
            }}
          >
            <div style={{ background: '#FAF8F5', padding: '18px', borderRadius: '14px', textAlign: 'center', border: '1px solid #EAE6DF' }}>
              <div style={{ fontSize: '24px', fontWeight: 800, color: '#0F3D3E' }}>
                {dataset.metadata.rows.toLocaleString()}
              </div>
              <div style={{ fontSize: '13px', color: '#648484', fontWeight: 600, marginTop: '4px' }}>
                Total Records
              </div>
            </div>

            <div style={{ background: '#FAF8F5', padding: '18px', borderRadius: '14px', textAlign: 'center', border: '1px solid #EAE6DF' }}>
              <div style={{ fontSize: '24px', fontWeight: 800, color: '#0F3D3E' }}>
                {dataset.metadata.columns}
              </div>
              <div style={{ fontSize: '13px', color: '#648484', fontWeight: 600, marginTop: '4px' }}>
                Features
              </div>
            </div>

            <div style={{ background: '#FAF8F5', padding: '18px', borderRadius: '14px', textAlign: 'center', border: '1px solid #EAE6DF' }}>
              <div style={{ fontSize: '24px', fontWeight: 800, color: '#EA580C' }}>
                {dataset.health.grade} ({dataset.health.overall_score}/100)
              </div>
              <div style={{ fontSize: '13px', color: '#648484', fontWeight: 600, marginTop: '4px' }}>
                Quality Rating
              </div>
            </div>

            <div style={{ background: '#FAF8F5', padding: '18px', borderRadius: '14px', textAlign: 'center', border: '1px solid #EAE6DF' }}>
              <div style={{ fontSize: '24px', fontWeight: 800, color: '#059669' }}>
                {(100 - dataset.metadata.missing_percentage).toFixed(1)}%
              </div>
              <div style={{ fontSize: '13px', color: '#648484', fontWeight: 600, marginTop: '4px' }}>
                Completeness
              </div>
            </div>
          </div>

          {/* Actionable Recommendations */}
          <div style={{ marginBottom: '32px' }}>
            <h4 style={{ fontSize: '20px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 16px' }}>
              Strategic Recommendations
            </h4>
            <ol style={{ paddingLeft: '22px', margin: 0, display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {report.recommendations.map((rec, i) => (
                <li key={i} style={{ fontSize: '15px', color: '#0F3D3E', lineHeight: 1.6 }}>
                  {rec}
                </li>
              ))}
            </ol>
          </div>

          {/* Analytical Constraints */}
          <div>
            <h4 style={{ fontSize: '18px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 14px' }}>
              Caveats & Generalization Constraints
            </h4>
            <ul style={{ paddingLeft: '22px', margin: 0, display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {report.limitations.map((lim, i) => (
                <li key={i} style={{ fontSize: '14px', color: '#476D6D', lineHeight: 1.6 }}>
                  {lim}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};
