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
          padding: '24px',
          borderRadius: '16px',
          border: '1px solid #E2E8F0',
          boxShadow: '0 4px 16px rgba(15, 61, 62, 0.04)',
          marginBottom: '24px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span
            style={{
              fontSize: '12px',
              fontWeight: 700,
              color: '#EA580C',
              textTransform: 'uppercase',
              letterSpacing: '0.6px',
              background: '#FFF7ED',
              padding: '4px 10px',
              borderRadius: '8px',
            }}
          >
            Executive Reporting
          </span>
          <span style={{ fontSize: '13px', color: '#64748B' }}>
            Stakeholder-Ready Markdown & Styled HTML
          </span>
        </div>
        <h2 style={{ fontSize: '22px', fontWeight: 700, color: '#0F3D3E', margin: '8px 0 4px' }}>
          Executive Data Intelligence Briefing
        </h2>
        <p style={{ fontSize: '14px', color: '#64748B', margin: '0 0 20px' }}>
          Download clean, verified executive reports synthesizing dataset health, auditable transformations, predictive modeling, and strategic recommendations.
        </p>

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
              padding: '10px 16px',
              borderRadius: '10px',
              border: '1px solid #CBD5E1',
              fontSize: '13px',
              background: '#FAF8F5',
            }}
          />
          <button
            onClick={() => handleGenerate()}
            disabled={isGenerating}
            style={{
              background: '#0F3D3E',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '10px',
              padding: '10px 18px',
              fontSize: '13px',
              fontWeight: 600,
              cursor: isGenerating ? 'not-allowed' : 'pointer',
              opacity: isGenerating ? 0.7 : 1,
            }}
          >
            {isGenerating ? 'Synthesizing...' : 'Update Briefing'}
          </button>

          <div style={{ display: 'flex', gap: '8px', marginLeft: 'auto' }}>
            <a
              href={getReportHtmlExportUrl(datasetId)}
              download
              style={{
                background: '#EA580C',
                color: '#FFFFFF',
                borderRadius: '10px',
                padding: '10px 16px',
                fontSize: '13px',
                fontWeight: 600,
                textDecoration: 'none',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                boxShadow: '0 2px 6px rgba(234, 88, 12, 0.25)',
              }}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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
                borderRadius: '10px',
                padding: '10px 16px',
                fontSize: '13px',
                fontWeight: 600,
                textDecoration: 'none',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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
                background: '#F1F5F9',
                color: '#475569',
                border: '1px solid #CBD5E1',
                borderRadius: '10px',
                padding: '10px 14px',
                fontSize: '13px',
                fontWeight: 600,
                textDecoration: 'none',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
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
            padding: '14px 18px',
            borderRadius: '10px',
            marginBottom: '20px',
            fontSize: '14px',
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
            borderRadius: '16px',
            padding: '32px',
            border: '1px solid #E2E8F0',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.03)',
          }}
        >
          {/* Executive Header */}
          <div style={{ borderBottom: '2px solid #EA580C', paddingBottom: '20px', marginBottom: '24px' }}>
            <span
              style={{
                fontSize: '11px',
                fontWeight: 700,
                color: '#EA580C',
                textTransform: 'uppercase',
                letterSpacing: '0.8px',
              }}
            >
              Confidential Executive Brief
            </span>
            <h3 style={{ fontSize: '24px', fontWeight: 800, color: '#0F3D3E', margin: '6px 0' }}>
              {report.dataset_name} — Strategic Intelligence Report
            </h3>
            <div style={{ fontSize: '13px', color: '#64748B' }}>
              Generated on {report.generated_at} &bull; Objective: {report.objective}
            </div>
          </div>

          {/* Quick Metrics */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: '16px',
              marginBottom: '28px',
            }}
          >
            <div style={{ background: '#FAF8F5', padding: '16px', borderRadius: '12px', textAlign: 'center', border: '1px solid #E2E8F0' }}>
              <div style={{ fontSize: '22px', fontWeight: 800, color: '#0F3D3E' }}>
                {dataset.metadata.rows.toLocaleString()}
              </div>
              <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#64748B', fontWeight: 600, marginTop: '4px' }}>
                Total Records
              </div>
            </div>

            <div style={{ background: '#FAF8F5', padding: '16px', borderRadius: '12px', textAlign: 'center', border: '1px solid #E2E8F0' }}>
              <div style={{ fontSize: '22px', fontWeight: 800, color: '#0F3D3E' }}>
                {dataset.metadata.columns}
              </div>
              <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#64748B', fontWeight: 600, marginTop: '4px' }}>
                Features Analyzed
              </div>
            </div>

            <div style={{ background: '#FAF8F5', padding: '16px', borderRadius: '12px', textAlign: 'center', border: '1px solid #E2E8F0' }}>
              <div style={{ fontSize: '22px', fontWeight: 800, color: '#EA580C' }}>
                {dataset.health.grade} ({dataset.health.overall_score}/100)
              </div>
              <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#64748B', fontWeight: 600, marginTop: '4px' }}>
                Quality Rating
              </div>
            </div>

            <div style={{ background: '#FAF8F5', padding: '16px', borderRadius: '12px', textAlign: 'center', border: '1px solid #E2E8F0' }}>
              <div style={{ fontSize: '22px', fontWeight: 800, color: '#059669' }}>
                {(100 - dataset.metadata.missing_percentage).toFixed(1)}%
              </div>
              <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#64748B', fontWeight: 600, marginTop: '4px' }}>
                Completeness
              </div>
            </div>
          </div>

          {/* Actionable Recommendations */}
          <div style={{ marginBottom: '28px' }}>
            <h4 style={{ fontSize: '17px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 12px' }}>
              Strategic Recommendations
            </h4>
            <ol style={{ paddingLeft: '20px', margin: 0, display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {report.recommendations.map((rec, i) => (
                <li key={i} style={{ fontSize: '14px', color: '#334155', lineHeight: 1.5 }}>
                  {rec}
                </li>
              ))}
            </ol>
          </div>

          {/* Analytical Constraints */}
          <div>
            <h4 style={{ fontSize: '17px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 12px' }}>
              Caveats & Generalization Constraints
            </h4>
            <ul style={{ paddingLeft: '20px', margin: 0, display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {report.limitations.map((lim, i) => (
                <li key={i} style={{ fontSize: '13px', color: '#64748B', lineHeight: 1.5 }}>
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
