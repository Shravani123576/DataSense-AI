import React, { useState, useEffect } from 'react';
import type {
  ProfileReport,
  CleaningRecommendation,
  CleaningAuditRecord,
  CleaningSummaryResponse,
} from '../types';
import {
  fetchCleaningRecommendations,
  previewCleaningAction,
  applyCleaningAction,
  fetchCleaningAudit,
  resetToRawDataset,
  getCleanedExportUrl,
} from '../api';

interface CleanStudioProps {
  dataset: ProfileReport;
  onDatasetUpdated: (updated: ProfileReport) => void;
}

export const CleanStudio: React.FC<CleanStudioProps> = ({ dataset, onDatasetUpdated }) => {
  const [recommendations, setRecommendations] = useState<CleaningRecommendation[]>([]);
  const [auditHistory, setAuditHistory] = useState<CleaningAuditRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);
  const [selectedStrategies, setSelectedStrategies] = useState<Record<string, string>>({});
  const [previewDiff, setPreviewDiff] = useState<any | null>(null);
  const [showPreviewModal, setShowPreviewModal] = useState<boolean>(false);
  const [feedbackMsg, setFeedbackMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const datasetId = dataset.metadata.id;

  const loadCleaningState = async () => {
    setLoading(true);
    try {
      const [recs, audit] = await Promise.all([
        fetchCleaningRecommendations(datasetId),
        fetchCleaningAudit(datasetId),
      ]);
      setRecommendations(recs);
      setAuditHistory(audit.audit_history || []);
      
      // Default strategies
      const defaults: Record<string, string> = {};
      recs.forEach((r) => {
        defaults[r.id] = r.strategy_options[0] || 'median';
      });
      setSelectedStrategies(defaults);
    } catch (err: any) {
      console.error('Error loading cleaning status:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadCleaningState();
  }, [datasetId]);

  const handleStrategyChange = (recId: string, strategy: string) => {
    setSelectedStrategies((prev) => ({ ...prev, [recId]: strategy }));
  };

  const handlePreview = async (rec: CleaningRecommendation) => {
    try {
      const strat = selectedStrategies[rec.id] || rec.strategy_options[0];
      const preview = await previewCleaningAction({
        dataset_id: datasetId,
        rule: rec.rule,
        column: rec.column,
        strategy: strat,
      });
      setPreviewDiff({ ...preview, recTitle: rec.title });
      setShowPreviewModal(true);
    } catch (err: any) {
      setFeedbackMsg({ type: 'error', text: err.message || 'Failed to preview cleaning.' });
    }
  };

  const handleApply = async (rec: CleaningRecommendation) => {
    setActionInProgress(rec.id);
    setFeedbackMsg(null);
    try {
      const strat = selectedStrategies[rec.id] || rec.strategy_options[0];
      const result: CleaningSummaryResponse = await applyCleaningAction({
        dataset_id: datasetId,
        rule: rec.rule,
        column: rec.column,
        strategy: strat,
      });

      setAuditHistory(result.audit_history);
      setRecommendations(result.remaining_recommendations);
      setFeedbackMsg({
        type: 'success',
        text: `Successfully applied: ${rec.title}. Health score is now ${result.current_health.overall_score}/100.`,
      });

      // Update parent dataset profile
      onDatasetUpdated({
        ...dataset,
        health: result.current_health,
        metadata: {
          ...dataset.metadata,
          rows: result.total_records,
        },
      });
    } catch (err: any) {
      setFeedbackMsg({ type: 'error', text: err.message || 'Failed to apply cleaning action.' });
    } finally {
      setActionInProgress(null);
    }
  };

  const handleRevert = async () => {
    if (!window.confirm('Revert all cleaning transformations back to the original raw dataset?')) {
      return;
    }
    setLoading(true);
    try {
      const result = await resetToRawDataset(datasetId);
      setFeedbackMsg({ type: 'success', text: 'Dataset reverted to original raw state.' });
      await loadCleaningState();
      onDatasetUpdated({
        ...dataset,
        health: result.health,
        metadata: {
          ...dataset.metadata,
          rows: result.rows,
        },
      });
    } catch (err: any) {
      setFeedbackMsg({ type: 'error', text: err.message || 'Failed to revert dataset.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '16px 20px 48px' }}>
      {/* Header Banner */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '16px',
          background: '#FFFFFF',
          padding: '24px',
          borderRadius: '16px',
          border: '1px solid #E2E8F0',
          boxShadow: '0 4px 16px rgba(15, 61, 62, 0.04)',
          marginBottom: '24px',
        }}
      >
        <div>
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
              Auditable Data Hygiene
            </span>
            <span style={{ fontSize: '13px', color: '#64748B' }}>
              Active Rows: <strong>{dataset.metadata.rows.toLocaleString()}</strong>
            </span>
          </div>
          <h2 style={{ fontSize: '22px', fontWeight: 700, color: '#0F3D3E', margin: '8px 0 4px' }}>
            Data Cleaning & Provenance Studio
          </h2>
          <p style={{ fontSize: '14px', color: '#64748B', margin: 0 }}>
            Inspect evidence-backed quality recommendations, choose transformation strategies, and track every change in the audit ledger.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          {auditHistory.length > 0 && (
            <button
              onClick={handleRevert}
              style={{
                background: '#F1F5F9',
                color: '#475569',
                border: '1px solid #CBD5E1',
                borderRadius: '10px',
                padding: '10px 16px',
                fontSize: '13px',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                transition: 'all 0.2s ease',
              }}
              title="Undo all cleaning actions and restore original raw data"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
              </svg>
              Revert to Raw
            </button>
          )}

          <a
            href={getCleanedExportUrl(datasetId)}
            download
            style={{
              background: '#0F3D3E',
              color: '#FFFFFF',
              borderRadius: '10px',
              padding: '10px 18px',
              fontSize: '13px',
              fontWeight: 600,
              textDecoration: 'none',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              boxShadow: '0 2px 8px rgba(15, 61, 62, 0.2)',
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Export Cleaned CSV
          </a>
        </div>
      </div>

      {/* Notification Banner */}
      {feedbackMsg && (
        <div
          style={{
            padding: '12px 18px',
            borderRadius: '10px',
            marginBottom: '20px',
            fontSize: '14px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: feedbackMsg.type === 'success' ? '#ECFDF5' : '#FEF2F2',
            color: feedbackMsg.type === 'success' ? '#065F46' : '#991B1B',
            border: `1px solid ${feedbackMsg.type === 'success' ? '#A7F3D0' : '#FECACA'}`,
          }}
        >
          <span>{feedbackMsg.text}</span>
          <button
            onClick={() => setFeedbackMsg(null)}
            style={{ background: 'transparent', border: 'none', cursor: 'pointer', fontSize: '16px' }}
          >
            &times;
          </button>
        </div>
      )}

      {/* Main Grid: Recommendations on Left, Audit History on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.2fr) minmax(0, 0.8fr)', gap: '24px' }}>
        {/* Left Column: Recommendations */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#0F3D3E', margin: 0 }}>
              Diagnostic Recommendations ({recommendations.length})
            </h3>
            <span style={{ fontSize: '12px', color: '#64748B' }}>Uncommitted suggestions require your approval</span>
          </div>

          {loading ? (
            <div style={{ padding: '40px', textAlign: 'center', color: '#64748B' }}>
              Diagnosing dataset anomalies...
            </div>
          ) : recommendations.length === 0 ? (
            <div
              style={{
                background: '#FFFFFF',
                borderRadius: '16px',
                padding: '40px',
                textAlign: 'center',
                border: '1px solid #E2E8F0',
              }}
            >
              <div
                style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '50%',
                  background: '#ECFDF5',
                  color: '#059669',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 12px',
                  fontSize: '22px',
                }}
              >
                &check;
              </div>
              <h4 style={{ margin: '0 0 6px', color: '#0F3D3E', fontSize: '16px' }}>Dataset Clean & Verified</h4>
              <p style={{ margin: 0, color: '#64748B', fontSize: '13px' }}>
                No urgent anomalies or missing values detected in the active working table.
              </p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {recommendations.map((rec) => {
                const currentStrat = selectedStrategies[rec.id] || rec.strategy_options[0];
                const isWorking = actionInProgress === rec.id;
                const isHigh = rec.severity === 'high';

                return (
                  <div
                    key={rec.id}
                    style={{
                      background: '#FFFFFF',
                      borderRadius: '14px',
                      padding: '20px',
                      border: '1px solid #E2E8F0',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.02)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span
                          style={{
                            fontSize: '11px',
                            fontWeight: 700,
                            padding: '3px 8px',
                            borderRadius: '6px',
                            textTransform: 'uppercase',
                            background: isHigh ? '#FEE2E2' : '#FEF3C7',
                            color: isHigh ? '#B91C1C' : '#B45309',
                          }}
                        >
                          {rec.severity}
                        </span>
                        <h4 style={{ margin: 0, fontSize: '15px', fontWeight: 700, color: '#0F3D3E' }}>
                          {rec.title}
                        </h4>
                      </div>
                      <span
                        style={{
                          fontSize: '12px',
                          color: '#64748B',
                          background: '#F1F5F9',
                          padding: '3px 8px',
                          borderRadius: '6px',
                          fontWeight: 600,
                        }}
                      >
                        {rec.affected_rows.toLocaleString()} rows ({rec.affected_percentage}%)
                      </span>
                    </div>

                    <p style={{ fontSize: '13px', color: '#475569', margin: '0 0 12px', lineHeight: 1.5 }}>
                      {rec.evidence}
                    </p>

                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        flexWrap: 'wrap',
                        gap: '12px',
                        background: '#FAF8F5',
                        padding: '12px 14px',
                        borderRadius: '10px',
                        border: '1px solid #E2E8F0',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <label style={{ fontSize: '12px', fontWeight: 600, color: '#64748B' }}>Strategy:</label>
                        <select
                          value={currentStrat}
                          onChange={(e) => handleStrategyChange(rec.id, e.target.value)}
                          style={{
                            padding: '6px 10px',
                            borderRadius: '6px',
                            border: '1px solid #CBD5E1',
                            background: '#FFFFFF',
                            fontSize: '12px',
                            color: '#1E293B',
                            fontWeight: 500,
                            cursor: 'pointer',
                          }}
                        >
                          {rec.strategy_options.map((opt) => (
                            <option key={opt} value={opt}>
                              {opt.replace(/_/g, ' ').toUpperCase()}
                            </option>
                          ))}
                        </select>
                      </div>

                      <div style={{ display: 'flex', gap: '8px' }}>
                        <button
                          onClick={() => handlePreview(rec)}
                          style={{
                            background: '#FFFFFF',
                            border: '1px solid #CBD5E1',
                            borderRadius: '6px',
                            padding: '6px 12px',
                            fontSize: '12px',
                            fontWeight: 600,
                            color: '#475569',
                            cursor: 'pointer',
                          }}
                        >
                          Preview Diff
                        </button>
                        <button
                          onClick={() => handleApply(rec)}
                          disabled={isWorking}
                          style={{
                            background: '#EA580C',
                            color: '#FFFFFF',
                            border: 'none',
                            borderRadius: '6px',
                            padding: '6px 14px',
                            fontSize: '12px',
                            fontWeight: 600,
                            cursor: isWorking ? 'not-allowed' : 'pointer',
                            opacity: isWorking ? 0.7 : 1,
                          }}
                        >
                          {isWorking ? 'Applying...' : 'Apply Fix'}
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Right Column: Audit Ledger */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#0F3D3E', margin: 0 }}>
              Audit History Log ({auditHistory.length})
            </h3>
            <span style={{ fontSize: '12px', color: '#64748B' }}>Complete transformation provenance</span>
          </div>

          <div
            style={{
              background: '#FFFFFF',
              borderRadius: '16px',
              border: '1px solid #E2E8F0',
              padding: '16px',
              maxHeight: '650px',
              overflowY: 'auto',
            }}
          >
            {auditHistory.length === 0 ? (
              <div style={{ padding: '32px 16px', textAlign: 'center', color: '#94A3B8', fontSize: '13px' }}>
                No transformations recorded yet. When you approve fixes, auditable records will be preserved here.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {auditHistory.map((item, idx) => (
                  <div
                    key={item.id || idx}
                    style={{
                      borderLeft: '3px solid #0F3D3E',
                      background: '#FAF8F5',
                      padding: '12px 14px',
                      borderRadius: '0 8px 8px 0',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#64748B', marginBottom: '4px' }}>
                      <span style={{ fontWeight: 700, color: '#0F3D3E', textTransform: 'uppercase' }}>
                        {item.rule.replace(/_/g, ' ')} {item.column ? `• ${item.column}` : ''}
                      </span>
                      <span>{item.timestamp}</span>
                    </div>
                    <p style={{ margin: '0 0 4px', fontSize: '13px', fontWeight: 600, color: '#1E293B' }}>
                      {item.transformation_summary}
                    </p>
                    <p style={{ margin: 0, fontSize: '11px', color: '#64748B', fontStyle: 'italic' }}>
                      Reason: {item.reason}
                    </p>
                    <div style={{ marginTop: '6px', fontSize: '11px', color: '#475569' }}>
                      Affected: <strong>{item.affected_rows.toLocaleString()} rows</strong> ({item.affected_percentage}%)
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Preview Diff Modal */}
      {showPreviewModal && previewDiff && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(15, 61, 62, 0.4)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '20px',
          }}
        >
          <div
            style={{
              background: '#FFFFFF',
              borderRadius: '16px',
              maxWidth: '600px',
              width: '100%',
              padding: '24px',
              boxShadow: '0 20px 40px rgba(0, 0, 0, 0.15)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ margin: 0, color: '#0F3D3E', fontSize: '18px', fontWeight: 700 }}>
                Cleaning Diff Preview
              </h3>
              <button
                onClick={() => setShowPreviewModal(false)}
                style={{ background: 'transparent', border: 'none', fontSize: '20px', cursor: 'pointer', color: '#64748B' }}
              >
                &times;
              </button>
            </div>

            <div style={{ background: '#FAF8F5', padding: '16px', borderRadius: '12px', border: '1px solid #E2E8F0', marginBottom: '16px' }}>
              <p style={{ margin: '0 0 8px', fontSize: '14px', fontWeight: 600, color: '#0F3D3E' }}>
                {previewDiff.summary}
              </p>
              <div style={{ fontSize: '12px', color: '#64748B', display: 'flex', gap: '16px' }}>
                <span>Affected Rows: <strong>{previewDiff.affected_rows}</strong></span>
                <span>Strategy: <strong>{previewDiff.strategy}</strong></span>
                {previewDiff.fill_value !== undefined && (
                  <span>Fill Value: <strong>{String(previewDiff.fill_value)}</strong></span>
                )}
              </div>
            </div>

            {previewDiff.preview_diff && previewDiff.preview_diff.length > 0 && (
              <div style={{ marginBottom: '16px' }}>
                <span style={{ fontSize: '12px', fontWeight: 700, color: '#64748B', textTransform: 'uppercase' }}>
                  Sample Records to be Removed:
                </span>
                <div style={{ maxHeight: '180px', overflowY: 'auto', marginTop: '6px', border: '1px solid #E2E8F0', borderRadius: '8px' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                    <tbody>
                      {previewDiff.preview_diff.map((row: any, i: number) => (
                        <tr key={i} style={{ borderBottom: '1px solid #F1F5F9' }}>
                          <td style={{ padding: '6px 10px', color: '#64748B' }}>#{i + 1}</td>
                          <td style={{ padding: '6px 10px', color: '#1E293B' }}>{JSON.stringify(row).slice(0, 100)}...</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button
                onClick={() => setShowPreviewModal(false)}
                style={{
                  background: '#F1F5F9',
                  color: '#475569',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '8px 16px',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                Close Preview
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
