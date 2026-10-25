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
          padding: '24px 28px',
          borderRadius: '20px',
          border: '1px solid #EAE6DF',
          boxShadow: '0 2px 10px rgba(15, 61, 62, 0.04)',
          marginBottom: '24px',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <h2 style={{ fontSize: '24px', fontWeight: 700, color: '#0F3D3E', margin: 0 }}>
              Clean Data
            </h2>
            <span style={{ fontSize: '14px', color: '#648484', background: '#FAF8F5', padding: '4px 12px', borderRadius: '8px', border: '1px solid #EAE6DF' }}>
              Active Rows: <strong style={{ color: '#0F3D3E' }}>{dataset.metadata.rows.toLocaleString()}</strong>
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          {auditHistory.length > 0 && (
            <button
              onClick={handleRevert}
              style={{
                background: '#FAF8F5',
                color: '#476D6D',
                border: '1px solid #CBD5E1',
                borderRadius: '12px',
                padding: '10px 18px',
                fontSize: '15px',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.2s ease',
              }}
              title="Undo all cleaning actions and restore original raw data"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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
              borderRadius: '12px',
              padding: '10px 22px',
              fontSize: '15px',
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
            padding: '14px 20px',
            borderRadius: '12px',
            marginBottom: '20px',
            fontSize: '15px',
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
            style={{ background: 'transparent', border: 'none', cursor: 'pointer', fontSize: '18px' }}
          >
            &times;
          </button>
        </div>
      )}

      {/* Data Lineage Stage Progression Indicator */}
      <div
        style={{
          background: '#FFFFFF',
          borderRadius: '16px',
          padding: '16px 22px',
          border: '1px solid #EAE6DF',
          marginBottom: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px',
          fontSize: '14px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontWeight: 700, color: '#0F3D3E' }}>Data Lineage:</span>
          <span style={{ background: '#FAF8F5', color: '#476D6D', padding: '4px 12px', borderRadius: '8px', fontWeight: 600, fontSize: '13px', border: '1px solid #EAE6DF' }}>
            STAGE 1: ORIGINAL (Immutable)
          </span>
          <span style={{ color: '#94A3B8' }}>&rarr;</span>
          <span style={{ background: '#ECFDF5', color: '#047857', padding: '4px 12px', borderRadius: '8px', fontWeight: 700, fontSize: '13px', border: '1px solid #A7F3D0' }}>
            STAGE 2: CLEANED ({auditHistory.length} Approved Action{auditHistory.length === 1 ? '' : 's'})
          </span>
        </div>

        <div style={{ display: 'flex', gap: '18px', color: '#648484', fontSize: '14px' }}>
          <span>Active Rows: <strong style={{ color: '#0F3D3E' }}>{dataset.metadata.rows.toLocaleString()}</strong></span>
          <span>Health: <strong style={{ color: '#EA580C' }}>Grade {dataset.health.grade} ({dataset.health.overall_score}/100)</strong></span>
        </div>
      </div>

      {/* Main Grid: Recommendations on Left, Audit History on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.2fr) minmax(0, 0.8fr)', gap: '24px' }}>
        {/* Left Column: Recommendations */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ fontSize: '20px', fontWeight: 700, color: '#0F3D3E', margin: 0 }}>
              Diagnostic Recommendations ({recommendations.length})
            </h3>
          </div>

          {loading ? (
            <div style={{ padding: '40px', textAlign: 'center', color: '#648484', fontSize: '15px' }}>
              Diagnosing dataset anomalies...
            </div>
          ) : recommendations.length === 0 ? (
            <div
              style={{
                background: '#FFFFFF',
                borderRadius: '16px',
                padding: '40px',
                textAlign: 'center',
                border: '1px solid #EAE6DF',
              }}
            >
              <div
                style={{
                  width: '52px',
                  height: '52px',
                  borderRadius: '50%',
                  background: '#ECFDF5',
                  color: '#059669',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 14px',
                  fontSize: '24px',
                }}
              >
                &check;
              </div>
              <h4 style={{ margin: '0 0 6px', color: '#0F3D3E', fontSize: '18px' }}>Dataset Clean & Verified</h4>
              <p style={{ margin: 0, color: '#648484', fontSize: '15px' }}>
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
                      borderRadius: '16px',
                      padding: '22px',
                      border: '1px solid #EAE6DF',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.02)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span
                          style={{
                            fontSize: '12px',
                            fontWeight: 700,
                            padding: '4px 10px',
                            borderRadius: '6px',
                            textTransform: 'uppercase',
                            background: isHigh ? '#FEE2E2' : '#FEF3C7',
                            color: isHigh ? '#B91C1C' : '#B45309',
                          }}
                        >
                          {rec.severity}
                        </span>
                        <h4 style={{ margin: 0, fontSize: '17px', fontWeight: 700, color: '#0F3D3E' }}>
                          {rec.title}
                        </h4>
                      </div>
                      <span
                        style={{
                          fontSize: '13px',
                          color: '#476D6D',
                          background: '#FAF8F5',
                          padding: '4px 10px',
                          borderRadius: '8px',
                          fontWeight: 600,
                          border: '1px solid #EAE6DF',
                        }}
                      >
                        {rec.affected_rows.toLocaleString()} rows ({rec.affected_percentage}%)
                      </span>
                    </div>

                    <p style={{ fontSize: '15px', color: '#476D6D', margin: '0 0 14px', lineHeight: 1.6 }}>
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
                        padding: '12px 16px',
                        borderRadius: '12px',
                        border: '1px solid #EAE6DF',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <label style={{ fontSize: '14px', fontWeight: 600, color: '#0F3D3E' }}>Strategy:</label>
                        <select
                          value={currentStrat}
                          onChange={(e) => handleStrategyChange(rec.id, e.target.value)}
                          style={{
                            padding: '7px 12px',
                            borderRadius: '8px',
                            border: '1px solid #CBD5E1',
                            background: '#FFFFFF',
                            fontSize: '14px',
                            color: '#0F3D3E',
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

                      <div style={{ display: 'flex', gap: '10px' }}>
                        <button
                          onClick={() => handlePreview(rec)}
                          style={{
                            background: '#FFFFFF',
                            border: '1px solid #CBD5E1',
                            borderRadius: '8px',
                            padding: '8px 16px',
                            fontSize: '14px',
                            fontWeight: 600,
                            color: '#476D6D',
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
                            borderRadius: '8px',
                            padding: '8px 18px',
                            fontSize: '14px',
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
            <h3 style={{ fontSize: '20px', fontWeight: 700, color: '#0F3D3E', margin: 0 }}>
              Audit History Log ({auditHistory.length})
            </h3>
          </div>

          <div
            style={{
              background: '#FFFFFF',
              borderRadius: '16px',
              border: '1px solid #EAE6DF',
              padding: '16px',
              maxHeight: '650px',
              overflowY: 'auto',
            }}
          >
            {auditHistory.length === 0 ? (
              <div style={{ padding: '36px 16px', textAlign: 'center', color: '#648484', fontSize: '15px' }}>
                No transformations recorded yet. Approved fixes will be logged here.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {auditHistory.map((item, idx) => (
                  <div
                    key={item.id || idx}
                    style={{
                      borderLeft: '4px solid #0F3D3E',
                      background: '#FAF8F5',
                      padding: '14px 16px',
                      borderRadius: '0 10px 10px 0',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', color: '#648484', marginBottom: '6px' }}>
                      <span style={{ fontWeight: 700, color: '#0F3D3E', textTransform: 'uppercase' }}>
                        {item.rule.replace(/_/g, ' ')} {item.column ? `• ${item.column}` : ''}
                      </span>
                      <span>{item.timestamp}</span>
                    </div>
                    <p style={{ margin: '0 0 6px', fontSize: '15px', fontWeight: 600, color: '#0F3D3E' }}>
                      {item.transformation_summary}
                    </p>
                    <p style={{ margin: 0, fontSize: '13px', color: '#648484' }}>
                      <strong>Reason:</strong> {item.reason}
                    </p>
                    <div style={{ marginTop: '8px', fontSize: '13px', color: '#476D6D' }}>
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
              <h3 style={{ margin: 0, color: '#0F3D3E', fontSize: '20px', fontWeight: 700 }}>
                Cleaning Diff Preview
              </h3>
              <button
                onClick={() => setShowPreviewModal(false)}
                style={{ background: 'transparent', border: 'none', fontSize: '22px', cursor: 'pointer', color: '#648484' }}
              >
                &times;
              </button>
            </div>

            <div style={{ background: '#FAF8F5', padding: '18px', borderRadius: '14px', border: '1px solid #EAE6DF', marginBottom: '16px' }}>
              <p style={{ margin: '0 0 10px', fontSize: '15px', fontWeight: 600, color: '#0F3D3E' }}>
                {previewDiff.summary}
              </p>
              <div style={{ fontSize: '14px', color: '#648484', display: 'flex', gap: '18px' }}>
                <span>Affected Rows: <strong style={{ color: '#0F3D3E' }}>{previewDiff.affected_rows}</strong></span>
                <span>Strategy: <strong style={{ color: '#0F3D3E' }}>{previewDiff.strategy}</strong></span>
                {previewDiff.fill_value !== undefined && (
                  <span>Fill Value: <strong style={{ color: '#0F3D3E' }}>{String(previewDiff.fill_value)}</strong></span>
                )}
              </div>
            </div>

            {previewDiff.preview_diff && previewDiff.preview_diff.length > 0 && (
              <div style={{ marginBottom: '16px' }}>
                <span style={{ fontSize: '13px', fontWeight: 700, color: '#0F3D3E', textTransform: 'uppercase' }}>
                  Sample Records to be Removed:
                </span>
                <div style={{ maxHeight: '180px', overflowY: 'auto', marginTop: '8px', border: '1px solid #EAE6DF', borderRadius: '10px' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                    <tbody>
                      {previewDiff.preview_diff.map((row: any, i: number) => (
                        <tr key={i} style={{ borderBottom: '1px solid #F0EDE6' }}>
                          <td style={{ padding: '8px 12px', color: '#648484' }}>#{i + 1}</td>
                          <td style={{ padding: '8px 12px', color: '#0F3D3E' }}>{JSON.stringify(row).slice(0, 100)}...</td>
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
                  background: '#FAF8F5',
                  color: '#476D6D',
                  border: '1px solid #CBD5E1',
                  borderRadius: '10px',
                  padding: '9px 18px',
                  fontSize: '14px',
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
