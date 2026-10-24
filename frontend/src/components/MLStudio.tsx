import React, { useState, useEffect } from 'react';
import type { ProfileReport, AutoMLResponse, WhatIfResponse } from '../types';
import { trainAutoML, simulateWhatIf } from '../api';
import { PlotlyChart } from './PlotlyChart';

interface MLStudioProps {
  dataset: ProfileReport;
}

export const MLStudio: React.FC<MLStudioProps> = ({ dataset }) => {
  const columns = dataset.columns.map((c) => c.column);
  const defaultTarget = columns.find((c) => ['churn', 'survived', 'target', 'label', 'rating', 'sales', 'price'].includes(c.toLowerCase())) || columns[columns.length - 1];

  const [selectedTarget, setSelectedTarget] = useState<string>(defaultTarget || columns[0] || '');
  const [taskType, setTaskType] = useState<'auto' | 'classification' | 'regression'>('auto');
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [mlResult, setMlResult] = useState<AutoMLResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Synchronize target selection and reset stale ML results when switching datasets
  useEffect(() => {
    const newCols = dataset.columns.map((c) => c.column);
    const newTarget = newCols.find((c) => ['churn', 'survived', 'target', 'label', 'rating', 'sales', 'price'].includes(c.toLowerCase())) || newCols[newCols.length - 1] || newCols[0] || '';
    setSelectedTarget(newTarget);
    setMlResult(null);
    setWhatIfResult(null);
    setErrorMsg(null);
    setWhatIfInputs({});
  }, [dataset.metadata.id]);

  // What-If Simulator state
  const [whatIfInputs, setWhatIfInputs] = useState<Record<string, any>>({});
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [whatIfResult, setWhatIfResult] = useState<WhatIfResponse | null>(null);

  const handleTrain = async () => {
    if (!selectedTarget) return;
    setIsTraining(true);
    setErrorMsg(null);
    setWhatIfResult(null);

    try {
      const response = await trainAutoML({
        dataset_id: dataset.metadata.id,
        target_column: selectedTarget,
        task_type: taskType,
      });
      setMlResult(response);

      // Pre-fill what-if inputs from candidate features with defaults
      const initialInputs: Record<string, any> = {};
      response.features_used.slice(0, 5).forEach((feat) => {
        const colStat = dataset.columns.find((c) => c.column === feat);
        if (colStat?.median !== undefined && colStat.median !== null) {
          initialInputs[feat] = colStat.median;
        } else if (colStat?.sample_values && colStat.sample_values.length > 0) {
          initialInputs[feat] = colStat.sample_values[0];
        } else {
          initialInputs[feat] = 0;
        }
      });
      setWhatIfInputs(initialInputs);
    } catch (err: any) {
      setErrorMsg(err.message || 'AutoML tournament failed.');
    } finally {
      setIsTraining(false);
    }
  };

  const handleInputChange = (feature: string, value: any) => {
    setWhatIfInputs((prev) => ({
      ...prev,
      [feature]: isNaN(Number(value)) ? value : Number(value),
    }));
  };

  const handleSimulate = async () => {
    if (!mlResult) return;
    setIsSimulating(true);
    try {
      const res = await simulateWhatIf({
        dataset_id: dataset.metadata.id,
        target_column: mlResult.target_column,
        feature_inputs: whatIfInputs,
      });
      setWhatIfResult(res);
    } catch (err: any) {
      alert(err.message || 'Failed to simulate scenario.');
    } finally {
      setIsSimulating(false);
    }
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '16px 20px 48px' }}>
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
            Automated Machine Learning
          </span>
          <span style={{ fontSize: '13px', color: '#64748B' }}>
            Leakage-Free Validation & Tournament Leaderboard
          </span>
        </div>
        <h2 style={{ fontSize: '22px', fontWeight: 700, color: '#0F3D3E', margin: '8px 0 4px' }}>
          AutoML & Counterfactual Simulation Studio
        </h2>
        <p style={{ fontSize: '14px', color: '#64748B', margin: '0 0 20px' }}>
          Compare baseline dummy models against Random Forests, Ridge/Logistic Regression, and Decision Trees with full overfitting detection and what-if simulations.
        </p>

        {/* Configuration Bar */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '16px',
            background: '#FAF8F5',
            padding: '16px',
            borderRadius: '12px',
            border: '1px solid #E2E8F0',
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <label style={{ fontSize: '12px', fontWeight: 700, color: '#0F3D3E' }}>
              Target Feature to Predict:
            </label>
            <select
              value={selectedTarget}
              onChange={(e) => setSelectedTarget(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: '8px',
                border: '1px solid #CBD5E1',
                background: '#FFFFFF',
                fontSize: '13px',
                fontWeight: 600,
                color: '#0F3D3E',
                minWidth: '220px',
              }}
            >
              {columns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <label style={{ fontSize: '12px', fontWeight: 700, color: '#0F3D3E' }}>
              Task Mode:
            </label>
            <select
              value={taskType}
              onChange={(e) => setTaskType(e.target.value as any)}
              style={{
                padding: '8px 12px',
                borderRadius: '8px',
                border: '1px solid #CBD5E1',
                background: '#FFFFFF',
                fontSize: '13px',
                fontWeight: 500,
                color: '#1E293B',
              }}
            >
              <option value="auto">Auto-Detect</option>
              <option value="classification">Classification (Discrete / Categories)</option>
              <option value="regression">Regression (Continuous Numeric)</option>
            </select>
          </div>

          <button
            onClick={handleTrain}
            disabled={isTraining || !selectedTarget}
            style={{
              marginTop: '18px',
              background: '#0F3D3E',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '10px',
              padding: '10px 24px',
              fontSize: '14px',
              fontWeight: 700,
              cursor: isTraining ? 'not-allowed' : 'pointer',
              opacity: isTraining ? 0.7 : 1,
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              boxShadow: '0 2px 10px rgba(15, 61, 62, 0.2)',
            }}
          >
            {isTraining ? (
              <>
                <span
                  style={{
                    width: '14px',
                    height: '14px',
                    border: '2px solid #FFFFFF',
                    borderTopColor: 'transparent',
                    borderRadius: '50%',
                    animation: 'spin 0.8s linear infinite',
                  }}
                />
                Training Tournament...
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                Run AutoML Tournament
              </>
            )}
          </button>
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
          <strong>AutoML Error:</strong> {errorMsg}
        </div>
      )}

      {/* AutoML Results */}
      {mlResult && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Best Model Banner */}
          <div
            style={{
              background: 'linear-gradient(135deg, #0F3D3E 0%, #155E60 100%)',
              color: '#FFFFFF',
              borderRadius: '16px',
              padding: '24px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '16px',
              boxShadow: '0 4px 20px rgba(15, 61, 62, 0.15)',
            }}
          >
            <div>
              <span
                style={{
                  fontSize: '11px',
                  fontWeight: 800,
                  textTransform: 'uppercase',
                  letterSpacing: '1px',
                  background: '#EA580C',
                  padding: '4px 10px',
                  borderRadius: '20px',
                }}
              >
                Tournament Champion
              </span>
              <h3 style={{ fontSize: '24px', fontWeight: 800, margin: '8px 0 4px' }}>
                {mlResult.best_model_name}
              </h3>
              <p style={{ margin: 0, fontSize: '13px', opacity: 0.85 }}>
                Target: <strong>{mlResult.target_column}</strong> &bull; Task:{' '}
                <strong>{mlResult.task_type.toUpperCase()}</strong> &bull; Samples:{' '}
                {mlResult.total_samples.toLocaleString()} ({mlResult.train_samples} train / {mlResult.test_samples} test)
              </p>
            </div>

            <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
              <div style={{ textAlign: 'center', background: 'rgba(255,255,255,0.1)', padding: '12px 20px', borderRadius: '12px' }}>
                <div style={{ fontSize: '26px', fontWeight: 800, color: '#FED7AA' }}>
                  {mlResult.leaderboard[0].test_score}
                </div>
                <div style={{ fontSize: '11px', textTransform: 'uppercase', opacity: 0.8 }}>
                  Holdout Test Score
                </div>
              </div>
              <div style={{ textAlign: 'center', background: 'rgba(255,255,255,0.1)', padding: '12px 20px', borderRadius: '12px' }}>
                <div style={{ fontSize: '26px', fontWeight: 800 }}>
                  {mlResult.leaderboard[0].train_score}
                </div>
                <div style={{ fontSize: '11px', textTransform: 'uppercase', opacity: 0.8 }}>
                  Train Score
                </div>
              </div>
            </div>
          </div>

          {/* Model Tournament Leaderboard Table */}
          <div
            style={{
              background: '#FFFFFF',
              borderRadius: '16px',
              padding: '24px',
              border: '1px solid #E2E8F0',
            }}
          >
            <h4 style={{ fontSize: '17px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 16px' }}>
              Model Tournament Leaderboard
            </h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                <thead>
                  <tr style={{ background: '#FAF8F5', borderBottom: '2px solid #E2E8F0' }}>
                    <th style={{ padding: '10px 14px', textAlign: 'left', color: '#0F3D3E' }}>Rank & Model</th>
                    <th style={{ padding: '10px 14px', textAlign: 'left', color: '#0F3D3E' }}>Algorithm</th>
                    <th style={{ padding: '10px 14px', textAlign: 'center', color: '#0F3D3E' }}>Holdout Test Score</th>
                    <th style={{ padding: '10px 14px', textAlign: 'center', color: '#0F3D3E' }}>Train Score</th>
                    <th style={{ padding: '10px 14px', textAlign: 'center', color: '#0F3D3E' }}>Overfitting Gap</th>
                    <th style={{ padding: '10px 14px', textAlign: 'right', color: '#0F3D3E' }}>Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {mlResult.leaderboard.map((m, idx) => {
                    const isWinner = idx === 0 && !m.is_baseline;
                    const isOverfit = m.overfitting_gap > 0.15;
                    return (
                      <tr
                        key={m.model_name}
                        style={{
                          borderBottom: '1px solid #F1F5F9',
                          background: isWinner ? '#FFF7ED' : 'transparent',
                        }}
                      >
                        <td style={{ padding: '12px 14px', fontWeight: 700, color: '#1E293B' }}>
                          #{idx + 1} {m.model_name}
                          {m.is_baseline && (
                            <span
                              style={{
                                marginLeft: '8px',
                                fontSize: '10px',
                                background: '#E2E8F0',
                                color: '#475569',
                                padding: '2px 6px',
                                borderRadius: '4px',
                              }}
                            >
                              Baseline
                            </span>
                          )}
                          {isWinner && (
                            <span
                              style={{
                                marginLeft: '8px',
                                fontSize: '10px',
                                background: '#EA580C',
                                color: '#FFFFFF',
                                padding: '2px 6px',
                                borderRadius: '4px',
                              }}
                            >
                              Winner
                            </span>
                          )}
                        </td>
                        <td style={{ padding: '12px 14px', color: '#64748B' }}>{m.algorithm}</td>
                        <td style={{ padding: '12px 14px', textAlign: 'center', fontWeight: 700, color: '#0F3D3E' }}>
                          {m.test_score}
                        </td>
                        <td style={{ padding: '12px 14px', textAlign: 'center', color: '#64748B' }}>
                          {m.train_score}
                        </td>
                        <td style={{ padding: '12px 14px', textAlign: 'center' }}>
                          <span
                            style={{
                              fontSize: '11px',
                              fontWeight: 600,
                              padding: '2px 8px',
                              borderRadius: '6px',
                              background: isOverfit ? '#FEE2E2' : '#ECFDF5',
                              color: isOverfit ? '#B91C1C' : '#047857',
                            }}
                          >
                            {m.overfitting_gap > 0 ? `+${m.overfitting_gap}` : m.overfitting_gap} (
                            {isOverfit ? 'High' : 'Low'})
                          </span>
                        </td>
                        <td style={{ padding: '12px 14px', textAlign: 'right', color: '#64748B' }}>
                          {m.training_time_ms} ms
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div style={{ marginTop: '14px', fontSize: '12px', color: '#64748B' }}>
              <strong>Generalization Assessment:</strong> {mlResult.overfitting_assessment}
            </div>
          </div>

          {/* Grid: Feature Importance on Left, What-If on Right */}
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '24px' }}>
            {/* Feature Importance */}
            <div
              style={{
                background: '#FFFFFF',
                borderRadius: '16px',
                padding: '24px',
                border: '1px solid #E2E8F0',
              }}
            >
              <h4 style={{ fontSize: '17px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 16px' }}>
                Key Predictive Drivers (Feature Importance)
              </h4>
              <p style={{ fontSize: '13px', color: '#64748B', margin: '0 0 16px' }}>
                Attributes with the highest relative contribution to the winning model's decisions:
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {mlResult.feature_importance.map((f, i) => (
                  <div key={f.feature}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '4px' }}>
                      <span style={{ fontWeight: 600, color: '#1E293B' }}>
                        #{i + 1} {f.feature}
                      </span>
                      <span style={{ fontWeight: 700, color: '#EA580C' }}>{f.percentage}%</span>
                    </div>
                    <div style={{ height: '8px', background: '#F1F5F9', borderRadius: '4px', overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${Math.min(100, f.percentage)}%`,
                          background: 'linear-gradient(90deg, #EA580C, #F97316)',
                          borderRadius: '4px',
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Counterfactual What-If Simulator */}
            <div
              style={{
                background: '#FFFFFF',
                borderRadius: '16px',
                padding: '24px',
                border: '1px solid #E2E8F0',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <h4 style={{ fontSize: '17px', fontWeight: 700, color: '#0F3D3E', margin: 0 }}>
                  What-If Counterfactual Simulator
                </h4>
                <span
                  style={{
                    fontSize: '11px',
                    fontWeight: 700,
                    color: '#EA580C',
                    background: '#FFF7ED',
                    padding: '2px 8px',
                    borderRadius: '6px',
                  }}
                >
                  Interactive
                </span>
              </div>
              <p style={{ fontSize: '13px', color: '#64748B', margin: '0 0 16px' }}>
                Adjust feature inputs to simulate predicted outcomes against typical baseline conditions:
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '16px' }}>
                {Object.keys(whatIfInputs).map((feat) => (
                  <div key={feat} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
                    <label style={{ fontSize: '12px', fontWeight: 600, color: '#475569', minWidth: '130px' }}>
                      {feat}:
                    </label>
                    <input
                      type="text"
                      value={whatIfInputs[feat] ?? ''}
                      onChange={(e) => handleInputChange(feat, e.target.value)}
                      style={{
                        padding: '6px 10px',
                        borderRadius: '6px',
                        border: '1px solid #CBD5E1',
                        fontSize: '13px',
                        width: '100%',
                        maxWidth: '160px',
                      }}
                    />
                  </div>
                ))}
              </div>

              <button
                onClick={handleSimulate}
                disabled={isSimulating}
                style={{
                  width: '100%',
                  background: '#EA580C',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '10px',
                  padding: '10px',
                  fontSize: '13px',
                  fontWeight: 700,
                  cursor: isSimulating ? 'not-allowed' : 'pointer',
                  opacity: isSimulating ? 0.7 : 1,
                  marginBottom: '16px',
                }}
              >
                {isSimulating ? 'Simulating Prediction...' : 'Simulate Scenario Outcome'}
              </button>

              {/* What-If Output Card */}
              {whatIfResult && (
                <div
                  style={{
                    background: '#FAF8F5',
                    padding: '16px',
                    borderRadius: '12px',
                    border: '1px solid #E2E8F0',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: '#64748B' }}>
                      Predicted Result:
                    </span>
                    <span style={{ fontSize: '16px', fontWeight: 800, color: '#0F3D3E' }}>
                      {String(whatIfResult.prediction)}
                    </span>
                  </div>

                  {whatIfResult.prediction_delta !== undefined && whatIfResult.prediction_delta !== null && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#64748B', marginBottom: '8px' }}>
                      <span>Delta vs Baseline ({whatIfResult.baseline_prediction}):</span>
                      <strong style={{ color: whatIfResult.prediction_delta >= 0 ? '#059669' : '#DC2626' }}>
                        {whatIfResult.prediction_delta > 0 ? `+${whatIfResult.prediction_delta}` : whatIfResult.prediction_delta}
                      </strong>
                    </div>
                  )}

                  <p style={{ fontSize: '12px', color: '#334155', margin: '8px 0', lineHeight: 1.5 }}>
                    {whatIfResult.explanation}
                  </p>

                  <div
                    style={{
                      background: '#FFFBEB',
                      border: '1px solid #FDE68A',
                      color: '#92400E',
                      padding: '8px 10px',
                      borderRadius: '6px',
                      fontSize: '11px',
                      fontStyle: 'italic',
                      marginTop: '8px',
                    }}
                  >
                    {whatIfResult.disclaimer}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Held-Out Evaluation Chart & Comparison */}
          {(mlResult.evaluation_chart_spec || mlResult.chart_spec) && (
            <div
              style={{
                background: '#FFFFFF',
                borderRadius: '16px',
                padding: '24px',
                border: '1px solid #E2E8F0',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h4 style={{ fontSize: '17px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 4px' }}>
                    Holdout Test Evaluation Visualizer
                  </h4>
                  <p style={{ fontSize: '13px', color: '#64748B', margin: 0 }}>
                    Direct evaluation of the champion model ({mlResult.best_model_name}) on {mlResult.test_samples} unseen test observations:
                  </p>
                </div>
                <span
                  style={{
                    fontSize: '11px',
                    fontWeight: 700,
                    color: '#6D28D9',
                    background: '#F5F3FF',
                    border: '1px solid #DDD6FE',
                    padding: '4px 10px',
                    borderRadius: '8px',
                  }}
                >
                  Holdout Validation
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: mlResult.chart_spec && mlResult.evaluation_chart_spec ? '1fr 1fr' : '1fr', gap: '20px' }}>
                {mlResult.evaluation_chart_spec && (
                  <div style={{ background: '#FAF8F5', padding: '16px', borderRadius: '12px', border: '1px solid #EAE6DF' }}>
                    <PlotlyChart spec={mlResult.evaluation_chart_spec} plotlySpec={mlResult.evaluation_chart_spec.plotly_spec} height={320} />
                  </div>
                )}
                {mlResult.chart_spec && (
                  <div style={{ background: '#FAF8F5', padding: '16px', borderRadius: '12px', border: '1px solid #EAE6DF' }}>
                    <PlotlyChart spec={mlResult.chart_spec} plotlySpec={mlResult.chart_spec.plotly_spec} height={320} />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Held-Out Prediction Samples Table */}
          {mlResult.held_out_predictions && mlResult.held_out_predictions.length > 0 && (
            <div
              style={{
                background: '#FFFFFF',
                borderRadius: '16px',
                padding: '24px',
                border: '1px solid #E2E8F0',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h4 style={{ fontSize: '17px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 4px' }}>
                    Held-Out Test Set Predictions (First {mlResult.held_out_predictions.length} Samples)
                  </h4>
                  <p style={{ fontSize: '13px', color: '#64748B', margin: 0 }}>
                    Comparing ground-truth values against model predictions on held-out test data:
                  </p>
                </div>
                <span
                  style={{
                    fontSize: '11px',
                    fontWeight: 700,
                    color: '#0D9488',
                    background: '#F0FDFA',
                    border: '1px solid #CCECE6',
                    padding: '4px 10px',
                    borderRadius: '8px',
                  }}
                >
                  Leakage-Safe Holdout
                </span>
              </div>

              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                  <thead>
                    <tr style={{ background: '#FAF8F5', borderBottom: '2px solid #E2E8F0' }}>
                      <th style={{ padding: '10px 14px', textAlign: 'left', color: '#0F3D3E' }}>Row #</th>
                      <th style={{ padding: '10px 14px', textAlign: 'left', color: '#0F3D3E' }}>Actual Ground Truth</th>
                      <th style={{ padding: '10px 14px', textAlign: 'left', color: '#0F3D3E' }}>Model Prediction</th>
                      <th style={{ padding: '10px 14px', textAlign: 'center', color: '#0F3D3E' }}>
                        {mlResult.task_type === 'classification' ? 'Outcome' : 'Residual (Error)'}
                      </th>
                      <th style={{ padding: '10px 14px', textAlign: 'left', color: '#0F3D3E' }}>Sample Features</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mlResult.held_out_predictions.map((p) => {
                      const isCorrect = p.is_correct;
                      return (
                        <tr key={p.row_index} style={{ borderBottom: '1px solid #F1F5F9' }}>
                          <td style={{ padding: '10px 14px', fontWeight: 600, color: '#64748B' }}>
                            #{p.row_index}
                          </td>
                          <td style={{ padding: '10px 14px', fontWeight: 700, color: '#0F3D3E' }}>
                            {String(p.actual)}
                          </td>
                          <td style={{ padding: '10px 14px', fontWeight: 700, color: '#EA580C' }}>
                            {String(p.predicted)}
                          </td>
                          <td style={{ padding: '10px 14px', textAlign: 'center' }}>
                            {mlResult.task_type === 'classification' ? (
                              <span
                                style={{
                                  fontSize: '11px',
                                  fontWeight: 700,
                                  padding: '2px 8px',
                                  borderRadius: '6px',
                                  background: isCorrect ? '#ECFDF5' : '#FEF2F2',
                                  color: isCorrect ? '#047857' : '#B91C1C',
                                  border: `1px solid ${isCorrect ? '#A7F3D0' : '#FECACA'}`,
                                }}
                              >
                                {isCorrect ? 'Correct' : 'Misclassified'}
                              </span>
                            ) : (
                              <span
                                style={{
                                  fontSize: '12px',
                                  fontFamily: 'monospace',
                                  color: p.error_or_residual && p.error_or_residual > 0 ? '#B91C1C' : '#047857',
                                }}
                              >
                                {p.error_or_residual !== undefined && p.error_or_residual !== null
                                  ? (p.error_or_residual > 0 ? `+${p.error_or_residual}` : `${p.error_or_residual}`)
                                  : '-'}
                              </span>
                            )}
                          </td>
                          <td style={{ padding: '10px 14px', fontSize: '11px', color: '#64748B' }}>
                            {Object.entries(p.features)
                              .slice(0, 3)
                              .map(([k, v]) => `${k}: ${v}`)
                              .join(' | ')}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Model Limitations & Validation Caveats */}
          {mlResult.limitations && mlResult.limitations.length > 0 && (
            <div
              style={{
                background: '#FAF8F5',
                borderRadius: '16px',
                padding: '20px 24px',
                border: '1px solid #EAE6DF',
              }}
            >
              <h5 style={{ fontSize: '14px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 10px' }}>
                Model Limitations & Scientific Caveats
              </h5>
              <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '12px', color: '#476D6D', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {mlResult.limitations.map((lim, idx) => (
                  <li key={idx}>{lim}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
