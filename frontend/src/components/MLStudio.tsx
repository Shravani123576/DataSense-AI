import React, { useState, useEffect, useMemo } from 'react';
import type { ProfileReport, AutoMLResponse, WhatIfResponse } from '../types';
import { trainAutoML, simulateWhatIf } from '../api';
import { PlotlyChart } from './PlotlyChart';

const AUTO_TIME_FEATURES: string[] = ['Year', 'year_index', 'lag_1', 'lag_2', 'rolling_mean_3', 'rolling_mean_5', 'diff_1'];

interface MLStudioProps {
  dataset: ProfileReport;
}

export const MLStudio: React.FC<MLStudioProps> = ({ dataset }) => {
  const columns = dataset.columns.map((c) => c.column);

  const getSmartDefaultTarget = (cols: typeof dataset.columns): string => {
    const colNames = cols.map((c) => c.column);
    // 1. Look for common predictive target terms
    const knownTargets = ['churn', 'survived', 'target', 'label', 'rating', 'sales', 'price', 'revenue', 'outcome', 'status'];
    const match = colNames.find((name) => knownTargets.includes(name.toLowerCase()));
    if (match) return match;

    // 2. Look for numerical columns that are not unique IDs
    const validNumeric = cols.filter((c) => {
      const name = c.column.toLowerCase();
      const isId = name.endsWith('id') || name.startsWith('id') || name === 'index' || name === 'key';
      return c.inferred_type === 'numeric' && !isId;
    });
    if (validNumeric.length > 0) {
      return validNumeric[validNumeric.length - 1].column;
    }

    // 3. Look for low-cardinality categorical target
    const validCat = cols.filter((c) => {
      const name = c.column.toLowerCase();
      const isId = name.endsWith('id') || name.startsWith('id');
      return !isId && c.unique_count >= 2 && c.unique_count <= 20;
    });
    if (validCat.length > 0) {
      return validCat[0].column;
    }

    return colNames[colNames.length - 1] || colNames[0] || '';
  };

  const initialTarget = getSmartDefaultTarget(dataset.columns);

  const getSmartInitialFeatures = (target: string) => {
    const colObj = dataset.columns.find((c) => c.column === target);
    const cont = colObj?.inferred_type === 'numeric' || (colObj && colObj.unique_count > 20);
    const others = columns.filter((c) => c !== target);
    if ((cont && others.length === 0) || columns.some((c) => c.toLowerCase().includes('year') || c.toLowerCase().includes('date'))) {
      return Array.from(new Set([...AUTO_TIME_FEATURES, ...others]));
    }
    return others;
  };

  const [selectedTarget, setSelectedTarget] = useState<string>(initialTarget);
  const [taskType, setTaskType] = useState<'auto' | 'classification' | 'regression'>('auto');
  const [validationStrategy, setValidationStrategy] = useState<'auto' | 'kfold_cv' | 'stratified_holdout' | 'chronological'>('auto');
  const [timeColumn, setTimeColumn] = useState<string>('');
  const [testSize, setTestSize] = useState<number>(0.2);
  const [algorithmFilter, setAlgorithmFilter] = useState<string>('all');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(() =>
    getSmartInitialFeatures(initialTarget)
  );
  const [showFeatureSelector, setShowFeatureSelector] = useState<boolean>(false);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [mlResult, setMlResult] = useState<AutoMLResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const selectedColObj = dataset.columns.find((c) => c.column === selectedTarget);
  const isContinuous = selectedColObj?.inferred_type === 'numeric' || (selectedColObj && selectedColObj.unique_count > 20);
  const otherColumns = columns.filter((c) => c !== selectedTarget);
  const isTimeSeries = (isContinuous && otherColumns.length === 0) || columns.some((c) => c.toLowerCase().includes('year') || c.toLowerCase().includes('date'));

  const availableFeatures: string[] = useMemo(() => {
    if (isTimeSeries) {
      return Array.from(new Set<string>([...AUTO_TIME_FEATURES, ...otherColumns]));
    }
    return otherColumns;
  }, [isTimeSeries, otherColumns]);

  // Synchronize target selection and reset stale ML results when switching datasets
  useEffect(() => {
    const newTarget = getSmartDefaultTarget(dataset.columns);
    setSelectedTarget(newTarget);
    setSelectedFeatures(getSmartInitialFeatures(newTarget));
    setMlResult(null);
    setWhatIfResult(null);
    setErrorMsg(null);
    setWhatIfInputs({});
    setValidationStrategy('auto');
    setTimeColumn('');
    setTestSize(0.2);
    setAlgorithmFilter('all');
    setShowFeatureSelector(false);
  }, [dataset.metadata.id]);

  const handleTargetChange = (newTarget: string) => {
    setSelectedTarget(newTarget);
    setSelectedFeatures(getSmartInitialFeatures(newTarget));
  };

  // What-If Simulator state
  const [whatIfInputs, setWhatIfInputs] = useState<Record<string, any>>({});
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [whatIfResult, setWhatIfResult] = useState<WhatIfResponse | null>(null);

  const handleTrain = async () => {
    if (!selectedTarget) {
      setErrorMsg('Please select a target feature to train models.');
      return;
    }
    setIsTraining(true);
    setErrorMsg(null);
    setWhatIfResult(null);

    const algoList = algorithmFilter === 'all'
      ? undefined
      : algorithmFilter === 'ensembles'
      ? ['Random Forest', 'Gradient Boosting']
      : algorithmFilter === 'linear'
      ? ['Logistic Regression', 'Ridge Regression']
      : [algorithmFilter];

    const activeFeatures = selectedFeatures.filter((c: string) => c !== selectedTarget);
    const finalFeatures = activeFeatures.length > 0 ? activeFeatures : availableFeatures.filter((c: string) => c !== selectedTarget);

    try {
      const response = await trainAutoML({
        dataset_id: dataset.metadata.id,
        target_column: selectedTarget,
        task_type: taskType,
        validation_strategy: validationStrategy,
        time_column: validationStrategy === 'chronological' ? (timeColumn || undefined) : undefined,
        test_size: testSize,
        selected_features: finalFeatures.length > 0 ? finalFeatures : undefined,
        algorithm_filter: algoList,
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
          padding: '24px 28px',
          borderRadius: '20px',
          border: '1px solid #EAE6DF',
          boxShadow: '0 2px 10px rgba(15, 61, 62, 0.04)',
          marginBottom: '24px',
        }}
      >
        <h2 style={{ fontSize: '24px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 16px' }}>
          Predict (AutoML)
        </h2>

        {/* Configuration Bar */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '16px',
            background: '#FAF8F5',
            padding: '16px 20px',
            borderRadius: '14px',
            border: '1px solid #EAE6DF',
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '14px', fontWeight: 600, color: '#0F3D3E' }}>
              Target Feature:
            </label>
            <select
              value={selectedTarget}
              onChange={(e) => handleTargetChange(e.target.value)}
              style={{
                padding: '9px 14px',
                borderRadius: '10px',
                border: '1px solid #CBD5E1',
                background: '#FFFFFF',
                fontSize: '14px',
                fontWeight: 600,
                color: '#0F3D3E',
                minWidth: '220px',
                cursor: 'pointer',
              }}
            >
              {columns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '14px', fontWeight: 600, color: '#0F3D3E' }}>
              Task Mode:
            </label>
            <select
              value={isContinuous ? 'regression' : taskType}
              onChange={(e) => setTaskType(e.target.value as any)}
              style={{
                padding: '9px 14px',
                borderRadius: '10px',
                border: '1px solid #CBD5E1',
                background: '#FFFFFF',
                fontSize: '14px',
                fontWeight: 500,
                color: '#0F3D3E',
                cursor: 'pointer',
              }}
            >
              <option value="auto">Auto-Detect {isContinuous ? '(Regression)' : ''}</option>
              <option value="regression">Regression (Continuous)</option>
              {!isContinuous && (
                <option value="classification">Classification (Categories)</option>
              )}
            </select>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '14px', fontWeight: 600, color: '#0F3D3E' }}>
              Validation Strategy:
            </label>
            <select
              value={validationStrategy}
              onChange={(e) => setValidationStrategy(e.target.value as any)}
              style={{
                padding: '9px 14px',
                borderRadius: '10px',
                border: '1px solid #CBD5E1',
                background: '#FFFFFF',
                fontSize: '14px',
                fontWeight: 500,
                color: '#0F3D3E',
                cursor: 'pointer',
              }}
            >
              <option value="auto">
                {isTimeSeries ? 'Auto (Time-ordered holdout)' : 'Auto (Stratified / 5-Fold)'}
              </option>
              <option value="chronological">Time-ordered holdout</option>
              <option value="kfold_cv">Rolling origin / K-Fold</option>
              {!isContinuous && (
                <option value="stratified_holdout">Stratified Holdout</option>
              )}
            </select>
          </div>

          {validationStrategy === 'chronological' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <label style={{ fontSize: '14px', fontWeight: 600, color: '#0F3D3E' }}>
                Time Column:
              </label>
              <select
                value={timeColumn}
                onChange={(e) => setTimeColumn(e.target.value)}
                style={{
                  padding: '9px 14px',
                  borderRadius: '10px',
                  border: '1px solid #CBD5E1',
                  background: '#FFFFFF',
                  fontSize: '14px',
                  fontWeight: 500,
                  color: '#0F3D3E',
                  cursor: 'pointer',
                }}
              >
                <option value="">Auto-Detect Date Column</option>
                {columns.filter(c => c !== selectedTarget).map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '14px', fontWeight: 600, color: '#0F3D3E' }}>
              Test Split:
            </label>
            <select
              value={testSize}
              onChange={(e) => setTestSize(Number(e.target.value))}
              style={{
                padding: '9px 14px',
                borderRadius: '10px',
                border: '1px solid #CBD5E1',
                background: '#FFFFFF',
                fontSize: '14px',
                fontWeight: 500,
                color: '#0F3D3E',
                cursor: 'pointer',
              }}
            >
              <option value={0.2}>80% Train / 20% Holdout Test</option>
              <option value={0.3}>70% Train / 30% Holdout Test</option>
              <option value={0.15}>85% Train / 15% Holdout Test</option>
            </select>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '14px', fontWeight: 600, color: '#0F3D3E' }}>
              Algorithms:
            </label>
            <select
              value={algorithmFilter}
              onChange={(e) => setAlgorithmFilter(e.target.value)}
              style={{
                padding: '9px 14px',
                borderRadius: '10px',
                border: '1px solid #CBD5E1',
                background: '#FFFFFF',
                fontSize: '14px',
                fontWeight: 500,
                color: '#0F3D3E',
                cursor: 'pointer',
              }}
            >
              <option value="all">AutoML Tournament (All Models)</option>
              <option value="ensembles">Ensembles (Random Forest + Boosting)</option>
              <option value="Gradient Boosting">Gradient Boosting</option>
              <option value="Random Forest">Random Forest</option>
              <option value="Decision Tree">Decision Tree</option>
              <option value="linear">Linear / Regularized Model</option>
            </select>
          </div>

          {/* Feature Predictor Selection Toggle */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '14px', fontWeight: 600, color: '#0F3D3E' }}>
              Predictor Features:
            </label>
            <button
              type="button"
              onClick={() => setShowFeatureSelector(!showFeatureSelector)}
              style={{
                padding: '9px 14px',
                borderRadius: '10px',
                border: '1px solid #CBD5E1',
                background: showFeatureSelector ? '#FFF7ED' : '#FFFFFF',
                color: showFeatureSelector ? '#EA580C' : '#0F3D3E',
                fontSize: '14px',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <span>{selectedFeatures.length} of {availableFeatures.length} Selected</span>
              <span style={{ fontSize: '11px' }}>{showFeatureSelector ? '▲' : '▼'}</span>
            </button>
          </div>

          <button
            onClick={handleTrain}
            disabled={isTraining || !selectedTarget || selectedFeatures.length === 0 || dataset.metadata.rows < 15}
            style={{
              marginTop: '22px',
              background: '#0F3D3E',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '12px',
              padding: '11px 26px',
              fontSize: '15px',
              fontWeight: 700,
              cursor: isTraining || !selectedTarget || selectedFeatures.length === 0 || dataset.metadata.rows < 15 ? 'not-allowed' : 'pointer',
              opacity: isTraining || !selectedTarget || selectedFeatures.length === 0 || dataset.metadata.rows < 15 ? 0.7 : 1,
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
                    width: '16px',
                    height: '16px',
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
          {selectedFeatures.length === 0 && (
            <div style={{ color: '#EA580C', fontSize: '13px', marginTop: '8px', fontWeight: 500 }}>
              Please select at least 1 predictor feature to run AutoML tournament.
            </div>
          )}
          {dataset.metadata.rows < 15 && (
            <div style={{ color: '#EA580C', fontSize: '13px', marginTop: '8px', fontWeight: 500 }}>
              Dataset contains fewer than 15 rows required for AutoML tournament.
            </div>
          )}
        </div>

        {/* Expandable Feature Selection Drawer */}
        {showFeatureSelector && (
          <div
            style={{
              marginTop: '16px',
              padding: '16px 20px',
              background: '#FAF8F5',
              borderRadius: '14px',
              border: '1px solid #EAE6DF',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: '#0F3D3E' }}>
                Select Predictor Variables to Train Model
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  type="button"
                  onClick={() => setSelectedFeatures(availableFeatures)}
                  style={{
                    background: '#FFFFFF',
                    border: '1px solid #CBD5E1',
                    borderRadius: '6px',
                    padding: '4px 10px',
                    fontSize: '12px',
                    fontWeight: 600,
                    color: '#0F3D3E',
                    cursor: 'pointer',
                  }}
                >
                  Select All
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedFeatures([])}
                  style={{
                    background: '#FFFFFF',
                    border: '1px solid #CBD5E1',
                    borderRadius: '6px',
                    padding: '4px 10px',
                    fontSize: '12px',
                    fontWeight: 600,
                    color: '#648484',
                    cursor: 'pointer',
                  }}
                >
                  Clear All
                </button>
              </div>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {availableFeatures.map((col: string) => {
                const isChecked = selectedFeatures.includes(col);
                const colType = AUTO_TIME_FEATURES.includes(col) ? 'engineered' : (dataset.columns.find((c) => c.column === col)?.inferred_type || 'feature');
                return (
                  <label
                    key={col}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '6px 12px',
                      borderRadius: '8px',
                      border: isChecked ? '1px solid #FED7AA' : '1px solid #E2E8F0',
                      background: isChecked ? '#FFF7ED' : '#FFFFFF',
                      color: isChecked ? '#C2410C' : '#476D6D',
                      fontSize: '13px',
                      fontWeight: isChecked ? 600 : 400,
                      cursor: 'pointer',
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedFeatures((prev) => [...prev, col]);
                        } else {
                          setSelectedFeatures((prev) => prev.filter((c) => c !== col));
                        }
                      }}
                      style={{ cursor: 'pointer' }}
                    />
                    <span>{col}</span>
                    <span style={{ fontSize: '10px', opacity: 0.7 }}>({colType})</span>
                  </label>
                );
              })}
            </div>
          </div>
        )}
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
              borderRadius: '20px',
              padding: '26px 30px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '20px',
              boxShadow: '0 4px 20px rgba(15, 61, 62, 0.15)',
            }}
          >
            <div>
              <span
                style={{
                  fontSize: '12px',
                  fontWeight: 800,
                  textTransform: 'uppercase',
                  letterSpacing: '0.8px',
                  background: '#EA580C',
                  padding: '4px 12px',
                  borderRadius: '20px',
                }}
              >
                Champion Model
              </span>
              <h3 style={{ fontSize: '26px', fontWeight: 800, margin: '10px 0 6px' }}>
                {mlResult.best_model_name}
              </h3>
              <p style={{ margin: 0, fontSize: '14px', opacity: 0.9 }}>
                Target: <strong>{mlResult.target_column}</strong> &bull; Task:{' '}
                <strong>{mlResult.task_type.toUpperCase()}</strong> &bull; Samples:{' '}
                {mlResult.total_samples.toLocaleString()} ({mlResult.train_samples} train / {mlResult.test_samples} test)
              </p>
            </div>

            <div style={{ display: 'flex', gap: '20px', alignItems: 'center', flexWrap: 'wrap' }}>
              <div style={{ textAlign: 'center', background: 'rgba(255,255,255,0.12)', padding: '14px 20px', borderRadius: '14px' }}>
                <div style={{ fontSize: '28px', fontWeight: 800, color: '#FED7AA' }}>
                  {mlResult.best_model_metrics['accuracy'] ?? mlResult.best_model_metrics['r2_score'] ?? mlResult.leaderboard[0].test_score ?? '-'}
                </div>
                <div style={{ fontSize: '12px', textTransform: 'uppercase', opacity: 0.85, fontWeight: 600 }}>
                  Holdout Test Score
                </div>
              </div>
              <div style={{ textAlign: 'center', background: 'rgba(255,255,255,0.12)', padding: '14px 20px', borderRadius: '14px' }}>
                <div style={{ fontSize: '28px', fontWeight: 800, color: '#FFFFFF' }}>
                  {mlResult.leaderboard[0].val_score ?? '-'}
                </div>
                <div style={{ fontSize: '12px', textTransform: 'uppercase', opacity: 0.85, fontWeight: 600 }}>
                  Validation Score (CV)
                </div>
              </div>
              <div style={{ textAlign: 'center', background: 'rgba(255,255,255,0.12)', padding: '14px 20px', borderRadius: '14px' }}>
                <div style={{ fontSize: '28px', fontWeight: 800 }}>
                  {mlResult.leaderboard[0].train_score}
                </div>
                <div style={{ fontSize: '12px', textTransform: 'uppercase', opacity: 0.85, fontWeight: 600 }}>
                  Train Score
                </div>
              </div>
            </div>
          </div>

          {/* Model Tournament Leaderboard Table */}
          <div
            style={{
              background: '#FFFFFF',
              borderRadius: '20px',
              padding: '26px',
              border: '1px solid #EAE6DF',
            }}
          >
            <h4 style={{ fontSize: '20px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 8px' }}>
              Model Tournament Leaderboard
            </h4>
            <p style={{ margin: '0 0 16px', fontSize: '14px', color: '#648484' }}>
              Candidates ranked exclusively by development cross-validation score (Val Score). Untouched final test evaluation is performed solely on the selected champion model.
            </p>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
                <thead>
                  <tr style={{ background: '#FAF8F5', borderBottom: '2px solid #EAE6DF' }}>
                    <th style={{ padding: '12px 16px', textAlign: 'left', color: '#0F3D3E' }}>Rank & Model</th>
                    <th style={{ padding: '12px 16px', textAlign: 'left', color: '#0F3D3E' }}>Algorithm</th>
                    <th style={{ padding: '12px 16px', textAlign: 'center', color: '#0F3D3E' }}>Validation Score (CV)</th>
                    <th style={{ padding: '12px 16px', textAlign: 'center', color: '#0F3D3E' }}>Holdout Test Score</th>
                    <th style={{ padding: '12px 16px', textAlign: 'center', color: '#0F3D3E' }}>Train Score</th>
                    <th style={{ padding: '12px 16px', textAlign: 'center', color: '#0F3D3E' }}>Overfitting Gap</th>
                    <th style={{ padding: '12px 16px', textAlign: 'right', color: '#0F3D3E' }}>Duration</th>
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
                          borderBottom: '1px solid #F0EDE6',
                          background: isWinner ? '#FFF7ED' : 'transparent',
                        }}
                      >
                        <td style={{ padding: '14px 16px', fontWeight: 700, color: '#0F3D3E' }}>
                          #{idx + 1} {m.model_name}
                          {m.is_baseline && (
                            <span
                              style={{
                                marginLeft: '8px',
                                fontSize: '11px',
                                background: '#E2E8F0',
                                color: '#476D6D',
                                padding: '3px 8px',
                                borderRadius: '6px',
                              }}
                            >
                              Baseline
                            </span>
                          )}
                          {isWinner && (
                            <span
                              style={{
                                marginLeft: '8px',
                                fontSize: '11px',
                                background: '#EA580C',
                                color: '#FFFFFF',
                                padding: '3px 8px',
                                borderRadius: '6px',
                              }}
                            >
                              Winner
                            </span>
                          )}
                        </td>
                        <td style={{ padding: '14px 16px', color: '#476D6D' }}>{m.algorithm}</td>
                        <td style={{ padding: '14px 16px', textAlign: 'center', fontWeight: 700, color: '#0F3D3E' }}>
                          {m.val_score !== undefined ? m.val_score : '-'}
                        </td>
                        <td style={{ padding: '14px 16px', textAlign: 'center', color: m.test_score !== null && m.test_score !== undefined ? '#0F3D3E' : '#94A3B8', fontWeight: m.test_score !== null && m.test_score !== undefined ? 700 : 400 }}>
                          {m.test_score !== null && m.test_score !== undefined ? m.test_score : 'Held Out'}
                        </td>
                        <td style={{ padding: '14px 16px', textAlign: 'center', color: '#476D6D' }}>
                          {m.train_score}
                        </td>
                        <td style={{ padding: '14px 16px', textAlign: 'center' }}>
                          <span
                            style={{
                              fontSize: '12px',
                              fontWeight: 600,
                              padding: '3px 10px',
                              borderRadius: '6px',
                              background: isOverfit ? '#FEE2E2' : '#ECFDF5',
                              color: isOverfit ? '#B91C1C' : '#047857',
                            }}
                          >
                            {m.overfitting_gap > 0 ? `+${m.overfitting_gap}` : m.overfitting_gap} (
                            {isOverfit ? 'High' : 'Low'})
                          </span>
                        </td>
                        <td style={{ padding: '14px 16px', textAlign: 'right', color: '#648484' }}>
                          {m.training_time_ms} ms
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div style={{ marginTop: '16px', fontSize: '14px', color: '#476D6D' }}>
              <strong className="text-[#0F3D3E]">Generalization Assessment:</strong> {mlResult.overfitting_assessment}
            </div>

            {/* Plain-English Metric Explanations Guide */}
            {mlResult.metric_explanations && (
              <div style={{ background: '#FAF8F5', borderRadius: '14px', padding: '18px 22px', border: '1px solid #EAE6DF', marginTop: '18px' }}>
                <h5 style={{ fontSize: '15px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 12px' }}>
                  Metric Guide & Held-Out Scores
                </h5>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '12px' }}>
                  {Object.entries(mlResult.metric_explanations).map(([k, desc]) => (
                    <div key={k} style={{ background: '#FFFFFF', padding: '12px 16px', borderRadius: '10px', border: '1px solid #EAE6DF', fontSize: '14px' }}>
                      <strong style={{ color: '#EA580C', textTransform: 'uppercase', letterSpacing: '0.4px', display: 'block', marginBottom: '4px' }}>
                        {k.replace('_', ' ')}: {mlResult.best_model_metrics[k] !== undefined ? mlResult.best_model_metrics[k] : ''}
                      </strong>
                      <span style={{ color: '#476D6D', lineHeight: 1.5 }}>{desc}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Validation & Lineage Notice */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '14px', background: '#F0FDF4', color: '#166534', border: '1px solid #BBF7D0', padding: '12px 18px', borderRadius: '14px' }}>
            <span style={{ fontWeight: 700, background: '#DCFCE7', padding: '3px 10px', borderRadius: '6px' }}>VERIFIED</span>
            <span><strong>Out-of-Sample Holdout:</strong> {mlResult.verification_status} &bull; <em>Strategy: {mlResult.validation_strategy}</em></span>
          </div>

          {/* Grid: Feature Importance on Left, What-If on Right */}
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '24px' }}>
            {/* Feature Importance */}
            <div
              style={{
                background: '#FFFFFF',
                borderRadius: '20px',
                padding: '24px',
                border: '1px solid #EAE6DF',
              }}
            >
              <h4 style={{ fontSize: '20px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 16px' }}>
                Feature Importance
              </h4>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                {mlResult.feature_importance.map((f, i) => (
                  <div key={f.feature}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px', marginBottom: '6px' }}>
                      <span style={{ fontWeight: 600, color: '#0F3D3E' }}>
                        #{i + 1} {f.feature}
                      </span>
                      <span style={{ fontWeight: 700, color: '#EA580C' }}>{f.percentage}%</span>
                    </div>
                    <div style={{ height: '8px', background: '#FAF8F5', borderRadius: '4px', overflow: 'hidden', border: '1px solid #EAE6DF' }}>
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
                borderRadius: '20px',
                padding: '24px',
                border: '1px solid #EAE6DF',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h4 style={{ fontSize: '20px', fontWeight: 700, color: '#0F3D3E', margin: 0 }}>
                  Scenario Simulator
                </h4>
                <span
                  style={{
                    fontSize: '12px',
                    fontWeight: 700,
                    color: '#EA580C',
                    background: '#FFF7ED',
                    padding: '3px 10px',
                    borderRadius: '8px',
                    border: '1px solid #FED7AA',
                  }}
                >
                  Interactive
                </span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '18px' }}>
                {Object.keys(whatIfInputs).map((feat) => (
                  <div key={feat} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
                    <label style={{ fontSize: '14px', fontWeight: 600, color: '#0F3D3E', minWidth: '130px' }}>
                      {feat}:
                    </label>
                    <input
                      type="text"
                      value={whatIfInputs[feat] ?? ''}
                      onChange={(e) => handleInputChange(feat, e.target.value)}
                      style={{
                        padding: '8px 12px',
                        borderRadius: '8px',
                        border: '1px solid #CBD5E1',
                        fontSize: '14px',
                        color: '#0F3D3E',
                        width: '100%',
                        maxWidth: '180px',
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
                  borderRadius: '12px',
                  padding: '12px',
                  fontSize: '15px',
                  fontWeight: 700,
                  cursor: isSimulating ? 'not-allowed' : 'pointer',
                  opacity: isSimulating ? 0.7 : 1,
                  marginBottom: '16px',
                  boxShadow: '0 2px 8px rgba(234, 88, 12, 0.25)',
                }}
              >
                {isSimulating ? 'Simulating Prediction...' : 'Simulate Scenario Outcome'}
              </button>

              {/* What-If Output Card */}
              {whatIfResult && (
                <div
                  style={{
                    background: '#FAF8F5',
                    padding: '18px',
                    borderRadius: '14px',
                    border: '1px solid #EAE6DF',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <span style={{ fontSize: '14px', fontWeight: 600, color: '#648484' }}>
                      Predicted Result:
                    </span>
                    <span style={{ fontSize: '18px', fontWeight: 800, color: '#0F3D3E' }}>
                      {String(whatIfResult.prediction)}
                    </span>
                  </div>

                  {whatIfResult.prediction_delta !== undefined && whatIfResult.prediction_delta !== null && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px', color: '#648484', marginBottom: '8px' }}>
                      <span>Delta vs Baseline ({whatIfResult.baseline_prediction}):</span>
                      <strong style={{ color: whatIfResult.prediction_delta >= 0 ? '#059669' : '#DC2626' }}>
                        {whatIfResult.prediction_delta > 0 ? `+${whatIfResult.prediction_delta}` : whatIfResult.prediction_delta}
                      </strong>
                    </div>
                  )}

                  <p style={{ fontSize: '14px', color: '#476D6D', margin: '8px 0', lineHeight: 1.5 }}>
                    {whatIfResult.explanation}
                  </p>

                  <div
                    style={{
                      background: '#FFFBEB',
                      border: '1px solid #FDE68A',
                      color: '#92400E',
                      padding: '10px 14px',
                      borderRadius: '8px',
                      fontSize: '13px',
                      marginTop: '10px',
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
                borderRadius: '20px',
                padding: '24px',
                border: '1px solid #EAE6DF',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h4 style={{ fontSize: '20px', fontWeight: 700, color: '#0F3D3E', margin: 0 }}>
                  Holdout Test Evaluation
                </h4>
                <span
                  style={{
                    fontSize: '12px',
                    fontWeight: 700,
                    color: '#6D28D9',
                    background: '#F5F3FF',
                    border: '1px solid #DDD6FE',
                    padding: '4px 12px',
                    borderRadius: '8px',
                  }}
                >
                  Holdout Validation
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: mlResult.residual_chart_spec ? 'repeat(auto-fit, minmax(320px, 1fr))' : (mlResult.chart_spec && mlResult.evaluation_chart_spec ? '1fr 1fr' : '1fr'), gap: '20px' }}>
                {mlResult.evaluation_chart_spec && (
                  <div style={{ background: '#FAF8F5', padding: '16px', borderRadius: '12px', border: '1px solid #EAE6DF' }}>
                    <PlotlyChart spec={mlResult.evaluation_chart_spec} plotlySpec={mlResult.evaluation_chart_spec.plotly_spec} height={320} />
                  </div>
                )}
                {mlResult.residual_chart_spec && (
                  <div style={{ background: '#FAF8F5', padding: '16px', borderRadius: '12px', border: '1px solid #EAE6DF' }}>
                    <PlotlyChart spec={mlResult.residual_chart_spec} plotlySpec={mlResult.residual_chart_spec.plotly_spec} height={320} />
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
                borderRadius: '20px',
                padding: '24px',
                border: '1px solid #EAE6DF',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h4 style={{ fontSize: '20px', fontWeight: 700, color: '#0F3D3E', margin: 0 }}>
                  Held-Out Predictions
                </h4>
                <span
                  style={{
                    fontSize: '12px',
                    fontWeight: 700,
                    color: '#0D9488',
                    background: '#F0FDFA',
                    border: '1px solid #CCECE6',
                    padding: '4px 12px',
                    borderRadius: '8px',
                  }}
                >
                  Leakage-Safe Holdout
                </span>
              </div>

              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
                  <thead>
                    <tr style={{ background: '#FAF8F5', borderBottom: '2px solid #EAE6DF' }}>
                      <th style={{ padding: '12px 14px', textAlign: 'left', color: '#0F3D3E' }}>Row #</th>
                      <th style={{ padding: '12px 14px', textAlign: 'left', color: '#0F3D3E' }}>Actual Ground Truth</th>
                      <th style={{ padding: '12px 14px', textAlign: 'left', color: '#0F3D3E' }}>Model Prediction</th>
                      <th style={{ padding: '12px 14px', textAlign: 'center', color: '#0F3D3E' }}>
                        {mlResult.task_type === 'classification' ? 'Outcome' : 'Residual (Error)'}
                      </th>
                      <th style={{ padding: '12px 14px', textAlign: 'left', color: '#0F3D3E' }}>Sample Features</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mlResult.held_out_predictions.map((p) => {
                      const isCorrect = p.is_correct;
                      return (
                        <tr key={p.row_index} style={{ borderBottom: '1px solid #F0EDE6' }}>
                          <td style={{ padding: '12px 14px', fontWeight: 600, color: '#648484' }}>
                            #{p.row_index}
                          </td>
                          <td style={{ padding: '12px 14px', fontWeight: 700, color: '#0F3D3E' }}>
                            {String(p.actual)}
                          </td>
                          <td style={{ padding: '12px 14px', fontWeight: 700, color: '#EA580C' }}>
                            {String(p.predicted)}
                          </td>
                          <td style={{ padding: '12px 14px', textAlign: 'center' }}>
                            {mlResult.task_type === 'classification' ? (
                              <span
                                style={{
                                  fontSize: '12px',
                                  fontWeight: 700,
                                  padding: '3px 10px',
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
                                  fontSize: '13px',
                                  fontFamily: 'monospace',
                                  fontWeight: 600,
                                  color: p.error_or_residual && p.error_or_residual > 0 ? '#B91C1C' : '#047857',
                                }}
                              >
                                {p.error_or_residual !== undefined && p.error_or_residual !== null
                                  ? (p.error_or_residual > 0 ? `+${p.error_or_residual}` : `${p.error_or_residual}`)
                                  : '-'}
                              </span>
                            )}
                          </td>
                          <td style={{ padding: '12px 14px', fontSize: '13px', color: '#648484' }}>
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
                padding: '22px 26px',
                border: '1px solid #EAE6DF',
              }}
            >
              <h5 style={{ fontSize: '16px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 10px' }}>
                Model Limitations & Scientific Caveats
              </h5>
              <ul style={{ margin: 0, paddingLeft: '22px', fontSize: '14px', color: '#476D6D', display: 'flex', flexDirection: 'column', gap: '8px', lineHeight: 1.6 }}>
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
