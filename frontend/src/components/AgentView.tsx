import React, { useState } from 'react';
import type { ProfileReport, AgentGoalResponse } from '../types';
import { executeAgentGoal } from '../api';

interface AgentViewProps {
  dataset: ProfileReport;
  onNavigateToTab?: (tab: string) => void;
}

export const AgentView: React.FC<AgentViewProps> = ({ dataset }) => {
  const [goal, setGoal] = useState<string>('');
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [agentResponse, setAgentResponse] = useState<AgentGoalResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const presetGoals = [
    `Audit data health, deduplicate, and identify top predictive factors`,
    `Train an AutoML classification tournament and compare model architectures`,
    `Examine statistical correlations, outliers, and prepare an executive briefing`,
  ];

  const handleExecute = async (goalText?: string, clarificationChoice?: string) => {
    const targetGoal = goalText || goal;
    if (!targetGoal.trim()) return;

    setIsRunning(true);
    setErrorMsg(null);

    try {
      const res = await executeAgentGoal({
        dataset_id: dataset.metadata.id,
        goal: targetGoal,
        clarification_response: clarificationChoice,
      });
      setAgentResponse(res);
    } catch (err: any) {
      setErrorMsg(err.message || 'Agent execution failed.');
    } finally {
      setIsRunning(false);
    }
  };

  const handleClarificationSelect = (choice: string) => {
    if (!agentResponse) return;
    handleExecute(agentResponse.goal, choice);
  };

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
            Autonomous Agent Orchestration
          </span>
          <span style={{ fontSize: '13px', color: '#64748B' }}>
            Multi-Step Planning & Verified Tool Execution
          </span>
        </div>
        <h2 style={{ fontSize: '22px', fontWeight: 700, color: '#0F3D3E', margin: '8px 0 4px' }}>
          Data Science Copilot & Goal Planner
        </h2>
        <p style={{ fontSize: '14px', color: '#64748B', margin: '0 0 20px' }}>
          State your data science objective in plain English. The agent formulates a step-by-step plan, asks for clarification when parameters are missing, executes verified tools, and summarizes results.
        </p>

        {/* Goal Input Box */}
        <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
          <input
            type="text"
            placeholder="e.g. Audit data health, remove duplicates, and build a model to predict..."
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleExecute()}
            style={{
              flex: 1,
              padding: '12px 18px',
              borderRadius: '12px',
              border: '1.5px solid #CBD5E1',
              fontSize: '14px',
              color: '#1E293B',
              outline: 'none',
              background: '#FAF8F5',
            }}
          />
          <button
            onClick={() => handleExecute()}
            disabled={isRunning || !goal.trim()}
            style={{
              background: '#0F3D3E',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: '12px',
              padding: '12px 24px',
              fontSize: '14px',
              fontWeight: 700,
              cursor: isRunning ? 'not-allowed' : 'pointer',
              opacity: isRunning ? 0.7 : 1,
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              boxShadow: '0 2px 8px rgba(15, 61, 62, 0.2)',
            }}
          >
            {isRunning ? (
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
                Executing Plan...
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
                Execute Goal
              </>
            )}
          </button>
        </div>

        {/* Preset Suggestions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '12px', color: '#94A3B8', fontWeight: 600 }}>Try objective:</span>
          {presetGoals.map((p, idx) => (
            <button
              key={idx}
              onClick={() => {
                setGoal(p);
                handleExecute(p);
              }}
              style={{
                background: '#FAF8F5',
                border: '1px solid #E2E8F0',
                borderRadius: '8px',
                padding: '6px 12px',
                fontSize: '12px',
                color: '#475569',
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              &ldquo;{p}&rdquo;
            </button>
          ))}
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

      {/* Clarification Required Card */}
      {agentResponse && agentResponse.requires_clarification && (
        <div
          style={{
            background: '#FFFBEB',
            border: '2px solid #F59E0B',
            borderRadius: '16px',
            padding: '24px',
            marginBottom: '24px',
            boxShadow: '0 4px 16px rgba(245, 158, 11, 0.1)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <span style={{ fontSize: '20px' }}>&bull;</span>
            <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: '#92400E' }}>
              Clarification Needed to Proceed
            </h3>
          </div>
          <p style={{ fontSize: '14px', color: '#78350F', margin: '0 0 16px', lineHeight: 1.5 }}>
            {agentResponse.clarification_question}
          </p>

          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            {agentResponse.clarification_options?.map((opt) => (
              <button
                key={opt}
                onClick={() => handleClarificationSelect(opt)}
                style={{
                  background: '#EA580C',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '10px',
                  padding: '8px 16px',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  boxShadow: '0 2px 6px rgba(234, 88, 12, 0.25)',
                }}
              >
                Target: {opt}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Execution Plan View */}
      {agentResponse && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Plan Steps Card */}
          <div
            style={{
              background: '#FFFFFF',
              borderRadius: '16px',
              padding: '24px',
              border: '1px solid #E2E8F0',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '17px', fontWeight: 700, color: '#0F3D3E', margin: 0 }}>
                Step-by-Step Analysis Plan
              </h3>
              <span
                style={{
                  fontSize: '12px',
                  fontWeight: 600,
                  padding: '4px 10px',
                  borderRadius: '6px',
                  background: agentResponse.plan.is_completed ? '#ECFDF5' : '#FEF3C7',
                  color: agentResponse.plan.is_completed ? '#047857' : '#B45309',
                }}
              >
                Status: {agentResponse.plan.status.replace(/_/g, ' ').toUpperCase()}
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {agentResponse.plan.steps.map((step) => {
                const isDone = step.status === 'completed';
                const isFail = step.status === 'failed';

                return (
                  <div
                    key={step.id}
                    style={{
                      borderLeft: `4px solid ${isDone ? '#059669' : isFail ? '#DC2626' : '#CBD5E1'}`,
                      background: '#FAF8F5',
                      padding: '16px',
                      borderRadius: '0 12px 12px 0',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '12px', fontWeight: 700, color: '#64748B' }}>
                          Step {step.step_number}
                        </span>
                        <strong style={{ fontSize: '14px', color: '#1E293B' }}>{step.title}</strong>
                        <span
                          style={{
                            fontSize: '11px',
                            background: '#E2E8F0',
                            color: '#475569',
                            padding: '2px 6px',
                            borderRadius: '4px',
                            fontWeight: 600,
                          }}
                        >
                          tool: {step.tool_name}
                        </span>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {step.execution_time_ms && (
                          <span style={{ fontSize: '11px', color: '#94A3B8' }}>{step.execution_time_ms} ms</span>
                        )}
                        <span
                          style={{
                            fontSize: '11px',
                            fontWeight: 700,
                            padding: '2px 8px',
                            borderRadius: '4px',
                            textTransform: 'uppercase',
                            background: isDone ? '#D1FAE5' : isFail ? '#FEE2E2' : '#F1F5F9',
                            color: isDone ? '#065F46' : isFail ? '#991B1B' : '#64748B',
                          }}
                        >
                          {step.status}
                        </span>
                      </div>
                    </div>

                    {step.result_summary && (
                      <p style={{ margin: '6px 0 0', fontSize: '13px', color: '#475569', lineHeight: 1.5 }}>
                        {step.result_summary}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Key Findings Card */}
          {agentResponse.findings && agentResponse.findings.length > 0 && (
            <div
              style={{
                background: '#FFFFFF',
                borderRadius: '16px',
                padding: '24px',
                border: '1px solid #E2E8F0',
              }}
            >
              <h3 style={{ fontSize: '17px', fontWeight: 700, color: '#0F3D3E', margin: '0 0 14px' }}>
                Verified Key Findings
              </h3>
              <ul style={{ margin: 0, paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {agentResponse.findings.map((finding, idx) => (
                  <li key={idx} style={{ fontSize: '14px', color: '#334155', lineHeight: 1.5 }}>
                    {finding}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
