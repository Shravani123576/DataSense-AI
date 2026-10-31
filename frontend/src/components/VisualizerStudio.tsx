import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  BarChart3,
  LineChart as LineChartIcon,
  PieChart as PieIcon,
  ScatterChart,
  Sparkles,
  Play,
  SlidersHorizontal,
  Bot,
  Send,
  X,
  ChevronDown,
  ArrowRight,
  MessageSquare,
  RefreshCw,
  AlertCircle,
  Lightbulb
} from 'lucide-react';
import type { ProfileReport, ChartSpec } from '../types';
import { generateCustomChart, askGraphAi, type GraphIntelligenceAskResponse, type StructuredAnalystAnswer } from '../api';
import { PlotlyChart } from './PlotlyChart';

export interface GraphChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  structuredAnswer?: StructuredAnalystAnswer | null;
  timestamp: string;
  response?: GraphIntelligenceAskResponse;
  chartTitle?: string;
  isError?: boolean;
  failedQuestion?: string;
}

interface Props {
  profile: ProfileReport;
  apiKey?: string;
}

export const VisualizerStudio: React.FC<Props> = ({ profile, apiKey }) => {
  const { columns, metadata, top_correlations } = profile;

  const numericCols = columns.filter((c) => c.inferred_type === 'numeric').map((c) => c.column);
  const catCols = columns.filter((c) => ['categorical', 'boolean', 'text'].includes(c.inferred_type)).map((c) => c.column);
  const allCols = columns.map((c) => c.column);

  const [chartType, setChartType] = useState<'bar' | 'line' | 'scatter' | 'histogram' | 'pie'>('bar');
  const [xCol, setXCol] = useState<string>(catCols[0] || allCols[0] || '');
  const [yCol, setYCol] = useState<string>(numericCols[0] || '');
  const [aggregation, setAggregation] = useState<string>('mean');
  const [currentSpec, setCurrentSpec] = useState<ChartSpec | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // "Ask about this graph" Chatbot state
  const [showGraphAi, setShowGraphAi] = useState(true);
  const [chatMessages, setChatMessages] = useState<GraphChatMessage[]>([]);
  const [graphQuestion, setGraphQuestion] = useState('');
  const [isAskingAi, setIsAskingAi] = useState(false);
  const [chartNotice, setChartNotice] = useState<string | null>(null);
  const prevSpecSigRef = useRef<string | null>(null);
  const activeChartSigRef = useRef<string>('');
  const [expandedTraces, setExpandedTraces] = useState<Record<string, boolean>>({});
  const [conversationId] = useState<string>(() => 'conv-' + Math.random().toString(36).substring(2, 9));

  const toggleTrace = (id: string) => {
    setExpandedTraces((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  // Recommended preset charts
  const [presetSpecs, setPresetSpecs] = useState<ChartSpec[]>([]);

  const [topN, setTopN] = useState<number>(20);
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc' | 'none'>('desc');
  const [orientation, setOrientation] = useState<'auto' | 'h' | 'v'>('auto');

  const handleBuildChart = async () => {
    if (!xCol) return;
    setIsLoading(true);
    setErrorMsg(null);

    try {
      const spec = await generateCustomChart({
        dataset_id: metadata.id,
        x_column: xCol,
        y_column: chartType === 'histogram' ? undefined : (yCol || undefined),
        chart_type: chartType,
        aggregation: aggregation,
        top_n: topN,
        sort_order: sortOrder,
        orientation: orientation,
      });
      setCurrentSpec(spec);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to render chart.');
    } finally {
      setIsLoading(false);
    }
  };

  // Generate smart preset charts when dataset changes
  useEffect(() => {
    const loadPresets = async () => {
      const presets: ChartSpec[] = [];
      try {
        if (catCols.length > 0) {
          const spec1 = await generateCustomChart({
            dataset_id: metadata.id,
            x_column: catCols[0],
            chart_type: 'bar',
            aggregation: 'count',
          });
          presets.push(spec1);
        }
        if (top_correlations.length > 0) {
          const pair = top_correlations[0];
          const spec2 = await generateCustomChart({
            dataset_id: metadata.id,
            x_column: pair.col1,
            y_column: pair.col2,
            chart_type: 'scatter',
          });
          presets.push(spec2);
        } else if (numericCols.length > 0) {
          const spec2 = await generateCustomChart({
            dataset_id: metadata.id,
            x_column: numericCols[0],
            chart_type: 'histogram',
          });
          presets.push(spec2);
        }
        setPresetSpecs(presets);
        if (presets.length > 0 && !currentSpec) {
          setCurrentSpec(presets[0]);
        }
      } catch (e) {
        console.error('Error generating preset charts', e);
      }
    };

    loadPresets();
  }, [metadata.id]);

  // Context-aware suggested questions for active chart
  const suggestedQuestions = useMemo(() => {
    if (!currentSpec) return [];
    const questions: string[] = ['What does this graph show?'];

    const isTemporal =
      currentSpec.chart_type === 'line' ||
      columns.find((c) => c.column === currentSpec.x_axis)?.inferred_type === 'datetime';

    if (isTemporal) {
      questions.push('What is the main trend?');
      questions.push('Why does the slope become steeper during this period?');
      questions.push('What is the rate of change per year or decade?');
      questions.push('How do two selected periods compare?');
      if ((currentSpec.data?.length || 0) >= 6) {
        questions.push('Is there evidence of seasonal behaviour?');
      }
      questions.push('Is the observed trend statistically significant?');
    } else if (currentSpec.chart_type === 'bar' || currentSpec.chart_type === 'pie') {
      questions.push('Which categories have the highest and lowest values?');
      if (currentSpec.y_axis) {
        questions.push('Which groups have the highest and lowest average values?');
        questions.push('Is the group difference statistically significant?');
      }
      questions.push('Are there unusual observations?');
    } else if (currentSpec.chart_type === 'scatter') {
      questions.push('What is the relationship between these variables?');
      questions.push('Are there unusual observations?');
      questions.push('Is the observed correlation statistically significant?');
    } else if (currentSpec.chart_type === 'histogram') {
      questions.push('What is the shape and spread of the distribution?');
      questions.push('Are there unusual observations?');
    }

    questions.push('What further analysis would be useful?');
    return questions;
  }, [currentSpec, columns]);

  // Track active chart signature and clear stale answers when chart changes
  useEffect(() => {
    if (currentSpec) {
      const sig = `${currentSpec.chart_type}:${currentSpec.x_axis}:${currentSpec.y_axis || ''}:${currentSpec.aggregation || ''}`;
      if (prevSpecSigRef.current && prevSpecSigRef.current !== sig) {
        setChatMessages([]);
        setChartNotice(`Active graph updated to ${currentSpec.title || currentSpec.chart_type}. Previous answers cleared.`);
        activeChartSigRef.current = sig;
      } else if (!prevSpecSigRef.current) {
        activeChartSigRef.current = sig;
      }
      prevSpecSigRef.current = sig;
    }
  }, [currentSpec]);

  const handleAskGraphAi = async (customQ?: string) => {
    const q = (customQ || graphQuestion).trim();
    if (!q) return;
    setChartNotice(null);

    const effSpec = currentSpec || {
      chart_type: chartType || 'bar',
      title: `${xCol || allCols[0] || 'Dataset'} Distribution`,
      x_axis: xCol || allCols[0] || 'x',
      y_axis: yCol || undefined,
      aggregation: aggregation || 'mean',
    };

    const userMsgId = `usr-${Date.now()}`;
    const userMsg: GraphChatMessage = {
      id: userMsgId,
      role: 'user',
      content: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      chartTitle: effSpec.title,
    };

    const nextHistory = [...chatMessages, userMsg];
    setChatMessages(nextHistory);
    setGraphQuestion('');
    setIsAskingAi(true);
    setShowGraphAi(true);

    const currentSig = `${effSpec.chart_type}:${effSpec.x_axis}:${effSpec.y_axis || ''}:${effSpec.aggregation || 'mean'}`;

    // Cleanly resolve underlying column names from axes/spec
    let resolvedX = (effSpec.x_axis || xCol || 'x').replace(/\s+Bins$/i, '').trim();
    if (!allCols.includes(resolvedX) && allCols.length > 0) {
      const matchX = allCols.find((c) => c.toLowerCase() === resolvedX.toLowerCase());
      if (matchX) resolvedX = matchX;
    }

    let resolvedY: string | null = null;
    if (effSpec.chart_type !== 'histogram') {
      if (yCol && allCols.includes(yCol)) {
        resolvedY = yCol;
      } else if (effSpec.y_axis) {
        const rawY = effSpec.y_axis.replace(/^(Mean|Sum|Max|Min|Avg|Average|Count|Frequency)\s+/i, '').trim();
        if (allCols.includes(rawY)) {
          resolvedY = rawY;
        } else {
          const matchY = allCols.find((c) => c.toLowerCase() === rawY.toLowerCase());
          if (matchY) resolvedY = matchY;
          else if (rawY.toLowerCase().includes('count') || rawY.toLowerCase().includes('frequency')) resolvedY = null;
          else resolvedY = rawY;
        }
      }
    }

    try {
      const resp = await askGraphAi({
        dataset_id: metadata.id,
        chart_spec: effSpec as any,
        x_column: resolvedX,
        y_column: resolvedY,
        chart_type: effSpec.chart_type,
        aggregation: effSpec.aggregation || aggregation || 'mean',
        filters: null,
        conversation_history: nextHistory.map((m) => ({ role: m.role, content: m.content })),
        conversation_id: conversationId,
        chart_signature: currentSig,
        dataset_version: (metadata as any).version ?? 0,
        question: q,
        api_key: apiKey || localStorage.getItem('datasense_gemini_key') || localStorage.getItem('datasense_gemini_api_key') || undefined,
      });

      // Update signature on successful request
      activeChartSigRef.current = currentSig;

      let answerObj: StructuredAnalystAnswer | null = null;
      let displayFinding = '';
      if (typeof resp.answer === 'object' && resp.answer !== null) {
        answerObj = resp.answer;
        displayFinding = resp.answer.finding;
      } else {
        displayFinding = String(resp.answer);
      }

      const assistantMsg: GraphChatMessage = {
        id: `asst-${Date.now()}`,
        role: 'assistant',
        content: displayFinding,
        structuredAnswer: answerObj,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        response: resp,
        chartTitle: effSpec.title,
      };

      setChatMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      const errorMsg: GraphChatMessage = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content: `Analysis Unavailable: ${err.message || 'Unable to compute graph intelligence for the selected features.'}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        isError: true,
        failedQuestion: q,
        chartTitle: effSpec.title,
      };
      setChatMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsAskingAi(false);
    }
  };

  const handleResetInvestigation = () => {
    setChatMessages([]);
    setGraphQuestion('');
    if (currentSpec) {
      activeChartSigRef.current = `${currentSpec.chart_type}:${currentSpec.x_axis}:${currentSpec.y_axis || ''}:${currentSpec.aggregation || ''}`;
    }
  };

  return (
    <div className="space-y-6">
      {/* Chart Builder Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Sidebar */}
        <div className="bg-white p-6 rounded-3xl border border-[#EAE6DF] shadow-xs space-y-5">
          <div className="flex items-center gap-2 pb-3 border-b border-[#F0EDE6]">
            <SlidersHorizontal className="w-5 h-5 text-[#EA580C]" />
            <h3 className="font-heading font-bold text-lg text-[#0F3D3E]">
              Chart Controls
            </h3>
          </div>

          {/* Chart Type Selector */}
          <div className="space-y-2">
            <label className="text-sm font-semibold text-[#0F3D3E]">
              Chart Type
            </label>
            <div className="grid grid-cols-5 gap-2 p-1.5 bg-[#FAF8F5] rounded-2xl border border-[#EAE6DF]">
              {[
                { type: 'bar', icon: BarChart3, label: 'Bar' },
                { type: 'line', icon: LineChartIcon, label: 'Line' },
                { type: 'scatter', icon: ScatterChart, label: 'Scatter' },
                { type: 'pie', icon: PieIcon, label: 'Donut' },
                { type: 'histogram', icon: BarChart3, label: 'Hist' },
              ].map((item) => {
                const Icon = item.icon;
                const isSelected = chartType === item.type;
                return (
                  <button
                    key={item.type}
                    onClick={() => setChartType(item.type as any)}
                    className={`flex flex-col items-center justify-center py-2.5 px-1 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
                      isSelected
                        ? 'bg-white text-[#EA580C] shadow-xs'
                        : 'text-[#648484] hover:text-[#0F3D3E]'
                    }`}
                  >
                    <Icon className="w-5 h-5 mb-1" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* X Axis Column */}
          <div className="space-y-2">
            <label className="text-sm font-semibold text-[#0F3D3E]">
              {chartType === 'histogram' ? 'Target Feature' : 'X Axis / Dimension'}
            </label>
            <select
              value={xCol}
              onChange={(e) => setXCol(e.target.value)}
              className="w-full bg-[#FAF8F5] border border-[#EAE6DF] rounded-xl px-3.5 py-2.5 text-sm text-[#0F3D3E] font-medium focus:outline-none focus:border-[#EA580C]"
            >
              {allCols.map((c) => (
                <option key={c} value={c}>
                  {c} ({columns.find((col) => col.column === c)?.inferred_type})
                </option>
              ))}
            </select>
          </div>

          {/* Y Axis Column (if not histogram) */}
          {chartType !== 'histogram' && (
            <div className="space-y-2">
              <label className="text-sm font-semibold text-[#0F3D3E]">
                Y Axis Metric
              </label>
              <select
                value={yCol}
                onChange={(e) => setYCol(e.target.value)}
                className="w-full bg-[#FAF8F5] border border-[#EAE6DF] rounded-xl px-3.5 py-2.5 text-sm text-[#0F3D3E] font-medium focus:outline-none focus:border-[#EA580C]"
              >
                <option value="">(Frequency Count of X)</option>
                {numericCols.map((c) => (
                  <option key={c} value={c}>
                    {c} (numeric)
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Aggregation (if Y Axis is selected) */}
          {chartType !== 'histogram' && chartType !== 'scatter' && yCol && (
            <div className="space-y-2">
              <label className="text-sm font-semibold text-[#0F3D3E]">
                Aggregation
              </label>
              <div className="grid grid-cols-4 gap-2">
                {['mean', 'sum', 'max', 'min'].map((agg) => (
                  <button
                    key={agg}
                    onClick={() => setAggregation(agg)}
                    className={`py-2 text-sm font-semibold rounded-xl border transition-all cursor-pointer capitalize ${
                      aggregation === agg
                        ? 'bg-[#FFF7ED] text-[#EA580C] border-[#FED7AA]'
                        : 'bg-[#FAF8F5] text-[#476D6D] border-[#EAE6DF] hover:bg-white'
                    }`}
                  >
                    {agg}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Top N Category Limit */}
          {chartType !== 'scatter' && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm font-semibold text-[#0F3D3E]">
                <span>Category Limit</span>
                <span className="text-xs font-normal text-[#648484]">{topN === 0 ? 'All Categories' : `Top ${topN}`}</span>
              </div>
              <div className="grid grid-cols-4 gap-2">
                {[
                  { label: 'Top 10', val: 10 },
                  { label: 'Top 20', val: 20 },
                  { label: 'Top 50', val: 50 },
                  { label: 'All', val: 0 },
                ].map((item) => (
                  <button
                    key={item.val}
                    type="button"
                    onClick={() => setTopN(item.val)}
                    className={`py-1.5 text-xs font-semibold rounded-xl border transition-all cursor-pointer ${
                      topN === item.val
                        ? 'bg-[#FFF7ED] text-[#EA580C] border-[#FED7AA]'
                        : 'bg-[#FAF8F5] text-[#476D6D] border-[#EAE6DF] hover:bg-white'
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Sort Order */}
          {chartType !== 'scatter' && chartType !== 'histogram' && (
            <div className="space-y-2">
              <label className="text-sm font-semibold text-[#0F3D3E]">
                Sort Categories
              </label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { id: 'desc', label: 'Highest' },
                  { id: 'asc', label: 'Lowest' },
                  { id: 'none', label: 'Original' },
                ].map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setSortOrder(item.id as any)}
                    className={`py-1.5 text-xs font-semibold rounded-xl border transition-all cursor-pointer ${
                      sortOrder === item.id
                        ? 'bg-[#FFF7ED] text-[#EA580C] border-[#FED7AA]'
                        : 'bg-[#FAF8F5] text-[#476D6D] border-[#EAE6DF] hover:bg-white'
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Bar Orientation */}
          {chartType === 'bar' && (
            <div className="space-y-2">
              <label className="text-sm font-semibold text-[#0F3D3E]">
                Bar Orientation
              </label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { id: 'auto', label: 'Auto (>=6 H)' },
                  { id: 'h', label: 'Horizontal' },
                  { id: 'v', label: 'Vertical' },
                ].map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setOrientation(item.id as any)}
                    className={`py-1.5 text-xs font-semibold rounded-xl border transition-all cursor-pointer ${
                      orientation === item.id
                        ? 'bg-[#FFF7ED] text-[#EA580C] border-[#FED7AA]'
                        : 'bg-[#FAF8F5] text-[#476D6D] border-[#EAE6DF] hover:bg-white'
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Render Action Button */}
          <button
            onClick={handleBuildChart}
            disabled={isLoading || !xCol}
            className="w-full bg-[#EA580C] hover:bg-[#C2410C] text-white py-3 rounded-xl font-semibold text-base flex items-center justify-center gap-2 transition-all shadow-xs cursor-pointer disabled:opacity-50"
          >
            <Play className="w-4 h-4 fill-current" />
            <span>{isLoading ? 'Generating Chart...' : 'Render Chart'}</span>
          </button>

          {errorMsg && (
            <div className="p-3 bg-[#FEF2F2] rounded-xl border border-[#FCA5A5] text-sm text-[#B91C1C]">
              {errorMsg}
            </div>
          )}
        </div>

        {/* Chart Stage Area */}
        <div className="bg-white p-6 rounded-3xl border border-[#EAE6DF] shadow-xs flex flex-col justify-between lg:col-span-2 space-y-4">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-[#F0EDE6] flex-wrap gap-3">
              <div className="flex items-center gap-3 flex-wrap">
                <h3 className="font-heading font-bold text-xl text-[#0F3D3E]">
                  {currentSpec ? currentSpec.title : 'Chart View'}
                </h3>
                {currentSpec?.is_subset && (
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-[#FFF7ED] text-[#EA580C] border border-[#FED7AA]">
                    Subset: Top {currentSpec.displayed_categories} of {currentSpec.total_categories}
                  </span>
                )}
              </div>

              {/* "Ask about this graph" Assistant Toggle */}
              <button
                onClick={() => setShowGraphAi(!showGraphAi)}
                id="graph-ai-assistant-toggle"
                className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold transition-all cursor-pointer shadow-xs ${
                  showGraphAi
                    ? 'bg-[#0F3D3E] text-white'
                    : 'bg-[#FFF7ED] text-[#EA580C] border border-[#FED7AA] hover:bg-[#FED7AA]/30'
                }`}
                title="Ask questions about this graph with grounded data-science evidence"
              >
                <MessageSquare className="w-4 h-4 text-[#EA580C]" />
                <span>Ask about this graph</span>
                {chatMessages.length > 0 && (
                  <span className="w-5 h-5 rounded-full bg-[#EA580C] text-white text-[10px] font-bold flex items-center justify-center">
                    {chatMessages.length}
                  </span>
                )}
                {showGraphAi ? <X className="w-4 h-4 ml-1" /> : <ChevronDown className="w-4 h-4 ml-1" />}
              </button>
            </div>

            {/* Low-frequency distribution banner */}
            {currentSpec?.low_frequency_note && (
              <div className="my-3 p-3 bg-[#F0FDFA] rounded-xl border border-[#99F6E4] text-xs text-[#0F766E] flex items-center gap-2">
                <span className="text-sm">ℹ️</span>
                <span><strong>Data Note:</strong> {currentSpec.low_frequency_note}</span>
              </div>
            )}

            {/* Plotly Canvas Container (Primary Visual Element on Top) */}
            <div className="mt-4 min-h-[380px] flex items-center justify-center">
              {currentSpec ? (
                <PlotlyChart spec={currentSpec} plotlySpec={currentSpec.plotly_spec} height={380} />
              ) : (
                <div className="text-center text-sm text-[#648484] space-y-2">
                  <BarChart3 className="w-8 h-8 text-[#A5C0C0] mx-auto animate-bounce" />
                  <p>Select features and click "Render Chart" to explore.</p>
                </div>
              )}
            </div>

            {/* "Ask about this graph" Dedicated Chat Panel (Directly Under Graph) */}
            {showGraphAi ? (
              <div className="mt-6 p-5 rounded-2xl bg-[#FAF8F5] border border-[#EAE6DF] space-y-4 animate-in fade-in duration-200">
                <div className="flex items-center justify-between pb-3 border-b border-[#EAE6DF] flex-wrap gap-2">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-xl bg-[#0F3D3E] text-white flex items-center justify-center">
                      <Bot className="w-4 h-4 text-[#FED7AA]" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h4 className="font-heading font-bold text-base text-[#0F3D3E]">
                          Ask about this graph
                        </h4>
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-[#E6F4EA] text-[#137333] border border-[#CEEAD6]">
                          V{(metadata as any).version ?? 0} Grounded
                        </span>
                      </div>
                      <p className="text-xs text-[#648484]">
                        Context: <span className="font-medium text-[#0F3D3E]">{currentSpec?.title || 'Selected Chart'}</span> ({currentSpec?.chart_type} • X: {currentSpec?.x_axis}{currentSpec?.y_axis ? `, Y: ${currentSpec.y_axis}` : ''})
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={handleResetInvestigation}
                      title="Start a new investigation thread"
                      className="bg-white hover:bg-[#F3F0EB] text-[#476D6D] hover:text-[#0F3D3E] px-3 py-1.5 rounded-xl border border-[#EAE6DF] text-xs font-semibold transition-all flex items-center gap-1.5 cursor-pointer shadow-2xs"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      <span>New Investigation</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowGraphAi(false)}
                      title="Collapse Assistant"
                      className="text-[#648484] hover:text-[#0F3D3E] p-1.5 rounded-lg hover:bg-white transition-all cursor-pointer"
                    >
                      <ChevronDown className="w-4 h-4 rotate-180" />
                    </button>
                  </div>
                </div>

                {/* Chart Signature Change Notice */}
                {chartNotice && (
                  <div className="p-2.5 bg-[#FFFBEB] border border-[#FDE68A] rounded-xl text-xs text-[#92400E] flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 shrink-0" />
                      <span>{chartNotice}</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => setChartNotice(null)}
                      className="font-bold cursor-pointer text-[#B45309] hover:text-[#78350F] px-1"
                    >
                      ✕
                    </button>
                  </div>
                )}

                {/* Conversation Message List */}
                <div className="space-y-3.5 max-h-96 overflow-y-auto pr-1">
                  {chatMessages.length === 0 ? (
                    <div className="p-4 bg-white rounded-xl border border-[#EAE6DF] text-center space-y-2">
                      <MessageSquare className="w-6 h-6 text-[#A5C0C0] mx-auto" />
                      <p className="text-xs text-[#648484]">
                        Ask any question about this graph. The AI Analyst computes evidence directly from the active dataset.
                      </p>
                    </div>
                  ) : (
                    chatMessages.map((msg) => (
                      <div
                        key={msg.id}
                        className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
                      >
                        {msg.role === 'user' ? (
                          <div className="max-w-[85%] bg-[#0F3D3E] text-white px-4 py-2.5 rounded-2xl rounded-tr-xs text-xs font-medium shadow-2xs">
                            {msg.content}
                          </div>
                        ) : (
                          <div className="max-w-[95%] w-full bg-white rounded-2xl rounded-tl-xs border border-[#CCECE6] p-4 text-xs text-[#0F3D3E] space-y-3 shadow-2xs">
                            {/* Assistant Message Header */}
                            <div className="flex items-center justify-between pb-2 border-b border-[#F0EDE6] flex-wrap gap-2">
                              <div className="flex items-center gap-2">
                                <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                  msg.response?.mode === 'llm'
                                    ? 'bg-[#ECFDF5] text-[#059669] border border-[#A7F3D0]'
                                    : 'bg-[#F0FDF4] text-[#15803D] border border-[#BBF7D0]'
                                }`}>
                                  {msg.response?.mode === 'llm' ? '🤖 LLM' : '⚡ Fallback'}
                                </span>
                                {msg.response?.grounded && (
                                  <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold text-emerald-800 bg-emerald-50 border border-emerald-200">
                                    ✓ Grounded
                                  </span>
                                )}
                              </div>
                              <span className="text-[10px] text-[#A5C0C0]">{msg.timestamp}</span>
                            </div>

                            {/* Error state with retry */}
                            {msg.isError ? (
                              <div className="flex items-center justify-between gap-2 text-xs text-[#B91C1C] bg-[#FEF2F2] p-2.5 rounded-xl border border-[#FCA5A5]">
                                <div className="flex items-center gap-1.5">
                                  <AlertCircle className="w-4 h-4 shrink-0 text-[#DC2626]" />
                                  <span>{msg.content}</span>
                                </div>
                                {msg.failedQuestion && (
                                  <button
                                    type="button"
                                    onClick={() => handleAskGraphAi(msg.failedQuestion)}
                                    disabled={isAskingAi}
                                    className="shrink-0 text-xs font-bold text-white bg-[#DC2626] hover:bg-[#B91C1C] px-2.5 py-1 rounded-lg transition-colors cursor-pointer"
                                  >
                                    Retry
                                  </button>
                                )}
                              </div>
                            ) : (
                              <>
                                {/* Structured Content: Four-part structure with guaranteed content */}
                                <div className="space-y-2.5 text-xs text-[#0F3D3E]">
                                  {/* 1. Finding */}
                                  <div>
                                    <strong className="text-[11px] uppercase tracking-wider text-[#476D6D] block mb-0.5">Finding:</strong>
                                    <div className="leading-relaxed font-medium">
                                      {msg.structuredAnswer?.finding || msg.content || 'Not computed: No primary analytical finding available.'}
                                    </div>
                                  </div>

                                  {/* 2. Calculated Evidence */}
                                  <div>
                                    <strong className="text-[11px] uppercase tracking-wider text-[#476D6D] block mb-0.5">Calculated Evidence:</strong>
                                    {msg.structuredAnswer?.evidence && msg.structuredAnswer.evidence.length > 0 ? (
                                      <ul className="list-disc pl-4 space-y-0.5 text-xs">
                                        {msg.structuredAnswer.evidence.map((ev, i) => (
                                          <li key={i}>{ev.replace(/^-\s*/, '')}</li>
                                        ))}
                                      </ul>
                                    ) : (
                                      <div className="text-[#648484] italic">Not computed: Direct calculation returned no auxiliary metrics for this question.</div>
                                    )}
                                  </div>

                                  {/* 3. Methodology */}
                                  <div>
                                    <strong className="text-[11px] uppercase tracking-wider text-[#476D6D] block mb-0.5">Methodology:</strong>
                                    <div className="text-[#476D6D]">
                                      {msg.structuredAnswer?.method || (
                                        <span className="text-[#648484] italic">Not computed: Direct deterministic calculation from active chart parameters.</span>
                                      )}
                                    </div>
                                  </div>

                                  {/* 4. Limitations */}
                                  <div>
                                    <strong className="text-[11px] uppercase tracking-wider text-[#476D6D] block mb-0.5">Limitations:</strong>
                                    {msg.structuredAnswer?.limitations && msg.structuredAnswer.limitations.length > 0 ? (
                                      <ul className="list-disc pl-4 space-y-0.5 text-[#648484]">
                                        {msg.structuredAnswer.limitations.map((lim, i) => (
                                          <li key={i}>{lim.replace(/^-\s*/, '')}</li>
                                        ))}
                                      </ul>
                                    ) : (
                                      <div className="text-[#648484] italic">Not computed: Observational sample boundaries and assumptions apply.</div>
                                    )}
                                  </div>
                                </div>

                                {/* Collapsible "Tools used (N)" row expanding to trace table */}
                                {msg.response?.trace && msg.response.trace.length > 0 && (
                                  <div className="pt-1 border-t border-[#F0EDE6]">
                                    <button
                                      type="button"
                                      onClick={() => toggleTrace(msg.id)}
                                      className="text-[11px] font-semibold text-[#EA580C] hover:text-[#C2410C] flex items-center gap-1 cursor-pointer"
                                    >
                                      <span>Tools used ({msg.response.trace.length})</span>
                                      <ChevronDown className={`w-3.5 h-3.5 transition-transform ${expandedTraces[msg.id] ? 'rotate-180' : ''}`} />
                                    </button>
                                    {expandedTraces[msg.id] && (
                                      <div className="mt-2 overflow-x-auto border border-[#EAE6DF] rounded-xl bg-[#FAF8F5] p-2 text-[10px]">
                                        <table className="w-full text-left">
                                          <thead>
                                            <tr className="border-b border-[#EAE6DF] text-[#476D6D]">
                                              <th className="py-1 px-1.5 font-semibold">Step</th>
                                              <th className="py-1 px-1.5 font-semibold">Tool</th>
                                              <th className="py-1 px-1.5 font-semibold">Duration</th>
                                              <th className="py-1 px-1.5 font-semibold">Output Summary</th>
                                            </tr>
                                          </thead>
                                          <tbody>
                                            {msg.response.trace.map((step) => (
                                              <tr key={step.step} className="border-b border-[#F0EDE6] last:border-0">
                                                <td className="py-1 px-1.5 font-bold text-[#0F3D3E]">{step.step}</td>
                                                <td className="py-1 px-1.5 font-mono text-[#EA580C]">{step.tool}</td>
                                                <td className="py-1 px-1.5 text-[#648484]">{step.duration_ms}ms</td>
                                                <td className="py-1 px-1.5 text-[#0F3D3E]">{step.output_summary}</td>
                                              </tr>
                                            ))}
                                          </tbody>
                                        </table>
                                      </div>
                                    )}
                                  </div>
                                )}

                                {/* 5. Suggested Next Step */}
                                {((msg.structuredAnswer && msg.structuredAnswer.next_step) || msg.response?.suggested_next_step) && (
                                  <div className="pt-2 border-t border-[#F0EDE6] flex items-center justify-between gap-2 bg-[#FAF8F5] p-2.5 rounded-xl border border-[#EAE6DF]">
                                    <div className="text-[11px] text-[#476D6D]">
                                      <strong className="text-[#0F3D3E]">💡 Suggested Next Step:</strong> {msg.structuredAnswer?.next_step || msg.response?.suggested_next_step}
                                    </div>
                                    <button
                                      type="button"
                                      onClick={() => handleAskGraphAi(msg.structuredAnswer?.next_step || msg.response!.suggested_next_step!)}
                                      disabled={isAskingAi}
                                      className="shrink-0 text-xs text-[#EA580C] hover:text-[#C2410C] font-semibold flex items-center gap-1 cursor-pointer bg-white px-2.5 py-1 rounded-lg border border-[#FED7AA] shadow-2xs transition-all disabled:opacity-50"
                                    >
                                      <span>Investigate</span>
                                      <ArrowRight className="w-3 h-3" />
                                    </button>
                                  </div>
                                )}
                              </>
                            )}
                          </div>
                        )}
                      </div>
                    ))
                  )}

                  {/* Loading Bubble */}
                  {isAskingAi && (
                    <div className="flex items-center gap-2 p-3 bg-[#F0FDF4] rounded-xl border border-[#BBF7D0] text-xs text-[#166534] w-fit">
                      <span className="w-2 h-2 rounded-full bg-[#16A34A] animate-ping" />
                      <span>Running tools…</span>
                    </div>
                  )}
                </div>

                {/* Context-Aware Suggested Question Pills */}
                <div className="space-y-1.5 pt-1">
                  <div className="flex items-center gap-1.5 text-[11px] font-bold text-[#476D6D]">
                    <Lightbulb className="w-3.5 h-3.5 text-[#EA580C]" />
                    <span>Suggested Questions for this {currentSpec?.chart_type} chart:</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto pr-1">
                    {suggestedQuestions.map((sq, idx) => (
                      <button
                        key={idx}
                        type="button"
                        disabled={isAskingAi}
                        onClick={() => handleAskGraphAi(sq)}
                        className="px-2.5 py-1 rounded-lg bg-white border border-[#EAE6DF] hover:border-[#EA580C] hover:text-[#EA580C] text-[11px] font-medium text-[#0F3D3E] transition-all cursor-pointer disabled:opacity-50 shadow-2xs"
                      >
                        {sq}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Question Input Bar */}
                <div className="flex items-center gap-2 pt-1">
                  <input
                    type="text"
                    placeholder="Ask a question about this graph or follow-up on previous findings..."
                    value={graphQuestion}
                    onChange={(e) => setGraphQuestion(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !isAskingAi && handleAskGraphAi()}
                    disabled={isAskingAi}
                    className="flex-1 bg-white border border-[#EAE6DF] rounded-xl px-4 py-2 text-xs text-[#0F3D3E] focus:outline-none focus:border-[#EA580C] disabled:bg-[#FAF8F5]"
                  />
                  <button
                    type="button"
                    onClick={() => handleAskGraphAi()}
                    disabled={isAskingAi || !graphQuestion.trim()}
                    className="bg-[#EA580C] hover:bg-[#C2410C] text-white px-3.5 py-2 rounded-xl text-xs font-semibold transition-all cursor-pointer shadow-xs disabled:opacity-50 flex items-center gap-1.5"
                  >
                    <span>Ask</span>
                    <Send className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            ) : (
              <div
                onClick={() => setShowGraphAi(true)}
                className="mt-4 p-3.5 rounded-2xl bg-[#FFF7ED] border border-[#FED7AA] hover:bg-[#FFEDD5] transition-all cursor-pointer flex items-center justify-between"
              >
                <div className="flex items-center gap-2.5 text-xs text-[#C2410C] font-semibold">
                  <Bot className="w-4 h-4 text-[#EA580C]" />
                  <span>Click to open Graph AI Analyst chatbot for this chart</span>
                </div>
                <div className="flex items-center gap-1 text-xs text-[#EA580C] font-bold">
                  <span>Open Analyst</span>
                  <ChevronDown className="w-3.5 h-3.5" />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Auto-Discovered Patterns / Preset Charts Row */}
      {presetSpecs.length > 0 && (
        <div className="bg-white p-6 rounded-3xl border border-[#EAE6DF] shadow-xs space-y-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-[#EA580C]" />
            <h3 className="font-heading font-bold text-xl text-[#0F3D3E]">
              Suggested Charts
            </h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {presetSpecs.map((pSpec, idx) => (
              <div
                key={idx}
                onClick={() => {
                  setCurrentSpec(pSpec);
                  setChatMessages([]);
                }}
                className="p-5 rounded-2xl bg-[#FAF8F5] border border-[#EAE6DF] hover:border-[#EA580C]/50 transition-all cursor-pointer space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-sm text-[#0F3D3E]">{pSpec.title}</span>
                  <span className="text-xs bg-white text-[#476D6D] border border-[#EAE6DF] px-2.5 py-1 rounded-full capitalize font-medium">
                    {pSpec.chart_type}
                  </span>
                </div>
                <PlotlyChart spec={pSpec} plotlySpec={pSpec.plotly_spec} height={200} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

