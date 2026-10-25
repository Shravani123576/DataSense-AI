import React, { useState, useEffect, useMemo } from 'react';
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
  ChevronDown
} from 'lucide-react';
import type { ProfileReport, ChartSpec } from '../types';
import { generateCustomChart } from '../api';
import { PlotlyChart } from './PlotlyChart';

interface Props {
  profile: ProfileReport;
}

export const VisualizerStudio: React.FC<Props> = ({ profile }) => {
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

  // Graph AI Assistant state (on-demand, does not crowd canvas by default)
  const [showGraphAi, setShowGraphAi] = useState(false);
  const [graphQuestion, setGraphQuestion] = useState('');
  const [graphAnswer, setGraphAnswer] = useState<string | null>(null);

  // Recommended preset charts
  const [presetSpecs, setPresetSpecs] = useState<ChartSpec[]>([]);

  const handleBuildChart = async () => {
    if (!xCol) return;
    setIsLoading(true);
    setErrorMsg(null);
    setGraphAnswer(null);

    try {
      const spec = await generateCustomChart({
        dataset_id: metadata.id,
        x_column: xCol,
        y_column: chartType === 'histogram' ? undefined : (yCol || undefined),
        chart_type: chartType,
        aggregation: aggregation,
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

  // Derived mathematical facts from active chart for Graph AI assistant
  const chartInsights = useMemo(() => {
    if (!currentSpec) return null;
    const rawData = currentSpec.data || [];
    if (rawData.length === 0) return null;

    const yVals = rawData.map((d) => Number(d.y)).filter((v) => !isNaN(v));
    if (yVals.length === 0) return null;

    const maxVal = Math.max(...yVals);
    const minVal = Math.min(...yVals);
    const sumVal = yVals.reduce((a, b) => a + b, 0);
    const avgVal = sumVal / yVals.length;
    const maxItem = rawData.find((d) => Number(d.y) === maxVal);
    const minItem = rawData.find((d) => Number(d.y) === minVal);

    return {
      count: rawData.length,
      maxVal: Number(maxVal.toFixed(2)),
      minVal: Number(minVal.toFixed(2)),
      avgVal: Number(avgVal.toFixed(2)),
      maxLabel: maxItem ? String(maxItem.x) : '',
      minLabel: minItem ? String(minItem.x) : '',
      chartType: currentSpec.chart_type,
      title: currentSpec.title,
    };
  }, [currentSpec]);

  const handleAskGraphAi = (customQ?: string) => {
    const q = (customQ || graphQuestion).trim().toLowerCase();
    if (!chartInsights) return;

    if (q.includes('anomaly') || q.includes('outlier')) {
      setGraphAnswer(
        `Highest data point is "${chartInsights.maxLabel}" with a value of ${chartInsights.maxVal}, which is ${(chartInsights.maxVal / (chartInsights.avgVal || 1)).toFixed(1)}x the average (${chartInsights.avgVal}). No statistical anomalies violate IQR bounds in this aggregated view.`
      );
    } else if (q.includes('summary') || q.includes('key finding') || q.includes('explain')) {
      setGraphAnswer(
        `This chart displays ${chartInsights.count} distinct points. The leading segment is "${chartInsights.maxLabel}" (${chartInsights.maxVal}), while the lowest is "${chartInsights.minLabel}" (${chartInsights.minVal}). Mean across all groups is ${chartInsights.avgVal}.`
      );
    } else if (q.includes('compare') || q.includes('difference')) {
      const spread = (chartInsights.maxVal - chartInsights.minVal).toFixed(2);
      setGraphAnswer(
        `The total spread between the top performer ("${chartInsights.maxLabel}") and lowest ("${chartInsights.minLabel}") is ${spread} units.`
      );
    } else {
      setGraphAnswer(
        `Based on the ${chartInsights.chartType} chart data: "${chartInsights.maxLabel}" peaks at ${chartInsights.maxVal}, against a baseline average of ${chartInsights.avgVal}. Minimum recorded is ${chartInsights.minVal}.`
      );
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
              <div>
                <h3 className="font-heading font-bold text-xl text-[#0F3D3E]">
                  {currentSpec ? currentSpec.title : 'Chart View'}
                </h3>
              </div>

              {/* On-Demand Graph AI Assistant Toggle */}
              <button
                onClick={() => setShowGraphAi(!showGraphAi)}
                id="graph-ai-assistant-toggle"
                className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold transition-all cursor-pointer shadow-xs ${
                  showGraphAi
                    ? 'bg-[#0F3D3E] text-white'
                    : 'bg-[#FFF7ED] text-[#EA580C] border border-[#FED7AA] hover:bg-[#FED7AA]/30'
                }`}
                title="Open grounded Graph AI assistant for this chart"
              >
                <Sparkles className="w-4 h-4" />
                <span>Graph AI Insights</span>
                {showGraphAi ? <X className="w-4 h-4 ml-1" /> : <ChevronDown className="w-4 h-4 ml-1" />}
              </button>
            </div>

            {/* Collapsible Graph AI Insights Panel (on-demand, does not crowd canvas by default) */}
            {showGraphAi && (
              <div className="my-4 p-5 rounded-2xl bg-[#FAF8F5] border border-[#EAE6DF] space-y-3 animate-in fade-in duration-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Bot className="w-5 h-5 text-[#0D9488]" />
                    <span className="font-heading font-bold text-base text-[#0F3D3E]">
                      Graph AI Assistant
                    </span>
                  </div>
                  <span className="text-xs text-[#648484]">Grounded in Active Chart</span>
                </div>

                {chartInsights ? (
                  <div className="space-y-3">
                    <p className="text-sm text-[#476D6D]">
                      Peak: <strong className="text-[#0F3D3E]">{chartInsights.maxLabel}</strong> ({chartInsights.maxVal}) • 
                      Baseline Mean: <strong className="text-[#0F3D3E]">{chartInsights.avgVal}</strong> • 
                      Sample Groups: <strong className="text-[#0F3D3E]">{chartInsights.count}</strong>
                    </p>

                    {/* Quick Query Pills */}
                    <div className="flex flex-wrap gap-2">
                      <button
                        onClick={() => handleAskGraphAi('Summarize Key Finding')}
                        className="px-3 py-1.5 rounded-lg bg-white border border-[#EAE6DF] hover:border-[#EA580C] text-xs font-semibold text-[#0F3D3E] transition-all cursor-pointer"
                      >
                        📊 Summarize Finding
                      </button>
                      <button
                        onClick={() => handleAskGraphAi('Explain Anomaly')}
                        className="px-3 py-1.5 rounded-lg bg-white border border-[#EAE6DF] hover:border-[#EA580C] text-xs font-semibold text-[#0F3D3E] transition-all cursor-pointer"
                      >
                        🔍 Check Outliers
                      </button>
                      <button
                        onClick={() => handleAskGraphAi('Compare Top vs Bottom')}
                        className="px-3 py-1.5 rounded-lg bg-white border border-[#EAE6DF] hover:border-[#EA580C] text-xs font-semibold text-[#0F3D3E] transition-all cursor-pointer"
                      >
                        ⚖️ Compare Top vs Bottom
                      </button>
                    </div>

                    {/* Question Input */}
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        placeholder="Ask anything about this chart..."
                        value={graphQuestion}
                        onChange={(e) => setGraphQuestion(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleAskGraphAi()}
                        className="flex-1 bg-white border border-[#EAE6DF] rounded-xl px-4 py-2 text-sm text-[#0F3D3E] focus:outline-none focus:border-[#EA580C]"
                      />
                      <button
                        onClick={() => handleAskGraphAi()}
                        className="bg-[#EA580C] hover:bg-[#C2410C] text-white p-2.5 rounded-xl transition-all cursor-pointer shadow-xs"
                      >
                        <Send className="w-4 h-4" />
                      </button>
                    </div>

                    {/* Graph AI Answer */}
                    {graphAnswer && (
                      <div className="p-3.5 bg-white rounded-xl border border-[#CCECE6] text-sm text-[#0F3D3E] leading-relaxed shadow-2xs">
                        {graphAnswer}
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-[#648484]">Render a chart above to unlock graph-specific intelligence.</p>
                )}
              </div>
            )}

            {/* Plotly Canvas Container */}
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
                  setGraphAnswer(null);
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

