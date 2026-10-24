import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  LineChart as LineChartIcon,
  PieChart as PieIcon,
  ScatterChart,
  Sparkles,
  Play,
  SlidersHorizontal
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

  // Recommended preset charts
  const [presetSpecs, setPresetSpecs] = useState<ChartSpec[]>([]);

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
      });
      setCurrentSpec(spec);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to render chart.');
    } finally {
      setIsLoading(false);
    }
  };

  // Generate 2 smart auto-preset charts when dataset changes
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

  return (
    <div className="space-y-6">
      {/* Chart Builder Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Sidebar */}
        <div className="bg-white p-6 rounded-3xl border border-[#EAE6DF] shadow-xs space-y-5">
          <div className="flex items-center gap-2 pb-3 border-b border-[#F0EDE6]">
            <SlidersHorizontal className="w-5 h-5 text-[#EA580C]" />
            <h3 className="font-heading font-bold text-base text-[#0F3D3E]">
              Custom Plotly Builder
            </h3>
          </div>

          {/* Chart Type Selector */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-[#476D6D] uppercase tracking-wider">
              Chart Type
            </label>
            <div className="grid grid-cols-5 gap-1.5 p-1 bg-[#FAF8F5] rounded-2xl border border-[#EAE6DF]">
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
                    className={`flex flex-col items-center justify-center py-2 px-1 rounded-xl text-xs font-medium transition-all cursor-pointer ${
                      isSelected
                        ? 'bg-white text-[#EA580C] shadow-xs font-bold'
                        : 'text-[#648484] hover:text-[#0F3D3E]'
                    }`}
                  >
                    <Icon className="w-4 h-4 mb-0.5" />
                    <span className="text-[10px]">{item.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* X Axis Column */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-[#476D6D] uppercase tracking-wider">
              {chartType === 'histogram' ? 'Target Feature (Numeric)' : 'X Axis / Grouping Dimension'}
            </label>
            <select
              value={xCol}
              onChange={(e) => setXCol(e.target.value)}
              className="w-full bg-[#FAF8F5] border border-[#EAE6DF] rounded-xl px-3 py-2 text-xs text-[#0F3D3E] font-medium focus:outline-none focus:border-[#EA580C]"
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
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#476D6D] uppercase tracking-wider">
                Y Axis Metric (Numeric)
              </label>
              <select
                value={yCol}
                onChange={(e) => setYCol(e.target.value)}
                className="w-full bg-[#FAF8F5] border border-[#EAE6DF] rounded-xl px-3 py-2 text-xs text-[#0F3D3E] font-medium focus:outline-none focus:border-[#EA580C]"
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
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#476D6D] uppercase tracking-wider">
                Aggregation Function
              </label>
              <div className="grid grid-cols-4 gap-1.5">
                {['mean', 'sum', 'max', 'min'].map((agg) => (
                  <button
                    key={agg}
                    onClick={() => setAggregation(agg)}
                    className={`py-1.5 text-xs font-medium rounded-xl border transition-all cursor-pointer capitalize ${
                      aggregation === agg
                        ? 'bg-[#FFF7ED] text-[#EA580C] border-[#FED7AA] font-bold'
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
            className="w-full bg-[#EA580C] hover:bg-[#C2410C] text-white py-2.5 rounded-xl font-medium text-xs md:text-sm flex items-center justify-center gap-2 transition-all shadow-xs cursor-pointer disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>{isLoading ? 'Plotting with Pandas...' : 'Render Interactive Plotly Chart'}</span>
          </button>

          {errorMsg && (
            <div className="p-3 bg-[#FEF2F2] rounded-xl border border-[#FCA5A5] text-xs text-[#B91C1C]">
              {errorMsg}
            </div>
          )}
        </div>

        {/* Chart Stage Area */}
        <div className="bg-white p-6 rounded-3xl border border-[#EAE6DF] shadow-xs flex flex-col justify-between lg:col-span-2 space-y-4">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-[#F0EDE6]">
              <div>
                <h3 className="font-heading font-bold text-lg text-[#0F3D3E]">
                  {currentSpec ? currentSpec.title : 'Interactive Chart Canvas'}
                </h3>
                {currentSpec?.description && (
                  <p className="text-xs text-[#648484]">{currentSpec.description}</p>
                )}
              </div>
              <span className="text-xs font-semibold text-[#0D9488] bg-[#F0FDFA] px-3 py-1 rounded-full border border-[#CCECE6]">
                Plotly 7.1
              </span>
            </div>

            {/* Plotly Canvas Container */}
            <div className="mt-4 min-h-[380px] flex items-center justify-center">
              {currentSpec ? (
                <PlotlyChart spec={currentSpec} plotlySpec={currentSpec.plotly_spec} height={380} />
              ) : (
                <div className="text-center text-xs text-[#648484] space-y-2">
                  <BarChart3 className="w-8 h-8 text-[#A5C0C0] mx-auto animate-bounce" />
                  <p>Select your features and click "Render Interactive Plotly Chart" to begin exploring.</p>
                </div>
              )}
            </div>
          </div>

          <div className="pt-3 border-t border-[#F0EDE6] flex items-center justify-between text-xs text-[#648484]">
            <span>Tip: Use Plotly's toolbar in the top right to zoom, pan, or download as PNG/SVG.</span>
          </div>
        </div>
      </div>

      {/* Auto-Discovered Patterns / Preset Charts Row */}
      {presetSpecs.length > 0 && (
        <div className="bg-white p-6 rounded-3xl border border-[#EAE6DF] shadow-xs space-y-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-[#EA580C]" />
            <h3 className="font-heading font-bold text-lg text-[#0F3D3E]">
              Auto-Discovered Pattern Charts
            </h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {presetSpecs.map((pSpec, idx) => (
              <div
                key={idx}
                onClick={() => setCurrentSpec(pSpec)}
                className="p-4 rounded-2xl bg-[#FAF8F5] border border-[#EAE6DF] hover:border-[#EA580C]/50 transition-all cursor-pointer space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-xs text-[#0F3D3E]">{pSpec.title}</span>
                  <span className="text-[10px] bg-white text-[#476D6D] border border-[#EAE6DF] px-2 py-0.5 rounded-full capitalize">
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
