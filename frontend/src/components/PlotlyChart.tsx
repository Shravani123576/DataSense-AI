import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import type { ChartSpec } from '../types';

interface PlotlyChartProps {
  spec?: ChartSpec | null;
  plotlySpec?: Record<string, any> | null;
  height?: number;
  className?: string;
}

export const PlotlyChart: React.FC<PlotlyChartProps> = ({
  spec,
  plotlySpec,
  height = 360,
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;

    let data: any[] = [];
    let layout: any = {
      autosize: true,
      height: height,
      margin: { t: 40, r: 24, b: 50, l: 56 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Inter, sans-serif', color: '#0F3D3E' },
      xaxis: {
        gridcolor: '#EFECE6',
        tickfont: { color: '#476D6D', size: 11 },
        title: { font: { color: '#0F3D3E', size: 12 } },
        zeroline: false,
      },
      yaxis: {
        gridcolor: '#EFECE6',
        tickfont: { color: '#476D6D', size: 11 },
        title: { font: { color: '#0F3D3E', size: 12 } },
        zeroline: false,
      },
      hoverlabel: {
        bgcolor: '#0F3D3E',
        font: { color: '#FFFFFF', family: 'Inter, sans-serif' },
      },
    };

    if (plotlySpec && plotlySpec.data) {
      data = plotlySpec.data;
      layout = {
        ...layout,
        ...(plotlySpec.layout || {}),
        height: height,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
      };
    } else if (spec) {
      const title = spec.title || '';
      const cType = spec.chart_type;

      if (cType === 'scatter' && spec.data) {
        data = [
          {
            type: 'scatter',
            mode: 'markers',
            x: spec.data.map((d) => d.x),
            y: spec.data.map((d) => d.y),
            marker: {
              size: 8,
              color: '#EA580C',
              opacity: 0.8,
              line: { color: '#FFFFFF', width: 1 },
            },
            name: title,
          },
        ];
      } else if (cType === 'pie' && spec.data) {
        data = [
          {
            type: 'pie',
            labels: spec.data.map((d) => String(d.x)),
            values: spec.data.map((d) => d.y),
            hole: 0.42,
            marker: {
              colors: [
                '#EA580C',
                '#0D9488',
                '#F59E0B',
                '#6366F1',
                '#EC4899',
                '#14B8A6',
                '#8B5CF6',
                '#10B981',
                '#3B82F6',
              ],
            },
            textinfo: 'label+percent',
            hoverinfo: 'label+value+percent',
          },
        ];
      } else if (cType === 'line' && spec.data) {
        data = [
          {
            type: 'scatter',
            mode: 'lines+markers',
            x: spec.data.map((d) => d.x),
            y: spec.data.map((d) => d.y),
            line: { color: '#EA580C', width: 3, shape: 'spline' },
            marker: { color: '#0F3D3E', size: 6 },
            name: spec.y_axis || 'Value',
          },
        ];
      } else if (spec.data) {
        data = [
          {
            type: 'bar',
            x: spec.data.map((d) => d.x),
            y: spec.data.map((d) => d.y),
            marker: {
              color: '#EA580C',
              opacity: 0.9,
            },
            name: spec.y_axis || 'Value',
          },
        ];
      }

      layout.title = {
        text: title,
        font: { family: 'Inter, sans-serif', size: 15, color: '#0F3D3E' },
      };
      if (spec.x_axis) layout.xaxis.title.text = spec.x_axis;
      if (spec.y_axis) layout.yaxis.title.text = spec.y_axis;
    }

    const config = {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'] as any,
      toImageButtonOptions: {
        format: 'png' as const,
        filename: 'datasense_chart',
        height: 600,
        width: 900,
        scale: 2,
      },
    };

    Plotly.react(container, data, layout, config as any);


    const handleResize = () => {
      if (container) {
        Plotly.Plots.resize(container);
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      Plotly.purge(container);
    };
  }, [spec, plotlySpec, height]);

  return (
    <div className={`w-full overflow-hidden ${className}`}>
      <div ref={containerRef} style={{ width: '100%', height: `${height}px` }} />
      {spec && (spec.data_stage || spec.rows_used || spec.transformations) && (
        <div
          style={{
            marginTop: '8px',
            padding: '8px 12px',
            background: '#FAF8F5',
            borderRadius: '8px',
            border: '1px solid #EAE6DF',
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            gap: '8px',
            fontSize: '11px',
            color: '#476D6D',
          }}
        >
          {spec.data_stage && (
            <span
              style={{
                fontWeight: 700,
                padding: '2px 8px',
                borderRadius: '6px',
                textTransform: 'uppercase',
                letterSpacing: '0.4px',
                fontSize: '10px',
                background:
                  spec.data_stage === 'model_ready_held_out'
                    ? '#F5F3FF'
                    : spec.data_stage === 'cleaned'
                    ? '#ECFDF5'
                    : spec.data_stage === 'raw'
                    ? '#F1F5F9'
                    : '#E6F4F1',
                color:
                  spec.data_stage === 'model_ready_held_out'
                    ? '#6D28D9'
                    : spec.data_stage === 'cleaned'
                    ? '#065F46'
                    : spec.data_stage === 'raw'
                    ? '#475569'
                    : '#0F3D3E',
                border: `1px solid ${
                  spec.data_stage === 'model_ready_held_out'
                    ? '#DDD6FE'
                    : spec.data_stage === 'cleaned'
                    ? '#A7F3D0'
                    : spec.data_stage === 'raw'
                    ? '#CBD5E1'
                    : '#B2DFDB'
                }`,
              }}
            >
              Stage: {spec.data_stage.replace(/_/g, ' ')}
            </span>
          )}

          {spec.rows_used !== undefined && spec.rows_used !== null && (
            <span style={{ fontWeight: 600 }}>
              &bull; Rows: <strong>{spec.rows_used.toLocaleString()}</strong>
            </span>
          )}

          {spec.transformations && (
            <span style={{ color: '#64748B' }}>
              &bull; {spec.transformations}
            </span>
          )}

          {spec.exclusions && (
            <span style={{ color: '#D97706', fontStyle: 'italic' }}>
              &bull; {spec.exclusions}
            </span>
          )}
        </div>
      )}
    </div>
  );
};
