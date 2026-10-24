import React, { useState } from 'react';
import type { ChartSpec } from '../types';
import { Download, BarChart2, TrendingUp, PieChart, Info } from 'lucide-react';

interface Props {
  spec: ChartSpec;
  height?: number;
}

export const InteractiveChart: React.FC<Props> = ({ spec, height = 320 }) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const data = spec.data || [];

  if (!data || data.length === 0) {
    return (
      <div className="chart-empty-state" style={{ height }}>
        <Info size={24} className="text-muted" />
        <p>No numeric values available to display for this chart configuration.</p>
      </div>
    );
  }

  const exportSvg = () => {
    const svgElem = document.getElementById(`chart-${spec.title.replace(/\s+/g, '-')}`);
    if (!svgElem) return;
    const svgData = new XMLSerializer().serializeToString(svgElem);
    const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${spec.title.toLowerCase().replace(/[^a-z0-9]/g, '_')}.svg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const chartId = `chart-${spec.title.replace(/\s+/g, '-')}`;
  const width = 640;
  const padding = { top: 30, right: 30, bottom: 65, left: 65 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  // Chart type logic
  const isPie = spec.chart_type === 'pie';
  const isScatter = spec.chart_type === 'scatter';
  const isLine = spec.chart_type === 'line';

  // Math for Bars / Lines / Scatter
  const yValues = data.map(d => Number(d.y) || 0);
  const minY = Math.min(0, Math.min(...yValues));
  const maxY = Math.max(...yValues, 1);
  const rangeY = maxY - minY || 1;

  const getYPos = (val: number) => {
    return padding.top + plotHeight - ((val - minY) / rangeY) * plotHeight;
  };

  // 1. PIE / DONUT RENDERING
  if (isPie) {
    const total = data.reduce((acc, curr) => acc + (Number(curr.y) || 0), 0);
    let cumulativeAngle = 0;
    const radius = Math.min(plotWidth, plotHeight) / 2.2;
    const innerRadius = radius * 0.55;
    const centerX = width / 2;
    const centerY = height / 2;

    const colors = [
      '#6366f1', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6',
      '#06b6d4', '#14b8a6', '#f97316', '#3b82f6', '#a855f7'
    ];

    const pieSlices = data.slice(0, 8).map((d, i) => {
      const val = Number(d.y) || 0;
      const angle = total > 0 ? (val / total) * 360 : 0;
      const startAngle = cumulativeAngle;
      const endAngle = cumulativeAngle + angle;
      cumulativeAngle += angle;

      const startRad = ((startAngle - 90) * Math.PI) / 180;
      const endRad = ((endAngle - 90) * Math.PI) / 180;

      const x1 = centerX + radius * Math.cos(startRad);
      const y1 = centerY + radius * Math.sin(startRad);
      const x2 = centerX + radius * Math.cos(endRad);
      const y2 = centerY + radius * Math.sin(endRad);

      const ix1 = centerX + innerRadius * Math.cos(startRad);
      const iy1 = centerY + innerRadius * Math.sin(startRad);
      const ix2 = centerX + innerRadius * Math.cos(endRad);
      const iy2 = centerY + innerRadius * Math.sin(endRad);

      const largeArc = angle > 180 ? 1 : 0;
      const pathData = `
        M ${ix1} ${iy1}
        L ${x1} ${y1}
        A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}
        L ${ix2} ${iy2}
        A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${ix1} ${iy1}
        Z
      `;

      return {
        pathData,
        color: colors[i % colors.length],
        label: String(d.x),
        value: val,
        percentage: total > 0 ? ((val / total) * 100).toFixed(1) : '0',
      };
    });

    return (
      <div className="interactive-chart-card">
        <div className="chart-header">
          <div className="chart-title-area">
            <PieChart size={18} className="chart-icon text-accent" />
            <span className="chart-title">{spec.title}</span>
          </div>
          <button className="icon-btn" onClick={exportSvg} title="Export as SVG">
            <Download size={15} />
          </button>
        </div>
        <div className="chart-svg-container" style={{ position: 'relative' }}>
          <svg id={chartId} viewBox={`0 0 ${width} ${height}`} className="responsive-svg">
            {pieSlices.map((slice, i) => (
              <path
                key={i}
                d={slice.pathData}
                fill={slice.color}
                className="pie-slice"
                opacity={hoveredIndex === null || hoveredIndex === i ? 0.9 : 0.4}
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
              />
            ))}
            <text x={centerX} y={centerY - 5} textAnchor="middle" className="donut-center-label">
              Total
            </text>
            <text x={centerX} y={centerY + 16} textAnchor="middle" className="donut-center-val">
              {total.toLocaleString()}
            </text>
          </svg>
          {hoveredIndex !== null && pieSlices[hoveredIndex] && (
            <div className="chart-floating-tooltip">
              <strong>{pieSlices[hoveredIndex].label}</strong>: {pieSlices[hoveredIndex].value.toLocaleString()} ({pieSlices[hoveredIndex].percentage}%)
            </div>
          )}
        </div>
        <div className="chart-legend">
          {pieSlices.map((s, idx) => (
            <div
              key={idx}
              className={`legend-item ${hoveredIndex === idx ? 'active' : ''}`}
              onMouseEnter={() => setHoveredIndex(idx)}
              onMouseLeave={() => setHoveredIndex(null)}
            >
              <span className="legend-dot" style={{ backgroundColor: s.color }} />
              <span className="legend-name">{s.label}</span>
              <span className="legend-val">{s.percentage}%</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // 2. SCATTER PLOT RENDERING
  if (isScatter) {
    const xValues = data.map(d => Number(d.x) || 0);
    const minX = Math.min(...xValues);
    const maxX = Math.max(...xValues, 1);
    const rangeX = maxX - minX || 1;

    const getXPos = (val: number) => {
      return padding.left + ((val - minX) / rangeX) * plotWidth;
    };

    return (
      <div className="interactive-chart-card">
        <div className="chart-header">
          <div className="chart-title-area">
            <TrendingUp size={18} className="chart-icon text-accent" />
            <span className="chart-title">{spec.title}</span>
          </div>
          <button className="icon-btn" onClick={exportSvg} title="Export as SVG">
            <Download size={15} />
          </button>
        </div>
        <div className="chart-svg-container">
          <svg id={chartId} viewBox={`0 0 ${width} ${height}`} className="responsive-svg">
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => {
              const y = padding.top + plotHeight * pct;
              const val = (maxY - pct * rangeY).toFixed(0);
              return (
                <g key={idx}>
                  <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
                  <text x={padding.left - 8} y={y + 4} textAnchor="end" className="chart-axis-tick">
                    {val}
                  </text>
                </g>
              );
            })}
            {/* Scatter points */}
            {data.map((d, i) => {
              const cx = getXPos(Number(d.x) || 0);
              const cy = getYPos(Number(d.y) || 0);
              const isHovered = hoveredIndex === i;
              return (
                <circle
                  key={i}
                  cx={cx}
                  cy={cy}
                  r={isHovered ? 6 : 4}
                  fill={isHovered ? '#10b981' : '#6366f1'}
                  opacity={isHovered ? 1 : 0.75}
                  className="scatter-dot"
                  onMouseEnter={() => setHoveredIndex(i)}
                  onMouseLeave={() => setHoveredIndex(null)}
                />
              );
            })}
            {/* Axis labels */}
            {spec.x_axis && (
              <text x={width / 2} y={height - 10} textAnchor="middle" className="chart-axis-label">
                {spec.x_axis}
              </text>
            )}
            {spec.y_axis && (
              <text
                x={-height / 2}
                y={18}
                textAnchor="middle"
                transform="rotate(-90)"
                className="chart-axis-label"
              >
                {spec.y_axis}
              </text>
            )}
          </svg>
          {hoveredIndex !== null && data[hoveredIndex] && (
            <div className="chart-floating-tooltip">
              <strong>X ({spec.x_axis || 'Value'}):</strong> {data[hoveredIndex].x}<br />
              <strong>Y ({spec.y_axis || 'Value'}):</strong> {data[hoveredIndex].y}
            </div>
          )}
        </div>
      </div>
    );
  }

  // 3. BAR & LINE CHART RENDERING
  const stepX = plotWidth / Math.max(data.length, 1);
  const barWidth = Math.max(12, Math.min(stepX * 0.65, 48));

  // Compute points string for line chart
  const linePoints = data.map((d, i) => {
    const x = padding.left + i * stepX + stepX / 2;
    const y = getYPos(Number(d.y) || 0);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="interactive-chart-card">
      <div className="chart-header">
        <div className="chart-title-area">
          {isLine ? <TrendingUp size={18} className="chart-icon text-accent" /> : <BarChart2 size={18} className="chart-icon text-accent" />}
          <span className="chart-title">{spec.title}</span>
        </div>
        <button className="icon-btn" onClick={exportSvg} title="Export as SVG">
          <Download size={15} />
        </button>
      </div>

      <div className="chart-svg-container">
        <svg id={chartId} viewBox={`0 0 ${width} ${height}`} className="responsive-svg">
          <defs>
            <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#818cf8" />
              <stop offset="100%" stopColor="#4f46e5" />
            </linearGradient>
            <linearGradient id="barGradientHover" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#34d399" />
              <stop offset="100%" stopColor="#059669" />
            </linearGradient>
            <linearGradient id="lineFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6366f1" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#6366f1" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => {
            const y = padding.top + plotHeight * pct;
            const val = (maxY - pct * rangeY).toFixed(0);
            return (
              <g key={idx}>
                <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
                <text x={padding.left - 8} y={y + 4} textAnchor="end" className="chart-axis-tick">
                  {val}
                </text>
              </g>
            );
          })}

          {/* Line Chart Area and Path */}
          {isLine && (
            <>
              {/* Shaded Area */}
              <polygon
                points={`${padding.left + stepX / 2},${padding.top + plotHeight} ${linePoints} ${padding.left + (data.length - 0.5) * stepX},${padding.top + plotHeight}`}
                fill="url(#lineFill)"
              />
              {/* Stroke line */}
              <polyline
                fill="none"
                stroke="#818cf8"
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={linePoints}
              />
            </>
          )}

          {/* Bar Chart Rectangles or Line Dot Points */}
          {data.map((d, i) => {
            const xCenter = padding.left + i * stepX + stepX / 2;
            const yPos = getYPos(Number(d.y) || 0);
            const isHovered = hoveredIndex === i;
            const barH = Math.max(2, padding.top + plotHeight - yPos);

            if (isLine) {
              return (
                <g key={i}>
                  <circle
                    cx={xCenter}
                    cy={yPos}
                    r={isHovered ? 6 : 4}
                    fill={isHovered ? '#34d399' : '#6366f1'}
                    stroke="#ffffff"
                    strokeWidth="1.5"
                    onMouseEnter={() => setHoveredIndex(i)}
                    onMouseLeave={() => setHoveredIndex(null)}
                    style={{ cursor: 'pointer', transition: 'r 0.15s ease' }}
                  />
                  {/* X Axis Label */}
                  <text
                    x={xCenter}
                    y={height - padding.bottom + 18}
                    textAnchor="middle"
                    className="chart-axis-tick x-tick"
                    transform={`rotate(-25 ${xCenter} ${height - padding.bottom + 18})`}
                  >
                    {String(d.x).slice(0, 12)}
                  </text>
                </g>
              );
            }

            return (
              <g key={i}>
                <rect
                  x={xCenter - barWidth / 2}
                  y={yPos}
                  width={barWidth}
                  height={barH}
                  rx={4}
                  fill={isHovered ? 'url(#barGradientHover)' : 'url(#barGradient)'}
                  onMouseEnter={() => setHoveredIndex(i)}
                  onMouseLeave={() => setHoveredIndex(null)}
                  style={{ cursor: 'pointer', transition: 'fill 0.2s ease' }}
                />
                {/* X Axis Label */}
                <text
                  x={xCenter}
                  y={height - padding.bottom + 18}
                  textAnchor="middle"
                  className="chart-axis-tick x-tick"
                  transform={`rotate(-25 ${xCenter} ${height - padding.bottom + 18})`}
                >
                  {String(d.x).slice(0, 12)}
                </text>
              </g>
            );
          })}

          {/* Axis Labels */}
          {spec.x_axis && (
            <text x={width / 2} y={height - 5} textAnchor="middle" className="chart-axis-label">
              {spec.x_axis}
            </text>
          )}
          {spec.y_axis && (
            <text
              x={-height / 2}
              y={18}
              textAnchor="middle"
              transform="rotate(-90)"
              className="chart-axis-label"
            >
              {spec.y_axis}
            </text>
          )}
        </svg>

        {hoveredIndex !== null && data[hoveredIndex] && (
          <div className="chart-floating-tooltip">
            <strong>{String(data[hoveredIndex].x)}</strong>: {Number(data[hoveredIndex].y).toLocaleString()}
          </div>
        )}
      </div>

      {spec.description && (
        <div className="chart-caption">
          <Info size={13} className="text-muted" />
          <span>{spec.description}</span>
        </div>
      )}
    </div>
  );
};
