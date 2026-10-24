import React, { useState } from 'react';
import {
  ShieldCheck,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  Download,
  Info,
  FileSpreadsheet
} from 'lucide-react';

import type { ProfileReport } from '../types';
import { getCleanedExportUrl, getCsvReportExportUrl } from '../api';

interface Props {
  profile: ProfileReport;
}

export const DataHealthRadar: React.FC<Props> = ({ profile }) => {
  const { metadata, health, columns, correlation_matrix, top_correlations } = profile;
  const [hoveredCell, setHoveredCell] = useState<{ col1: string; col2: string; val: number } | null>(null);

  const numericColumns = columns.filter((c) => c.inferred_type === 'numeric');
  const outlierColumns = numericColumns.filter((c) => (c.outlier_count || 0) > 0);

  // Correlation matrix columns (max 8 for clean grid display)
  const matrixCols = Object.keys(correlation_matrix).slice(0, 8);

  const getCorrBg = (val: number) => {
    if (val === 1) return 'bg-[#EA580C] text-white font-bold';
    if (val >= 0.5) return 'bg-[#EA580C]/25 text-[#9A3412] font-semibold';
    if (val >= 0.2) return 'bg-[#EA580C]/10 text-[#C2410C]';
    if (val <= -0.5) return 'bg-[#0D9488]/30 text-[#0F3D3E] font-semibold';
    if (val <= -0.2) return 'bg-[#0D9488]/15 text-[#134E4A]';
    return 'bg-[#FAF8F5] text-[#648484]';
  };

  return (
    <div className="space-y-6">
      {/* Top Health Overview Hero */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Score Card */}
        <div className="bg-white p-6 rounded-3xl border border-[#EAE6DF] shadow-xs flex flex-col justify-between space-y-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-[#476D6D] uppercase tracking-wider">
              Data Quality Index
            </span>
            <span className="px-3 py-1 rounded-full text-xs font-bold bg-[#FFF7ED] text-[#EA580C] border border-[#FED7AA]">
              Grade {health.grade}
            </span>
          </div>

          <div className="flex items-center gap-5">
            <div className="w-24 h-24 rounded-full border-4 border-[#EA580C] bg-[#FFF7ED] flex flex-col items-center justify-center shrink-0 shadow-xs">
              <span className="text-3xl font-extrabold text-[#EA580C]">{health.overall_score}</span>
              <span className="text-[10px] font-semibold text-[#9A3412] tracking-wider uppercase">Score</span>
            </div>
            <div className="space-y-1">
              <h4 className="font-heading font-bold text-base text-[#0F3D3E]">Dataset Readiness</h4>
              <p className="text-xs text-[#476D6D] leading-relaxed">
                {health.overall_score >= 85
                  ? 'Pristine analytical shape. Ready for machine learning, business intelligence, and high-confidence decision making.'
                  : health.overall_score >= 70
                  ? 'Good condition. Minor imputations recommended before mission-critical modeling.'
                  : 'Requires preprocessing. Review missing values and duplicate rows below.'}
              </p>
            </div>
          </div>

          {/* 4 Pillars Progress Bars */}
          <div className="grid grid-cols-2 gap-3 pt-2 border-t border-[#F0EDE6]">
            <div>
              <div className="flex justify-between text-xs text-[#476D6D] mb-1">
                <span>Completeness</span>
                <strong className="text-[#0F3D3E]">{health.completeness_score}%</strong>
              </div>
              <div className="h-1.5 w-full bg-[#FAF8F5] rounded-full overflow-hidden border border-[#EAE6DF]">
                <div
                  className="h-full bg-[#EA580C] rounded-full"
                  style={{ width: `${health.completeness_score}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs text-[#476D6D] mb-1">
                <span>Uniqueness</span>
                <strong className="text-[#0F3D3E]">{health.uniqueness_score}%</strong>
              </div>
              <div className="h-1.5 w-full bg-[#FAF8F5] rounded-full overflow-hidden border border-[#EAE6DF]">
                <div
                  className="h-full bg-[#0D9488] rounded-full"
                  style={{ width: `${health.uniqueness_score}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs text-[#476D6D] mb-1">
                <span>Outlier Health</span>
                <strong className="text-[#0F3D3E]">{health.outlier_score}%</strong>
              </div>
              <div className="h-1.5 w-full bg-[#FAF8F5] rounded-full overflow-hidden border border-[#EAE6DF]">
                <div
                  className="h-full bg-[#F59E0B] rounded-full"
                  style={{ width: `${health.outlier_score}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs text-[#476D6D] mb-1">
                <span>Consistency</span>
                <strong className="text-[#0F3D3E]">{health.consistency_score}%</strong>
              </div>
              <div className="h-1.5 w-full bg-[#FAF8F5] rounded-full overflow-hidden border border-[#EAE6DF]">
                <div
                  className="h-full bg-[#6366F1] rounded-full"
                  style={{ width: `${health.consistency_score}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Quality Badges & Automated Remediation */}
        <div className="bg-white p-6 rounded-3xl border border-[#EAE6DF] shadow-xs flex flex-col justify-between space-y-4 lg:col-span-2">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <ShieldCheck className="w-5 h-5 text-[#0D9488]" />
              <h3 className="font-heading font-bold text-lg text-[#0F3D3E]">
                Automated Quality Diagnostics
              </h3>
            </div>

            {/* Badges */}
            <div className="flex flex-wrap gap-2 mb-4">
              {health.badges.map((b, i) => (
                <span
                  key={i}
                  className="text-xs bg-[#F0FDFA] text-[#0F3D3E] border border-[#CCECE6] px-3 py-1 rounded-full font-medium flex items-center gap-1.5"
                >
                  <CheckCircle className="w-3.5 h-3.5 text-[#0D9488]" />
                  <span>{b}</span>
                </span>
              ))}
            </div>

            {/* Recommendations */}
            <div className="space-y-2">
              <span className="text-xs font-semibold text-[#476D6D] uppercase tracking-wider block">
                Recommended Actions:
              </span>
              <div className="space-y-1.5">
                {health.recommendations.map((rec, i) => (
                  <div
                    key={i}
                    className="p-2.5 rounded-xl bg-[#FAF8F5] border border-[#EAE6DF] text-xs text-[#0F3D3E] flex items-start gap-2"
                  >
                    <Info className="w-4 h-4 text-[#EA580C] shrink-0 mt-0.5" />
                    <span>{rec}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Action Export Buttons */}
          <div className="flex flex-wrap gap-3 pt-3 border-t border-[#F0EDE6]">
            <a
              href={getCsvReportExportUrl(metadata.id)}
              download
              className="inline-flex items-center gap-2 bg-[#EA580C] hover:bg-[#C2410C] text-white px-4 py-2 rounded-full text-xs font-medium transition-all shadow-xs"
            >
              <FileSpreadsheet className="w-4 h-4" />
              <span>Download Full CSV Audit</span>
            </a>
            <a
              href={getCleanedExportUrl(metadata.id)}
              download
              className="inline-flex items-center gap-2 bg-white hover:bg-[#FAF8F5] text-[#0F3D3E] border border-[#EAE6DF] px-4 py-2 rounded-full text-xs font-medium transition-all"
            >
              <Download className="w-4 h-4 text-[#0D9488]" />
              <span>Download Cleaned CSV</span>
            </a>
          </div>
        </div>
      </div>

      {/* Outlier Radar Section */}
      <div className="bg-white p-6 rounded-3xl border border-[#EAE6DF] shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-[#F59E0B]" />
            <h3 className="font-heading font-bold text-lg text-[#0F3D3E]">
              Statistical Outlier Radar (1.5x IQR Rule)
            </h3>
          </div>
          <span className="text-xs text-[#648484] bg-[#FAF8F5] px-3 py-1 rounded-full border border-[#EAE6DF]">
            Tukey Method
          </span>
        </div>

        {outlierColumns.length === 0 ? (
          <div className="p-4 bg-[#F0FDFA] rounded-2xl border border-[#CCECE6] text-xs text-[#0F3D3E] flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-[#0D9488]" />
            <span>Zero extreme outliers detected across all numeric features! Distributions are well-behaved.</span>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {outlierColumns.map((col, idx) => (
              <div key={idx} className="p-4 rounded-2xl bg-[#FAF8F5] border border-[#EAE6DF] space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-xs text-[#0F3D3E]">{col.column}</span>
                  <span className="text-[11px] bg-[#FEF3C7] text-[#B45309] font-bold px-2 py-0.5 rounded-full">
                    {col.outlier_count} Outliers ({col.outlier_percentage}%)
                  </span>
                </div>
                <div className="text-[11px] text-[#476D6D] grid grid-cols-2 gap-1 pt-1 border-t border-[#EAE6DF]">
                  <div>IQR: <strong className="text-[#0F3D3E]">{col.iqr}</strong></div>
                  <div>Median: <strong className="text-[#0F3D3E]">{col.median}</strong></div>
                  <div>Q1 (25%): <strong className="text-[#0F3D3E]">{col.q25}</strong></div>
                  <div>Q3 (75%): <strong className="text-[#0F3D3E]">{col.q75}</strong></div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Correlation Matrix & Top Pairs */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Heatmap Grid */}
        <div className="bg-white p-6 rounded-3xl border border-[#EAE6DF] shadow-xs space-y-4 lg:col-span-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-[#EA580C]" />
              <h3 className="font-heading font-bold text-lg text-[#0F3D3E]">
                Pearson Correlation Heatmap
              </h3>
            </div>
            <span className="text-xs text-[#648484]">Numeric Variables</span>
          </div>

          {matrixCols.length < 2 ? (
            <div className="p-4 text-xs text-[#648484] bg-[#FAF8F5] rounded-xl border border-[#EAE6DF]">
              Dataset requires at least 2 distinct numeric columns to calculate correlation matrix.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="border-collapse text-xs">
                <thead>
                  <tr>
                    <th className="p-2 text-left text-[#648484] font-medium">Feature</th>
                    {matrixCols.map((c, i) => (
                      <th key={i} className="p-2 text-center text-[#476D6D] font-medium max-w-[80px] truncate" title={c}>
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {matrixCols.map((r, rIdx) => (
                    <tr key={rIdx}>
                      <td className="p-2 text-[#476D6D] font-medium max-w-[100px] truncate border-t border-[#F0EDE6]" title={r}>
                        {r}
                      </td>
                      {matrixCols.map((c, cIdx) => {
                        const val = correlation_matrix[r]?.[c] ?? 0;
                        return (
                          <td
                            key={cIdx}
                            onMouseEnter={() => setHoveredCell({ col1: r, col2: c, val })}
                            onMouseLeave={() => setHoveredCell(null)}
                            className={`p-2 text-center border border-[#F0EDE6] transition-transform hover:scale-110 cursor-pointer ${getCorrBg(val)}`}
                          >
                            {val}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>

              {hoveredCell && (
                <div className="mt-3 p-2 bg-[#FAF8F5] rounded-xl border border-[#EAE6DF] text-xs text-[#0F3D3E] flex items-center justify-between">
                  <span><strong>{hoveredCell.col1}</strong> ↔ <strong>{hoveredCell.col2}</strong></span>
                  <span className="font-bold text-[#EA580C]">r = {hoveredCell.val}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Top Correlated Pairs */}
        <div className="bg-white p-6 rounded-3xl border border-[#EAE6DF] shadow-xs space-y-4">
          <h3 className="font-heading font-bold text-base text-[#0F3D3E]">
            Top Correlated Relationships
          </h3>

          {top_correlations.length === 0 ? (
            <div className="text-xs text-[#648484] p-3 bg-[#FAF8F5] rounded-xl border border-[#EAE6DF]">
              No strong linear correlations (|r| ≥ 0.15) detected.
            </div>
          ) : (
            <div className="space-y-2">
              {top_correlations.slice(0, 6).map((pair, idx) => {
                const absR = Math.abs(pair.correlation);
                const strength = absR >= 0.7 ? 'Strong' : absR >= 0.35 ? 'Moderate' : 'Weak';
                const isPos = pair.correlation > 0;
                return (
                  <div key={idx} className="p-3 rounded-2xl bg-[#FAF8F5] border border-[#EAE6DF] space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-[#0F3D3E]">{pair.col1} & {pair.col2}</span>
                      <span className={`font-bold ${isPos ? 'text-[#EA580C]' : 'text-[#0D9488]'}`}>
                        {isPos ? '+' : ''}{pair.correlation}
                      </span>
                    </div>
                    <div className="text-[11px] text-[#648484]">
                      {strength} {isPos ? 'Positive' : 'Negative'} Relationship
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
