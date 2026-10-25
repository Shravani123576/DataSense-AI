import React, { useState, useEffect } from 'react';
import {
  Table as TableIcon,
  Search,
  ChevronLeft,
  ChevronRight,
  BookOpen,
  Hash,
  Type,
  Calendar,
  ToggleLeft
} from 'lucide-react';
import type { ProfileReport, DatasetPreviewData, ColumnStat } from '../types';
import { fetchDatasetPreview } from '../api';

interface Props {
  profile: ProfileReport;
}

export const DataGridPreview: React.FC<Props> = ({ profile }) => {
  const { metadata, columns } = profile;
  const [page, setPage] = useState(1);
  const pageSize = 15;

  const [search, setSearch] = useState('');
  const [previewData, setPreviewData] = useState<DatasetPreviewData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'grid' | 'dictionary'>('grid');
  const [selectedColumn, setSelectedColumn] = useState<ColumnStat | null>(columns[0] || null);

  useEffect(() => {
    let isCurrent = true;
    const loadPreview = async () => {
      setIsLoading(true);
      try {
        const data = await fetchDatasetPreview(metadata.id, page, pageSize, search || undefined);
        if (isCurrent) setPreviewData(data);
      } catch (e) {
        console.error('Error fetching preview:', e);
      } finally {
        if (isCurrent) setIsLoading(false);
      }
    };

    loadPreview();
    return () => {
      isCurrent = false;
    };
  }, [metadata.id, page, pageSize, search]);

  const totalPages = previewData ? Math.ceil(previewData.total_rows / pageSize) : 1;

  const getTypeBadge = (type: string) => {
    switch (type) {
      case 'numeric':
        return (
          <span className="inline-flex items-center gap-1 text-xs bg-[#FFF7ED] text-[#EA580C] px-2.5 py-0.5 rounded-full font-semibold border border-[#FED7AA]">
            <Hash className="w-3 h-3" />
            numeric
          </span>
        );
      case 'categorical':
      case 'text':
        return (
          <span className="inline-flex items-center gap-1 text-xs bg-[#F0FDFA] text-[#0D9488] px-2.5 py-0.5 rounded-full font-semibold border border-[#CCECE6]">
            <Type className="w-3 h-3" />
            {type}
          </span>
        );
      case 'datetime':
        return (
          <span className="inline-flex items-center gap-1 text-xs bg-[#EEF2FF] text-[#6366F1] px-2.5 py-0.5 rounded-full font-semibold border border-[#C7D2FE]">
            <Calendar className="w-3 h-3" />
            datetime
          </span>
        );
      case 'boolean':
        return (
          <span className="inline-flex items-center gap-1 text-xs bg-[#FEF3C7] text-[#D97706] px-2.5 py-0.5 rounded-full font-semibold border border-[#FDE68A]">
            <ToggleLeft className="w-3 h-3" />
            boolean
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 text-xs bg-[#FAF8F5] text-[#476D6D] px-2.5 py-0.5 rounded-full font-semibold border border-[#EAE6DF]">
            {type}
          </span>
        );
    }
  };

  return (
    <div className="space-y-5">
      {/* Subnav Pills */}
      <div className="flex items-center gap-2 bg-[#F5F2EB] p-1.5 rounded-2xl w-fit border border-[#EAE6DF]">
        <button
          onClick={() => setActiveTab('grid')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all cursor-pointer ${
            activeTab === 'grid'
              ? 'bg-white text-[#0F3D3E] shadow-xs font-bold'
              : 'text-[#476D6D] hover:text-[#0F3D3E]'
          }`}
        >
          <TableIcon className="w-4 h-4 text-[#EA580C]" />
          <span>Data Table</span>
        </button>
        <button
          onClick={() => setActiveTab('dictionary')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all cursor-pointer ${
            activeTab === 'dictionary'
              ? 'bg-white text-[#0F3D3E] shadow-xs font-bold'
              : 'text-[#476D6D] hover:text-[#0F3D3E]'
          }`}
        >
          <BookOpen className="w-4 h-4 text-[#0D9488]" />
          <span>Feature Profiles</span>
        </button>
      </div>

      {activeTab === 'grid' && (
        <div className="bg-white rounded-3xl border border-[#EAE6DF] shadow-xs overflow-hidden">
          {/* Table Toolbar */}
          <div className="p-4 border-b border-[#F0EDE6] flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="relative w-full sm:w-80">
              <Search className="w-4 h-4 text-[#648484] absolute left-3.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search across rows..."
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value);
                  setPage(1);
                }}
                className="w-full bg-[#FAF8F5] border border-[#EAE6DF] rounded-full pl-10 pr-4 py-2 text-sm text-[#0F3D3E] focus:outline-none focus:border-[#EA580C]"
              />
            </div>

            <div className="flex items-center gap-3 w-full sm:w-auto justify-between sm:justify-end text-sm text-[#476D6D]">
              <span>
                Showing <strong>{previewData?.total_rows.toLocaleString() || metadata.rows.toLocaleString()}</strong> rows ({metadata.columns} columns)
              </span>

              <div className="flex items-center gap-1">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1 || isLoading}
                  className="p-2 rounded-xl border border-[#EAE6DF] bg-[#FAF8F5] hover:bg-white disabled:opacity-40 cursor-pointer"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <span className="px-2.5 font-medium">
                  {page} / {totalPages}
                </span>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages || isLoading}
                  className="p-2 rounded-xl border border-[#EAE6DF] bg-[#FAF8F5] hover:bg-white disabled:opacity-40 cursor-pointer"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Table Container */}
          <div className="overflow-x-auto max-h-[520px]">
            <table className="w-full text-left text-sm border-collapse">
              <thead className="sticky top-0 bg-[#FAF8F5] z-10 border-b border-[#EAE6DF]">
                <tr>
                  <th className="py-3 px-3.5 text-[#648484] font-semibold">#</th>
                  {(previewData?.columns || metadata.column_names).map((col, idx) => {
                    const cStat = columns.find((c) => c.column === col);
                    return (
                      <th key={idx} className="py-2.5 px-3 whitespace-nowrap">
                        <div className="flex items-center gap-1.5">
                          <span className="font-semibold text-[#0F3D3E]">{col}</span>
                          {cStat && getTypeBadge(cStat.inferred_type)}
                        </div>
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody className="divide-y divide-[#F0EDE6]">
                {isLoading ? (
                  <tr>
                    <td colSpan={metadata.columns + 1} className="py-12 text-center text-xs text-[#648484]">
                      Loading real records from Pandas runtime...
                    </td>
                  </tr>
                ) : !previewData?.rows || previewData.rows.length === 0 ? (
                  <tr>
                    <td colSpan={metadata.columns + 1} className="py-12 text-center text-xs text-[#648484]">
                      No matching rows found for query "{search}".
                    </td>
                  </tr>
                ) : (
                  previewData.rows.map((row, rIdx) => (
                    <tr key={rIdx} className="hover:bg-[#FFFBF5] transition-colors">
                      <td className="py-2.5 px-3.5 text-[#648484] font-mono text-xs">
                        {(page - 1) * pageSize + rIdx + 1}
                      </td>
                      {previewData.columns.map((col, cIdx) => (
                        <td key={cIdx} className="py-2.5 px-3.5 text-sm text-[#0F3D3E] whitespace-nowrap max-w-[240px] truncate">
                          {row[col] !== null ? String(row[col]) : <span className="text-[#648484]/50 italic">null</span>}
                        </td>
                      ))}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Feature Dictionary Tab */}
      {activeTab === 'dictionary' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Columns Selector Sidebar */}
          <div className="bg-white p-4 rounded-3xl border border-[#EAE6DF] shadow-xs space-y-2 max-h-[580px] overflow-y-auto">
            <span className="text-sm font-bold text-[#476D6D] uppercase tracking-wider block px-2 mb-2">
              All Columns ({columns.length})
            </span>
            {columns.map((c) => {
              const isSelected = selectedColumn?.column === c.column;
              return (
                <button
                  key={c.column}
                  onClick={() => setSelectedColumn(c)}
                  className={`w-full text-left p-3.5 rounded-2xl transition-all flex items-center justify-between cursor-pointer ${
                    isSelected
                      ? 'bg-[#FFF7ED] border border-[#FED7AA] shadow-2xs'
                      : 'hover:bg-[#FAF8F5] border border-transparent'
                  }`}
                >
                  <div className="space-y-0.5">
                    <div className="font-semibold text-sm text-[#0F3D3E]">{c.column}</div>
                    <div className="text-xs text-[#648484]">{c.unique_count} distinct values</div>
                  </div>
                  <div>{getTypeBadge(c.inferred_type)}</div>
                </button>
              );
            })}
          </div>

          {/* Column Detailed Statistical Breakdown */}
          {selectedColumn && (
            <div className="bg-white p-6 rounded-3xl border border-[#EAE6DF] shadow-xs md:col-span-2 space-y-5">
              <div className="flex items-center justify-between pb-3 border-b border-[#F0EDE6]">
                <div>
                  <h3 className="font-heading font-bold text-xl text-[#0F3D3E]">
                    {selectedColumn.column}
                  </h3>
                  <p className="text-sm text-[#648484]">
                    Pandas dtype: <code className="font-mono text-xs bg-[#FAF8F5] px-1.5 py-0.5 rounded border border-[#EAE6DF]">{selectedColumn.dtype}</code>
                  </p>
                </div>
                <div>{getTypeBadge(selectedColumn.inferred_type)}</div>
              </div>

              {/* Quality Indicators */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                <div className="p-3.5 bg-[#FAF8F5] rounded-2xl border border-[#EAE6DF]">
                  <div className="text-xs md:text-sm text-[#648484]">Unique Values</div>
                  <div className="text-xl font-bold text-[#0F3D3E]">{selectedColumn.unique_count.toLocaleString()}</div>
                  <div className="text-xs text-[#476D6D]">{selectedColumn.unique_percentage}% unique</div>
                </div>

                <div className="p-3.5 bg-[#FAF8F5] rounded-2xl border border-[#EAE6DF]">
                  <div className="text-xs md:text-sm text-[#648484]">Missing Cells</div>
                  <div className="text-xl font-bold text-[#0F3D3E]">{selectedColumn.null_count}</div>
                  <div className="text-xs text-[#476D6D]">{selectedColumn.null_percentage}% null</div>
                </div>

                {selectedColumn.inferred_type === 'numeric' && (
                  <>
                    <div className="p-3.5 bg-[#FAF8F5] rounded-2xl border border-[#EAE6DF]">
                      <div className="text-xs md:text-sm text-[#648484]">Mean (Avg)</div>
                      <div className="text-xl font-bold text-[#EA580C]">{selectedColumn.mean ?? '-'}</div>
                      <div className="text-xs text-[#476D6D]">Median: {selectedColumn.median ?? '-'}</div>
                    </div>

                    <div className="p-3.5 bg-[#FAF8F5] rounded-2xl border border-[#EAE6DF]">
                      <div className="text-xs md:text-sm text-[#648484]">Outliers (IQR)</div>
                      <div className="text-xl font-bold text-[#B45309]">{selectedColumn.outlier_count ?? 0}</div>
                      <div className="text-xs text-[#476D6D]">{selectedColumn.outlier_percentage ?? 0}% rate</div>
                    </div>
                  </>
                )}
              </div>

              {/* Numeric Statistical Percentiles */}
              {selectedColumn.inferred_type === 'numeric' && (
                <div className="space-y-2.5">
                  <span className="text-sm font-bold text-[#476D6D] uppercase tracking-wider block">
                    Distribution & Percentile Markers
                  </span>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-sm">
                    <div className="p-3 bg-[#FAF8F5] rounded-xl border border-[#EAE6DF]">
                      <span className="text-xs text-[#648484] block">Min:</span>
                      <strong className="text-sm font-bold text-[#0F3D3E]">{selectedColumn.min}</strong>
                    </div>
                    <div className="p-3 bg-[#FAF8F5] rounded-xl border border-[#EAE6DF]">
                      <span className="text-xs text-[#648484] block">25th Percentile (Q1):</span>
                      <strong className="text-sm font-bold text-[#0F3D3E]">{selectedColumn.q25}</strong>
                    </div>
                    <div className="p-3 bg-[#FAF8F5] rounded-xl border border-[#EAE6DF]">
                      <span className="text-xs text-[#648484] block">75th Percentile (Q3):</span>
                      <strong className="text-sm font-bold text-[#0F3D3E]">{selectedColumn.q75}</strong>
                    </div>
                    <div className="p-3 bg-[#FAF8F5] rounded-xl border border-[#EAE6DF]">
                      <span className="text-xs text-[#648484] block">Max:</span>
                      <strong className="text-sm font-bold text-[#0F3D3E]">{selectedColumn.max}</strong>
                    </div>
                  </div>
                </div>
              )}

              {/* Categorical Frequency Breakdown */}
              {selectedColumn.top_categories && selectedColumn.top_categories.length > 0 && (
                <div className="space-y-2.5">
                  <span className="text-sm font-bold text-[#476D6D] uppercase tracking-wider block">
                    Top Occurring Categories
                  </span>
                  <div className="space-y-2">
                    {selectedColumn.top_categories.map((cat, cIdx) => (
                      <div key={cIdx} className="space-y-1">
                        <div className="flex justify-between text-sm text-[#0F3D3E]">
                          <span className="font-semibold">{cat.value}</span>
                          <span className="text-sm font-medium text-[#476D6D]">{cat.count} ({cat.percentage}%)</span>
                        </div>
                        <div className="h-2 w-full bg-[#FAF8F5] rounded-full overflow-hidden border border-[#EAE6DF]">
                          <div
                            className="h-full bg-[#0D9488] rounded-full"
                            style={{ width: `${cat.percentage}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Sample Clean Values */}
              <div className="space-y-2.5 pt-2 border-t border-[#F0EDE6]">
                <span className="text-sm font-bold text-[#476D6D] uppercase tracking-wider block">
                  Sample Data Values
                </span>
                <div className="flex flex-wrap gap-2">
                  {selectedColumn.sample_values.map((v, sIdx) => (
                    <span
                      key={sIdx}
                      className="px-3 py-1.5 bg-[#FAF8F5] text-[#0F3D3E] rounded-lg text-sm font-mono border border-[#EAE6DF]"
                    >
                      {v !== null ? String(v) : 'null'}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
