import React, { useState, useRef, useEffect } from 'react';
import {
  Sparkles,
  Upload,
  Download,
  Settings,
  Database,
  ChevronDown,
  FileSpreadsheet,
  CheckCircle2,
  FileText,
  Filter
} from 'lucide-react';
import type { ProfileReport, SampleDatasetInfo } from '../types';
import { getSummaryExportUrl, getCleanedExportUrl, getCsvReportExportUrl } from '../api';

interface Props {
  profile: ProfileReport | null;
  samples: SampleDatasetInfo[];
  selectedSampleId: string;
  onSelectSample: (id: string) => void;
  onOpenUpload: () => void;
  onOpenSettings: () => void;
  isLoading: boolean;
}

export const Header: React.FC<Props> = ({
  profile,
  samples,
  selectedSampleId,
  onSelectSample,
  onOpenUpload,
  onOpenSettings,
  isLoading,
}) => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);

  const dropdownRef = useRef<HTMLDivElement>(null);
  const exportRef = useRef<HTMLDivElement>(null);

  const datasetId = profile?.metadata.id || '';

  // Close dropdowns on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
      if (exportRef.current && !exportRef.current.contains(event.target as Node)) {
        setExportOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <header className="sticky top-0 z-30 bg-[#FAF8F5]/90 backdrop-blur-md border-b border-[#EAE6DF] px-4 md:px-8 py-3.5 flex items-center justify-between">
      {/* Brand Logo & Name */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-[#EA580C] to-[#C2410C] flex items-center justify-center text-white shadow-xs">
          <Sparkles className="w-5 h-5 fill-current/20" />
        </div>
        <span className="font-heading font-extrabold text-2xl text-[#0F3D3E] tracking-tight">
          DataSense <span className="text-[#EA580C]">AI</span>
        </span>
      </div>

      {/* Center / Right Action Controls */}
      <div className="flex items-center gap-3">
        {/* Sample Datasets Selector */}
        <div className="relative" ref={dropdownRef}>
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            disabled={isLoading}
            id="sample-dataset-dropdown-btn"
            className="flex items-center gap-2.5 bg-white hover:bg-[#F5F2EB] text-[#0F3D3E] px-4 py-2 rounded-full border border-[#EAE6DF] text-sm md:text-base font-semibold transition-all shadow-2xs cursor-pointer disabled:opacity-50"
          >
            <Database className="w-4 h-4 text-[#EA580C]" />
            <span className="max-w-[150px] md:max-w-[240px] truncate text-left">
              {profile ? profile.metadata.name : 'Select Dataset'}
            </span>
            <ChevronDown className="w-4 h-4 text-[#648484]" />
          </button>

          {dropdownOpen && (
            <div className="absolute left-0 sm:left-auto sm:right-0 mt-2 w-80 md:w-96 bg-white rounded-2xl border border-[#EAE6DF] shadow-lg py-2 z-50 animate-in fade-in slide-in-from-top-2 duration-150 max-h-[460px] overflow-y-auto">
              <div className="px-4 py-2 border-b border-[#F0EDE6] text-xs font-semibold text-[#476D6D] uppercase tracking-wider">
                Select Built-in Dataset
              </div>
              <div className="divide-y divide-[#F5F2EB]">
                {samples.map((s) => {
                  const isSelected = selectedSampleId === s.id;
                  return (
                    <button
                      key={s.id}
                      onClick={() => {
                        onSelectSample(s.id);
                        setDropdownOpen(false);
                      }}
                      className={`w-full text-left p-3.5 transition-colors flex items-start justify-between gap-2 hover:bg-[#FFFBF5] cursor-pointer ${
                        isSelected ? 'bg-[#FFF7ED]/60' : ''
                      }`}
                    >
                      <div className="space-y-1">
                        <div className="font-semibold text-xs md:text-sm text-[#0F3D3E] flex items-center gap-1.5">
                          <span>{s.name}</span>
                          {s.id === 'zomato_restaurants' && (
                            <span className="text-[10px] bg-[#FEF3C7] text-[#D97706] px-1.5 py-0.2 rounded-md font-bold">
                              Culinary
                            </span>
                          )}
                          {s.id === 'bigdata_100k' && (
                            <span className="text-[10px] bg-[#EEF2FF] text-[#4F46E5] px-1.5 py-0.2 rounded-md font-bold">
                              100k+
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-[#648484] line-clamp-2">{s.description}</p>
                        <div className="flex flex-wrap gap-1.5 pt-1">
                          <span className="text-[10px] bg-[#FAF8F5] text-[#2D5555] px-2 py-0.5 rounded-full border border-[#EAE6DF]">
                            {s.category}
                          </span>
                          <span className="text-[10px] bg-[#FAF8F5] text-[#2D5555] px-2 py-0.5 rounded-full border border-[#EAE6DF]">
                            {s.rows.toLocaleString()} rows
                          </span>
                          <span className="text-[10px] bg-[#FAF8F5] text-[#2D5555] px-2 py-0.5 rounded-full border border-[#EAE6DF]">
                            {s.columns} cols
                          </span>
                        </div>
                      </div>
                      {isSelected && (
                        <CheckCircle2 className="w-4 h-4 text-[#EA580C] shrink-0 mt-0.5" />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Upload Custom Dataset */}
        <button
          onClick={onOpenUpload}
          className="flex items-center gap-2 bg-[#EA580C] hover:bg-[#C2410C] text-white px-4 py-2 rounded-full text-sm md:text-base font-semibold transition-all shadow-xs cursor-pointer"
        >
          <Upload className="w-4 h-4" />
          <span className="hidden sm:inline">Upload</span>
        </button>

        {/* Export Dropdown */}
        <div className="relative" ref={exportRef}>
          <button
            onClick={() => setExportOpen(!exportOpen)}
            disabled={!profile}
            className="flex items-center gap-2 bg-white hover:bg-[#F5F2EB] text-[#0F3D3E] px-4 py-2 rounded-full border border-[#EAE6DF] text-sm md:text-base font-semibold transition-all shadow-2xs cursor-pointer disabled:opacity-50"
          >
            <Download className="w-4 h-4 text-[#476D6D]" />
            <span className="hidden md:inline">Export</span>
            <ChevronDown className="w-4 h-4 text-[#648484]" />
          </button>

          {exportOpen && datasetId && (
            <div className="absolute right-0 mt-2 w-64 bg-white rounded-2xl border border-[#EAE6DF] shadow-lg py-1.5 z-50 animate-in fade-in duration-150">
              <div className="px-3 py-1.5 text-[11px] font-semibold text-[#648484] uppercase tracking-wider border-b border-[#F0EDE6]">
                Export Data & Reports
              </div>
              <a
                href={getCsvReportExportUrl(datasetId)}
                download
                onClick={() => setExportOpen(false)}
                className="flex items-center gap-2.5 px-3.5 py-2.5 text-xs text-[#0F3D3E] hover:bg-[#FFF7ED] hover:text-[#EA580C] transition-colors"
              >
                <FileSpreadsheet className="w-4 h-4 text-[#EA580C]" />
                <div>
                  <div className="font-medium">DataSense Report (CSV)</div>
                  <div className="text-[11px] text-[#648484]">Full stats, health, and metadata</div>
                </div>
              </a>
              <a
                href={getCleanedExportUrl(datasetId)}
                download
                onClick={() => setExportOpen(false)}
                className="flex items-center gap-2.5 px-3.5 py-2.5 text-xs text-[#0F3D3E] hover:bg-[#FFF7ED] hover:text-[#EA580C] transition-colors"
              >
                <Filter className="w-4 h-4 text-[#0D9488]" />
                <div>
                  <div className="font-medium">Cleaned Dataset (CSV)</div>
                  <div className="text-[11px] text-[#648484]">Deduplicated & imputed nulls</div>
                </div>
              </a>
              <a
                href={getSummaryExportUrl(datasetId)}
                download
                onClick={() => setExportOpen(false)}
                className="flex items-center gap-2.5 px-3.5 py-2.5 text-xs text-[#0F3D3E] hover:bg-[#FFF7ED] hover:text-[#EA580C] transition-colors"
              >
                <FileText className="w-4 h-4 text-[#6366F1]" />
                <div>
                  <div className="font-medium">Executive Audit (Markdown)</div>
                  <div className="text-[11px] text-[#648484]">Formatted documentation report</div>
                </div>
              </a>
            </div>
          )}
        </div>

        {/* Settings Button */}
        <button
          onClick={onOpenSettings}
          className="p-2 rounded-full bg-white hover:bg-[#F5F2EB] text-[#476D6D] border border-[#EAE6DF] transition-all cursor-pointer"
          title="Configure LLM API Key"
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
};
