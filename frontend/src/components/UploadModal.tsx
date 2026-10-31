import React, { useState, useRef } from 'react';
import {
  Upload,
  X,
  FileSpreadsheet,
  AlertCircle,
  Database,
  ArrowRight,
  Layers,
  Link2,
  CheckCircle2,
  AlertTriangle,
  Table as TableIcon
} from 'lucide-react';

import {
  uploadDatasetFile,
  inspectExcelWorksheets,
  uploadMultiDatasetFiles,
  fetchTableRelationships,
  previewTableJoin,
  applyTableJoin
} from '../api';
import type {
  ProfileReport,
  SampleDatasetInfo,
  WorkbookInspectResponse,
  MultiFileUploadResponse,
  RelationshipDetectionResponse,
  JoinPreviewResponse
} from '../types';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (report: ProfileReport) => void;
  samples: SampleDatasetInfo[];
  onSelectSample: (id: string) => void;
}

export const UploadModal: React.FC<Props> = ({
  isOpen,
  onClose,
  onSuccess,
  samples,
  onSelectSample,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedSheet, setSelectedSheet] = useState<string | null>(null);
  const [workbookInfo, setWorkbookInfo] = useState<WorkbookInspectResponse | null>(null);
  const [multiInventory, setMultiInventory] = useState<MultiFileUploadResponse | null>(null);
  const [relationships, setRelationships] = useState<RelationshipDetectionResponse | null>(null);
  const [joinPreview, setJoinPreview] = useState<JoinPreviewResponse | null>(null);
  const [selectedJoinTableA, setSelectedJoinTableA] = useState<string>('');
  const [selectedJoinTableB, setSelectedJoinTableB] = useState<string>('');
  const [selectedColA, setSelectedColA] = useState<string>('');
  const [selectedColB, setSelectedColB] = useState<string>('');
  const [selectedJoinType, setSelectedJoinType] = useState<string>('inner');

  const [isUploading, setIsUploading] = useState(false);
  const [isDetectingRels, setIsDetectingRels] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFiles(Array.from(e.dataTransfer.files));
    }
  };

  const validateAndSetFiles = async (files: File[]) => {
    const validExtensions = ['.csv', '.xlsx', '.xls', '.zip', '.txt', '.tsv'];
    const invalidFile = files.find(
      (f) => !validExtensions.some((ext) => f.name.toLowerCase().endsWith(ext))
    );

    if (invalidFile) {
      setErrorMsg(`Unsupported file '${invalidFile.name}'. Please upload CSV (.csv), Excel (.xlsx, .xls), or ZIP (.zip) files.`);
      setSelectedFiles([]);
      return;
    }

    const totalSize = files.reduce((acc, f) => acc + f.size, 0);
    if (totalSize > 100 * 1024 * 1024) {
      setErrorMsg('Total upload size exceeds 100 MB limit.');
      setSelectedFiles([]);
      return;
    }

    setErrorMsg(null);
    setSelectedFiles(files);
    setWorkbookInfo(null);
    setMultiInventory(null);
    setRelationships(null);
    setJoinPreview(null);
    setSelectedSheet(null);

    // If single Excel file, inspect worksheets to see if multiple sheets exist
    const isSingleExcel =
      files.length === 1 &&
      (files[0].name.toLowerCase().endsWith('.xlsx') || files[0].name.toLowerCase().endsWith('.xls'));
    if (isSingleExcel) {
      try {
        const inspect = await inspectExcelWorksheets(files[0]);
        if (inspect.total_sheets > 1) {
          setWorkbookInfo(inspect);
          const firstNonEmpty = inspect.sheets.find((s) => !s.is_empty);
          setSelectedSheet(firstNonEmpty ? firstNonEmpty.sheet_name : inspect.sheets[0].sheet_name);
        }
      } catch (err: any) {
        // Non-critical: Proceed with normal upload if inspection fails
        console.warn('Workbook sheet inspection warning:', err);
      }
    }
  };

  const handleUploadSubmit = async () => {
    if (selectedFiles.length === 0) return;
    setIsUploading(true);
    setErrorMsg(null);

    try {
      // Case A: Multiple files or ZIP uploaded -> Inventory pipeline
      if (selectedFiles.length > 1 || selectedFiles[0].name.toLowerCase().endsWith('.zip')) {
        const multiResp = await uploadMultiDatasetFiles(selectedFiles);
        setMultiInventory(multiResp);
        if (multiResp.tables.length >= 2) {
          // Pre-populate join dropdowns
          setSelectedJoinTableA(multiResp.tables[0].table_id);
          setSelectedJoinTableB(multiResp.tables[1].table_id);
          if (multiResp.tables[0].column_names.length > 0) {
            setSelectedColA(multiResp.tables[0].column_names[0]);
          }
          if (multiResp.tables[1].column_names.length > 0) {
            setSelectedColB(multiResp.tables[1].column_names[0]);
          }
        }
      } else {
        // Case B: Single CSV or Excel file -> Direct instant profiling
        const report = await uploadDatasetFile(selectedFiles[0], selectedSheet || undefined);
        onSuccess(report);
        onClose();
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to upload and analyze dataset.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleSelectStandaloneTable = async (tableId: string) => {
    setIsUploading(true);
    setErrorMsg(null);
    try {
      // Re-fetch profile for the selected table
      const res = await fetch(`/api/dataset/${encodeURIComponent(tableId)}/profile`);
      if (!res.ok) throw new Error('Failed to load table profile.');
      const report: ProfileReport = await res.json();
      onSuccess(report);
      onClose();
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to select table.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDetectRelationships = async () => {
    if (!multiInventory) return;
    setIsDetectingRels(true);
    setErrorMsg(null);
    try {
      const rels = await fetchTableRelationships(multiInventory.session_id);
      setRelationships(rels);

      // Pre-select top candidate key if discovered
      if (rels.relationships.length > 0) {
        const firstRel = rels.relationships[0];
        setSelectedJoinTableA(firstRel.table_a_id);
        setSelectedJoinTableB(firstRel.table_b_id);
        if (firstRel.candidate_keys.length > 0) {
          const topKey = firstRel.candidate_keys[0];
          setSelectedColA(topKey.table_a_column);
          setSelectedColB(topKey.table_b_column);
        }
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to analyze relationships.');
    } finally {
      setIsDetectingRels(false);
    }
  };

  const handlePreviewJoin = async () => {
    if (!selectedJoinTableA || !selectedJoinTableB || !selectedColA || !selectedColB) {
      setErrorMsg('Please select both tables and join columns.');
      return;
    }
    setIsUploading(true);
    setErrorMsg(null);
    try {
      const prev = await previewTableJoin({
        table_a_id: selectedJoinTableA,
        table_b_id: selectedJoinTableB,
        join_column_a: selectedColA,
        join_column_b: selectedColB,
        join_type: selectedJoinType
      });
      setJoinPreview(prev);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to preview join.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleApplyJoin = async () => {
    if (!selectedJoinTableA || !selectedJoinTableB || !selectedColA || !selectedColB) return;
    setIsUploading(true);
    setErrorMsg(null);
    try {
      const report = await applyTableJoin({
        table_a_id: selectedJoinTableA,
        table_b_id: selectedJoinTableB,
        join_column_a: selectedColA,
        join_column_b: selectedColB,
        join_type: selectedJoinType
      });
      onSuccess(report);
      onClose();
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to execute join.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-[#0F3D3E]/40 backdrop-blur-xs flex items-center justify-center p-3 sm:p-4 overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl max-h-[90vh] bg-white rounded-3xl border border-[#EAE6DF] shadow-2xl flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150 my-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header - Fixed & Pinned */}
        <div className="p-5 border-b border-[#F0EDE6] flex items-center justify-between shrink-0 bg-white">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-[#FFF7ED] text-[#EA580C] flex items-center justify-center">
              <Upload className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-heading font-bold text-xl text-[#0F3D3E]">
                Upload CSV, Excel, or ZIP Datasets
              </h3>
              <p className="text-xs text-[#648484]">
                Single file or multi-table archives with grounded schema detection
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-full hover:bg-[#FAF8F5] text-[#648484] hover:text-[#0F3D3E] transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable Body */}
        <div className="p-5 sm:p-6 space-y-5 overflow-y-auto flex-1">
          {/* Dropzone */}
          {!multiInventory && (
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
              className={`p-6 sm:p-7 rounded-2xl border-2 border-dashed transition-all text-center cursor-pointer ${
                dragActive
                  ? 'border-[#EA580C] bg-[#FFF7ED]'
                  : selectedFiles.length > 0
                  ? 'border-[#0D9488] bg-[#F0FDFA]'
                  : 'border-[#EAE6DF] bg-[#FAF8F5] hover:border-[#EA580C]/50 hover:bg-[#FFFBF5]'
              }`}
            >
              <input
                ref={inputRef}
                type="file"
                multiple
                accept=".csv,.xlsx,.xls,.zip,.txt,.tsv"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) {
                    validateAndSetFiles(Array.from(e.target.files));
                  }
                }}
              />

              {selectedFiles.length > 0 ? (
                <div className="space-y-3">
                  <FileSpreadsheet className="w-11 h-11 text-[#0D9488] mx-auto animate-bounce-short" />
                  <div>
                    <div className="font-bold text-base text-[#0F3D3E] break-all">
                      {selectedFiles.length === 1
                        ? selectedFiles[0].name
                        : `${selectedFiles.length} files selected (${(
                            selectedFiles.reduce((acc, f) => acc + f.size, 0) /
                            1024 /
                            1024
                          ).toFixed(2)} MB total)`}
                    </div>
                    <div className="text-sm text-[#0D9488] font-medium mt-0.5">
                      Ready for validated ingestion
                    </div>
                  </div>
                  <div className="text-xs text-[#648484] hover:text-[#EA580C] underline cursor-pointer">
                    Click to select different files
                  </div>
                </div>
              ) : (
                <div className="space-y-2">
                  <Upload className="w-10 h-10 text-[#A5C0C0] mx-auto" />
                  <div className="text-base font-semibold text-[#0F3D3E]">
                    Drag and drop your spreadsheet(s) or ZIP archive here
                  </div>
                  <div className="text-sm text-[#648484]">
                    Accepts CSV (.csv), Excel (.xlsx, .xls), or ZIP archives up to 100 MB
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Worksheet Selection when Excel has multiple sheets */}
          {workbookInfo && workbookInfo.sheets.length > 1 && !multiInventory && (
            <div className="p-4 bg-[#FAF8F5] rounded-2xl border border-[#EAE6DF] space-y-3">
              <div className="flex items-center gap-2 text-sm font-bold text-[#0F3D3E]">
                <Layers className="w-4 h-4 text-[#EA580C]" />
                <span>Workbook Worksheets Detected ({workbookInfo.sheets.length})</span>
              </div>
              <p className="text-xs text-[#648484]">
                Choose which worksheet to analyze, or view worksheet dimensions:
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-44 overflow-y-auto pr-1">
                {workbookInfo.sheets.map((sheet) => (
                  <button
                    key={sheet.sheet_name}
                    type="button"
                    onClick={() => setSelectedSheet(sheet.sheet_name)}
                    className={`p-3 rounded-xl border text-left transition-all cursor-pointer flex flex-col justify-between ${
                      selectedSheet === sheet.sheet_name
                        ? 'border-[#EA580C] bg-[#FFF7ED] ring-1 ring-[#EA580C]'
                        : 'border-[#EAE6DF] bg-white hover:border-[#648484]/40'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-sm text-[#0F3D3E] truncate">{sheet.sheet_name}</span>
                      {selectedSheet === sheet.sheet_name && (
                        <CheckCircle2 className="w-4 h-4 text-[#EA580C] shrink-0" />
                      )}
                    </div>
                    <div className="text-xs text-[#648484] mt-1">
                      {sheet.is_empty ? (
                        <span className="text-[#DC2626] font-medium">Empty (0 rows)</span>
                      ) : (
                        `${sheet.rows.toLocaleString()} rows • ${sheet.columns} cols`
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Multi-Table Inventory Display */}
          {multiInventory && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-bold text-base text-[#0F3D3E] flex items-center gap-2">
                    <TableIcon className="w-4 h-4 text-[#0D9488]" />
                    <span>Table Inventory ({multiInventory.total_tables} Tables Ingested)</span>
                  </h4>
                  <p className="text-xs text-[#648484]">
                    Select any table to analyze individually, or detect candidate relationships to join.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={handleDetectRelationships}
                  disabled={isDetectingRels || multiInventory.tables.length < 2}
                  className="bg-[#0F3D3E] hover:bg-[#1B4D4E] text-white px-3.5 py-1.5 rounded-full text-xs font-bold transition-all shadow-2xs inline-flex items-center gap-1.5 cursor-pointer disabled:opacity-40"
                >
                  <Link2 className="w-3.5 h-3.5 text-[#A5C0C0]" />
                  <span>{isDetectingRels ? 'Inspecting...' : 'Detect Relationships'}</span>
                </button>
              </div>

              {/* Table List */}
              <div className="space-y-2 max-h-52 overflow-y-auto pr-1">
                {multiInventory.tables.map((tbl) => (
                  <div
                    key={tbl.table_id}
                    className="p-3 bg-[#FAF8F5] rounded-xl border border-[#EAE6DF] flex items-center justify-between gap-3"
                  >
                    <div className="min-w-0">
                      <div className="font-bold text-sm text-[#0F3D3E] truncate">
                        {tbl.filename} {tbl.sheet_name ? `• ${tbl.sheet_name}` : ''}
                      </div>
                      <div className="text-xs text-[#648484] mt-0.5">
                        {tbl.rows.toLocaleString()} rows • {tbl.columns} columns ({tbl.column_names.slice(0, 3).join(', ')}
                        {tbl.column_names.length > 3 ? '...' : ''})
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleSelectStandaloneTable(tbl.table_id)}
                      className="bg-white hover:bg-[#FFF7ED] text-[#0F3D3E] hover:text-[#EA580C] px-3 py-1.5 rounded-full border border-[#EAE6DF] text-xs font-bold transition-all cursor-pointer shrink-0"
                    >
                      Analyze Table
                    </button>
                  </div>
                ))}
              </div>

              {/* Relationships & Join Discovery Section */}
              {relationships && (
                <div className="p-4 bg-[#F0FDFA] rounded-2xl border border-[#99F6E4] space-y-3 animate-in fade-in">
                  <div className="font-bold text-sm text-[#0F3D3E] flex items-center gap-2">
                    <Link2 className="w-4 h-4 text-[#0D9488]" />
                    <span>Schema Relationship Analysis</span>
                  </div>

                  {relationships.relationships.map((rel, idx) => (
                    <div key={idx} className="bg-white p-3 rounded-xl border border-[#CCFBF1] space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-bold text-[#0F3D3E]">
                          {rel.table_a_name} ↔ {rel.table_b_name}
                        </span>
                        <span
                          className={`px-2 py-0.5 rounded-full font-bold uppercase text-[10px] ${
                            rel.status === 'related'
                              ? 'bg-[#DCFCE7] text-[#15803D]'
                              : rel.status === 'ambiguous'
                              ? 'bg-[#FEF3C7] text-[#B45309]'
                              : 'bg-[#F3F4F6] text-[#6B7280]'
                          }`}
                        >
                          {rel.status}
                        </span>
                      </div>
                      <p className="text-xs text-[#476D6D]">{rel.evidence_summary}</p>
                    </div>
                  ))}

                  {/* Join Configuration Box */}
                  <div className="p-3 bg-white rounded-xl border border-[#EAE6DF] space-y-2">
                    <div className="text-xs font-bold text-[#0F3D3E]">Propose Table Join:</div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                      <div>
                        <label className="block text-[#648484] mb-1">Left Key</label>
                        <input
                          type="text"
                          value={selectedColA}
                          onChange={(e) => setSelectedColA(e.target.value)}
                          className="w-full p-1.5 border border-[#EAE6DF] rounded-md font-mono text-xs"
                          placeholder="e.g. customer_id"
                        />
                      </div>
                      <div>
                        <label className="block text-[#648484] mb-1">Right Key</label>
                        <input
                          type="text"
                          value={selectedColB}
                          onChange={(e) => setSelectedColB(e.target.value)}
                          className="w-full p-1.5 border border-[#EAE6DF] rounded-md font-mono text-xs"
                          placeholder="e.g. customer_id"
                        />
                      </div>
                      <div>
                        <label className="block text-[#648484] mb-1">Join Type</label>
                        <select
                          value={selectedJoinType}
                          onChange={(e) => setSelectedJoinType(e.target.value)}
                          className="w-full p-1.5 border border-[#EAE6DF] rounded-md text-xs bg-white"
                        >
                          <option value="inner">Inner Join</option>
                          <option value="left">Left Join</option>
                          <option value="right">Right Join</option>
                          <option value="outer">Full Outer</option>
                        </select>
                      </div>
                      <div className="flex items-end">
                        <button
                          type="button"
                          onClick={handlePreviewJoin}
                          disabled={isUploading}
                          className="w-full bg-[#EA580C] hover:bg-[#C2410C] text-white p-1.5 rounded-md font-bold text-xs transition-colors cursor-pointer"
                        >
                          Preview Join
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Join Preview Output */}
                  {joinPreview && (
                    <div className="p-3.5 bg-white rounded-xl border border-[#0D9488] space-y-2.5 animate-in fade-in">
                      <div className="flex items-center justify-between text-xs font-bold text-[#0F3D3E]">
                        <span>Expected Join Outcome</span>
                        <span className="text-[#0D9488] font-mono">
                          {joinPreview.expected_rows.toLocaleString()} rows • {joinPreview.columns_after_join.length} columns
                        </span>
                      </div>

                      {joinPreview.cartesian_multiplication_risk && (
                        <div className="p-2.5 bg-[#FEF2F2] border border-[#FCA5A5] rounded-lg text-xs text-[#B91C1C] flex items-center gap-2">
                          <AlertTriangle className="w-4 h-4 shrink-0" />
                          <span>{joinPreview.risk_warning}</span>
                        </div>
                      )}

                      <div className="text-xs text-[#648484] grid grid-cols-2 gap-2">
                        <div>Unmatched in Left: {joinPreview.unmatched_rows_a.toLocaleString()}</div>
                        <div>Unmatched in Right: {joinPreview.unmatched_rows_b.toLocaleString()}</div>
                      </div>

                      <div className="pt-2 flex justify-end">
                        <button
                          type="button"
                          onClick={handleApplyJoin}
                          disabled={isUploading}
                          className="bg-[#0D9488] hover:bg-[#0F766E] text-white px-4 py-2 rounded-full text-xs font-bold transition-all shadow-xs flex items-center gap-1.5 cursor-pointer"
                        >
                          <CheckCircle2 className="w-4 h-4" />
                          <span>Approve & Join Tables</span>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {errorMsg && (
            <div className="p-3.5 bg-[#FEF2F2] rounded-xl border border-[#FCA5A5] text-sm text-[#B91C1C] flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Quick Sample Links */}
          {!multiInventory && (
            <div className="space-y-2.5 pt-2 border-t border-[#F0EDE6]">
              <span className="text-sm font-semibold text-[#476D6D] block">
                Or explore with a built-in sample:
              </span>
              <div className="flex flex-wrap gap-2">
                {samples.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => {
                      onSelectSample(s.id);
                      onClose();
                    }}
                    className="text-sm bg-[#FAF8F5] hover:bg-[#FFF7ED] text-[#0F3D3E] hover:text-[#EA580C] px-3.5 py-2 rounded-full border border-[#EAE6DF] transition-all flex items-center gap-1.5 shadow-2xs cursor-pointer font-medium"
                  >
                    <Database className="w-3.5 h-3.5 text-[#EA580C]" />
                    <span>{s.name}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer - Pinned to bottom */}
        <div className="p-4 bg-[#FAF8F5] border-t border-[#EAE6DF] flex items-center justify-end gap-3 shrink-0">
          <button
            onClick={onClose}
            className="px-5 py-2.5 rounded-full border border-[#EAE6DF] text-sm font-semibold text-[#476D6D] hover:bg-white transition-all cursor-pointer"
          >
            Cancel
          </button>
          {!multiInventory && (
            <button
              onClick={handleUploadSubmit}
              disabled={selectedFiles.length === 0 || isUploading}
              className="bg-[#EA580C] hover:bg-[#C2410C] text-white px-6 py-2.5 rounded-full text-sm font-bold transition-all shadow-xs flex items-center gap-2 cursor-pointer disabled:opacity-40"
            >
              {isUploading ? (
                <>
                  <div className="w-4 h-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
                  <span>Ingesting & Profiling...</span>
                </>
              ) : (
                <>
                  <span>
                    {selectedFiles.length > 1 || (selectedFiles[0] && selectedFiles[0].name.endsWith('.zip'))
                      ? 'Process Multi-Table Upload'
                      : 'Start Ingestion & Profiling'}
                  </span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
