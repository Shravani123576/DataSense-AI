import React, { useState, useRef } from 'react';
import {
  Upload,
  X,
  FileSpreadsheet,
  AlertCircle,
  Database,
  ArrowRight,
  Layers
} from 'lucide-react';

import { uploadDatasetFile } from '../api';
import type { ProfileReport, SampleDatasetInfo } from '../types';

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
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
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
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const validateAndSetFile = (file: File) => {
    const validExtensions = ['.csv', '.xlsx', '.xls'];
    const hasValidExt = validExtensions.some((ext) => file.name.toLowerCase().endsWith(ext));

    if (!hasValidExt) {
      setErrorMsg('Please upload a valid CSV (.csv) or Excel (.xlsx, .xls) spreadsheet.');
      setSelectedFile(null);
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setErrorMsg('File size exceeds 50 MB limit.');
      setSelectedFile(null);
      return;
    }

    setErrorMsg(null);
    setSelectedFile(file);
  };

  const handleUploadSubmit = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setErrorMsg(null);

    try {
      const report = await uploadDatasetFile(selectedFile);
      onSuccess(report);
      onClose();
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to upload and analyze dataset.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-[#0F3D3E]/40 backdrop-blur-xs flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl bg-white rounded-3xl border border-[#EAE6DF] shadow-xl overflow-hidden animate-in fade-in zoom-in-95 duration-150"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-5 border-b border-[#F0EDE6] flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-full bg-[#FFF7ED] text-[#EA580C] flex items-center justify-center">
              <Upload className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-heading font-bold text-base text-[#0F3D3E]">
                Upload CSV or Excel Dataset
              </h3>
              <p className="text-xs text-[#648484]">Supports 100,000+ rows with instant profiling</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-full hover:bg-[#FAF8F5] text-[#648484] hover:text-[#0F3D3E] transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-6 space-y-5">
          {/* Dropzone */}
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            className={`p-8 rounded-2xl border-2 border-dashed transition-all text-center cursor-pointer ${
              dragActive
                ? 'border-[#EA580C] bg-[#FFF7ED]'
                : selectedFile
                ? 'border-[#0D9488] bg-[#F0FDFA]'
                : 'border-[#EAE6DF] bg-[#FAF8F5] hover:border-[#EA580C]/50 hover:bg-[#FFFBF5]'
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".csv,.xlsx,.xls"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  validateAndSetFile(e.target.files[0]);
                }
              }}
            />

            {selectedFile ? (
              <div className="space-y-2">
                <FileSpreadsheet className="w-10 h-10 text-[#0D9488] mx-auto" />
                <div className="font-semibold text-sm text-[#0F3D3E]">{selectedFile.name}</div>
                <div className="text-xs text-[#648484]">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Ready for analysis
                </div>
                <span className="inline-block text-xs text-[#EA580C] underline pt-1">
                  Click to select a different file
                </span>
              </div>
            ) : (
              <div className="space-y-2">
                <Upload className="w-8 h-8 text-[#A5C0C0] mx-auto" />
                <div className="text-sm font-semibold text-[#0F3D3E]">
                  Drag and drop your spreadsheet here
                </div>
                <div className="text-xs text-[#648484]">
                  Accepts CSV (.csv) or Excel (.xlsx, .xls) up to 50 MB
                </div>
                <div className="inline-flex items-center gap-1.5 text-xs text-[#EA580C] bg-white px-3 py-1 rounded-full border border-[#FED7AA] shadow-2xs font-medium">
                  <Layers className="w-3.5 h-3.5" />
                  <span>Optimized for datasets with 100,000+ rows</span>
                </div>
              </div>
            )}
          </div>

          {errorMsg && (
            <div className="p-3 bg-[#FEF2F2] rounded-xl border border-[#FCA5A5] text-xs text-[#B91C1C] flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Quick Sample Links */}
          <div className="space-y-2 pt-2 border-t border-[#F0EDE6]">
            <span className="text-xs font-semibold text-[#476D6D] uppercase tracking-wider block">
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
                  className="text-xs bg-[#FAF8F5] hover:bg-[#FFF7ED] text-[#0F3D3E] hover:text-[#EA580C] px-3 py-1.5 rounded-full border border-[#EAE6DF] transition-all flex items-center gap-1.5 shadow-2xs cursor-pointer"
                >
                  <Database className="w-3 h-3 text-[#EA580C]" />
                  <span>{s.name}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-[#FAF8F5] border-t border-[#EAE6DF] flex items-center justify-end gap-2.5">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-full border border-[#EAE6DF] text-xs font-medium text-[#476D6D] hover:bg-white transition-all cursor-pointer"
          >
            Cancel
          </button>
          <button
            onClick={handleUploadSubmit}
            disabled={!selectedFile || isUploading}
            className="bg-[#EA580C] hover:bg-[#C2410C] text-white px-5 py-2 rounded-full text-xs font-medium transition-all shadow-xs flex items-center gap-2 cursor-pointer disabled:opacity-40"
          >
            {isUploading ? (
              <>
                <div className="w-3.5 h-3.5 rounded-full border-2 border-white border-t-transparent animate-spin" />
                <span>Ingesting & Profiling...</span>
              </>
            ) : (
              <>
                <span>Start Ingestion & Profiling</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
