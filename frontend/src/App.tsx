import React, { useState, useEffect } from 'react';
import {
  MessageSquareCode,
  ShieldCheck,
  BarChart3,
  Sparkles,
  AlertCircle,
  Database,
  RefreshCw,
  Utensils,
  Upload,
  ArrowRight,
  Table as TableIcon,
  ChevronDown,
  ChevronUp,
  FileText
} from 'lucide-react';
import type { ProfileReport, SampleDatasetInfo } from './types';

import { fetchSamples, loadSampleDataset, uploadDatasetFile } from './api';
import { Header } from './components/Header';
import { ChatAgent } from './components/ChatAgent';
import { DataHealthRadar } from './components/DataHealthRadar';
import { VisualizerStudio } from './components/VisualizerStudio';
import { DataGridPreview } from './components/DataGridPreview';
import { ZomatoHub } from './components/ZomatoHub';
import { UploadModal } from './components/UploadModal';
import { ApiKeyModal } from './components/ApiKeyModal';
import { CleanStudio } from './components/CleanStudio';
import { MLStudio } from './components/MLStudio';
import { AgentView } from './components/AgentView';
import { ReportView } from './components/ReportView';

export const App: React.FC = () => {
  const [profile, setProfile] = useState<ProfileReport | null>(null);
  const [samples, setSamples] = useState<SampleDatasetInfo[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<string>('');
  const [activeTab, setActiveTab] = useState<'review' | 'clean' | 'charts' | 'predict' | 'copilot' | 'export' | 'zomato'>('review');
  const [copilotSubTab, setCopilotSubTab] = useState<'chat' | 'planner'>('chat');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Review sub-accordions (keeping advanced details optional)
  const [showTablePreview, setShowTablePreview] = useState<boolean>(false);
  const [showFeatureDictionary, setShowFeatureDictionary] = useState<boolean>(false);

  // Modals state
  const [isUploadOpen, setIsUploadOpen] = useState<boolean>(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [apiKey, setApiKey] = useState<string>(() => {
    return localStorage.getItem('datasense_gemini_key') || '';
  });

  const handleSaveApiKey = (newKey: string) => {
    setApiKey(newKey);
    if (newKey) {
      localStorage.setItem('datasense_gemini_key', newKey);
    } else {
      localStorage.removeItem('datasense_gemini_key');
    }
  };

  // Initial load: Fetch sample datasets list, but do NOT automatically load Zomato
  useEffect(() => {
    const initApp = async () => {
      setIsLoading(true);
      setErrorMsg(null);
      try {
        const samplesList = await fetchSamples();
        setSamples(samplesList);
        // Start in clean onboarding state; datasets are loaded on-demand
      } catch (err: any) {
        console.error('Failed to initialize app', err);
        setErrorMsg('Failed to connect to DataSense AI backend. Ensure the server is running on port 8000.');
      } finally {
        setIsLoading(false);
      }
    };

    initApp();
  }, []);

  const handleSelectSample = async (sampleId: string) => {
    setSelectedSampleId(sampleId);
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const report = await loadSampleDataset(sampleId);
      setProfile(report);
      // If user explicitly selected Zomato, switch to Zomato Hub, otherwise start in Review
      if (sampleId === 'zomato_restaurants') {
        setActiveTab('zomato');
      } else {
        setActiveTab('review');
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to load sample dataset');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadSuccess = (newProfile: ProfileReport) => {
    setProfile(newProfile);
    setSelectedSampleId('');
    setActiveTab('review');
  };

  const handleResetDataset = () => {
    setProfile(null);
    setSelectedSampleId('');
    setActiveTab('review');
    setErrorMsg(null);
  };

  // Direct file drop handler for onboarding screen
  const handleDropFile = async (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setIsLoading(true);
      setErrorMsg(null);
      try {
        const report = await uploadDatasetFile(file);
        handleUploadSuccess(report);
      } catch (err: any) {
        setErrorMsg(err.message || 'Failed to upload dataset.');
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="min-h-screen bg-[#FAF8F5] text-[#0F3D3E] flex flex-col font-sans selection:bg-[#FFF7ED] selection:text-[#EA580C]">
      {/* Header */}
      <Header
        profile={profile}
        samples={samples}
        selectedSampleId={selectedSampleId}
        onSelectSample={handleSelectSample}
        onOpenUpload={() => setIsUploadOpen(true)}
        onOpenSettings={() => setIsSettingsOpen(true)}
        isLoading={isLoading}
      />

      {/* Main Content Layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 md:px-8 py-6 space-y-6">
        {errorMsg && (
          <div className="p-4 rounded-2xl bg-[#FEF2F2] border border-[#FCA5A5] text-xs text-[#B91C1C] flex items-center justify-between shadow-xs">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
            <button
              onClick={() => setErrorMsg(null)}
              className="p-1 hover:bg-white rounded-lg transition-colors cursor-pointer text-xs font-semibold"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* 1. ONBOARDING / EMPTY STATE: Upload Dataset or Pick Sample */}
        {!profile && !isLoading && (
          <div className="space-y-8 py-4 animate-in fade-in duration-300">
            {/* Hero Card */}
            <div className="bg-white p-8 md:p-12 rounded-3xl border border-[#EAE6DF] shadow-xs text-center space-y-4 max-w-4xl mx-auto">
              <div className="inline-flex items-center gap-2 bg-[#FFF7ED] text-[#EA580C] px-3.5 py-1.5 rounded-full text-xs font-bold border border-[#FED7AA]">
                <Sparkles className="w-3.5 h-3.5 fill-current/20" />
                <span>Beginner-Friendly AI Data Science Studio</span>
              </div>
              <h1 className="text-3xl md:text-4xl font-extrabold text-[#0F3D3E] tracking-tight font-heading">
                Turn any spreadsheet into verified insights in seconds.
              </h1>
              <p className="text-sm md:text-base text-[#476D6D] max-w-2xl mx-auto leading-relaxed">
                Upload your CSV or Excel dataset. DataSense AI validates data types, detects anomalies, calculates accurate statistics, generates interactive Plotly charts, and answers questions in plain English—with zero mathematical hallucination.
              </p>

              {/* 4-Step Flow Pill Indicators */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 max-w-3xl mx-auto text-left">
                <div className="bg-[#FAF8F5] p-3 rounded-2xl border border-[#EAE6DF]">
                  <span className="text-[10px] font-bold text-[#EA580C] uppercase tracking-wider block">Step 1</span>
                  <strong className="text-xs font-semibold text-[#0F3D3E] block">Upload Dataset</strong>
                  <span className="text-[11px] text-[#648484]">CSV or Excel (100k+ rows)</span>
                </div>
                <div className="bg-[#FAF8F5] p-3 rounded-2xl border border-[#EAE6DF]">
                  <span className="text-[10px] font-bold text-[#0D9488] uppercase tracking-wider block">Step 2</span>
                  <strong className="text-xs font-semibold text-[#0F3D3E] block">Review Data</strong>
                  <span className="text-[11px] text-[#648484]">Health score & quality audit</span>
                </div>
                <div className="bg-[#FAF8F5] p-3 rounded-2xl border border-[#EAE6DF]">
                  <span className="text-[10px] font-bold text-[#6366F1] uppercase tracking-wider block">Step 3</span>
                  <strong className="text-xs font-semibold text-[#0F3D3E] block">Ask Questions</strong>
                  <span className="text-[11px] text-[#648484]">Plain English conversational AI</span>
                </div>
                <div className="bg-[#FAF8F5] p-3 rounded-2xl border border-[#EAE6DF]">
                  <span className="text-[10px] font-bold text-[#059669] uppercase tracking-wider block">Step 4</span>
                  <strong className="text-xs font-semibold text-[#0F3D3E] block">View Results</strong>
                  <span className="text-[11px] text-[#648484]">Plotly charts & verified code</span>
                </div>
              </div>

              {/* Dropzone Area */}
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDropFile}
                onClick={() => setIsUploadOpen(true)}
                className="mt-6 border-2 border-dashed border-[#FED7AA] hover:border-[#EA580C] bg-[#FFFBF5] hover:bg-[#FFF7ED]/50 rounded-3xl p-8 transition-all cursor-pointer flex flex-col items-center justify-center space-y-3 group"
              >
                <div className="w-14 h-14 rounded-2xl bg-[#FFF7ED] text-[#EA580C] group-hover:scale-105 transition-transform flex items-center justify-center shadow-xs">
                  <Upload className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-[#0F3D3E]">
                    Drag & Drop your CSV or Excel file here
                  </h3>
                  <p className="text-xs text-[#648484] mt-1">
                    Supports .csv, .xlsx, and .xls spreadsheets up to 100,000+ rows
                  </p>
                </div>
                <button
                  type="button"
                  className="bg-[#EA580C] hover:bg-[#C2410C] text-white px-5 py-2 rounded-full text-xs font-semibold transition-all shadow-xs cursor-pointer"
                >
                  Browse Files
                </button>
              </div>
            </div>

            {/* Sample Datasets Grid */}
            <div className="max-w-4xl mx-auto space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-[#476D6D] uppercase tracking-wider">
                  Or explore a beginner sample dataset:
                </span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {samples.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => handleSelectSample(s.id)}
                    className="bg-white p-5 rounded-2xl border border-[#EAE6DF] hover:border-[#EA580C] shadow-xs text-left transition-all hover:-translate-y-0.5 cursor-pointer flex flex-col justify-between space-y-3 group"
                  >
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-bold text-[#EA580C] bg-[#FFF7ED] px-2 py-0.5 rounded-full border border-[#FED7AA]">
                          {s.category}
                        </span>
                        {s.id === 'bigdata_100k' && (
                          <span className="text-[10px] font-bold text-[#4F46E5] bg-[#EEF2FF] px-1.5 py-0.5 rounded-md">
                            100k+ Rows
                          </span>
                        )}
                        {s.id === 'zomato_restaurants' && (
                          <span className="text-[10px] font-bold text-[#D97706] bg-[#FEF3C7] px-1.5 py-0.5 rounded-md">
                            Culinary
                          </span>
                        )}
                      </div>
                      <h4 className="font-heading font-bold text-sm text-[#0F3D3E] group-hover:text-[#EA580C] transition-colors">
                        {s.name}
                      </h4>
                      <p className="text-xs text-[#648484] line-clamp-2 leading-relaxed">
                        {s.description}
                      </p>
                    </div>

                    <div className="flex items-center justify-between pt-2 border-t border-[#F5F2EB] text-[11px] text-[#476D6D]">
                      <span>{s.rows.toLocaleString()} rows • {s.columns} cols</span>
                      <span className="text-[#EA580C] font-semibold flex items-center gap-1 group-hover:translate-x-0.5 transition-transform">
                        Explore <ArrowRight className="w-3 h-3" />
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* 2. ACTIVE DATASET STATE */}
        {profile && (
          <>
            {/* Active Dataset Banner with Clean Metrics */}
            <div className="bg-white p-4 md:p-5 rounded-3xl border border-[#EAE6DF] shadow-xs flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3.5">
                <div className="w-11 h-11 rounded-2xl bg-[#FFF7ED] text-[#EA580C] flex items-center justify-center shrink-0">
                  <Database className="w-5 h-5" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="font-heading font-bold text-base md:text-lg text-[#0F3D3E]">
                      {profile.metadata.name}
                    </h2>
                    <span className="text-[11px] font-bold bg-[#ECFDF5] text-[#059669] px-2.5 py-0.5 rounded-full border border-[#A7F3D0]">
                      Quality Grade {profile.health.grade} ({profile.health.overall_score}/100)
                    </span>
                  </div>
                  <p className="text-xs text-[#648484]">
                    Source file: <code className="bg-[#FAF8F5] px-1.5 py-0.5 rounded border border-[#EAE6DF] text-[#0F3D3E]">{profile.metadata.filename}</code> • {profile.metadata.rows.toLocaleString()} records • {profile.metadata.columns} features • {profile.metadata.missing_cells} missing ({profile.metadata.missing_percentage}%)
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsUploadOpen(true)}
                  className="flex items-center gap-1.5 bg-[#FAF8F5] hover:bg-[#F5F2EB] text-[#0F3D3E] px-3.5 py-2 rounded-full border border-[#EAE6DF] text-xs font-semibold transition-all cursor-pointer shadow-2xs"
                >
                  <Upload className="w-3.5 h-3.5 text-[#EA580C]" />
                  <span>Upload Another File</span>
                </button>
                <button
                  onClick={handleResetDataset}
                  className="p-2 hover:bg-[#FEF2F2] text-[#648484] hover:text-[#DC2626] rounded-full transition-colors cursor-pointer"
                  title="Close active dataset"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* 6-Stage Data Science Lifecycle Navigation Bar */}
            <div className="flex items-center gap-1.5 p-1 bg-[#F5F2EB] rounded-2xl border border-[#EAE6DF] w-fit overflow-x-auto max-w-full">
              <button
                onClick={() => setActiveTab('review')}
                id="tab-understand-data"
                className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs md:text-sm font-medium transition-all cursor-pointer whitespace-nowrap ${
                  activeTab === 'review'
                    ? 'bg-white text-[#0F3D3E] shadow-xs font-bold border border-[#E2E8F0]'
                    : 'text-[#476D6D] hover:text-[#0F3D3E]'
                }`}
              >
                <ShieldCheck className="w-4 h-4 text-[#0D9488]" />
                <span>1. Understand</span>
              </button>

              <button
                onClick={() => setActiveTab('clean')}
                id="tab-clean-studio"
                className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs md:text-sm font-medium transition-all cursor-pointer whitespace-nowrap ${
                  activeTab === 'clean'
                    ? 'bg-white text-[#EA580C] shadow-xs font-bold border border-[#E2E8F0]'
                    : 'text-[#476D6D] hover:text-[#0F3D3E]'
                }`}
              >
                <Database className="w-4 h-4 text-[#EA580C]" />
                <span>2. Clean Studio</span>
              </button>

              <button
                onClick={() => setActiveTab('charts')}
                id="tab-visualizer"
                className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs md:text-sm font-medium transition-all cursor-pointer whitespace-nowrap ${
                  activeTab === 'charts'
                    ? 'bg-white text-[#0F3D3E] shadow-xs font-bold border border-[#E2E8F0]'
                    : 'text-[#476D6D] hover:text-[#0F3D3E]'
                }`}
              >
                <BarChart3 className="w-4 h-4 text-[#6366F1]" />
                <span>3. Analyze</span>
              </button>

              <button
                onClick={() => setActiveTab('predict')}
                id="tab-predict-studio"
                className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs md:text-sm font-medium transition-all cursor-pointer whitespace-nowrap ${
                  activeTab === 'predict'
                    ? 'bg-white text-[#0F3D3E] shadow-xs font-bold border border-[#E2E8F0]'
                    : 'text-[#476D6D] hover:text-[#0F3D3E]'
                }`}
              >
                <Sparkles className="w-4 h-4 text-[#EA580C]" />
                <span>4. Predict (AutoML)</span>
              </button>

              <button
                onClick={() => setActiveTab('copilot')}
                id="tab-copilot-agent"
                className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs md:text-sm font-medium transition-all cursor-pointer whitespace-nowrap ${
                  activeTab === 'copilot'
                    ? 'bg-white text-[#0F3D3E] shadow-xs font-bold border border-[#E2E8F0]'
                    : 'text-[#476D6D] hover:text-[#0F3D3E]'
                }`}
              >
                <MessageSquareCode className="w-4 h-4 text-[#0D9488]" />
                <span>5. Agent Copilot</span>
              </button>

              <button
                onClick={() => setActiveTab('export')}
                id="tab-executive-report"
                className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs md:text-sm font-medium transition-all cursor-pointer whitespace-nowrap ${
                  activeTab === 'export'
                    ? 'bg-white text-[#0F3D3E] shadow-xs font-bold border border-[#E2E8F0]'
                    : 'text-[#476D6D] hover:text-[#0F3D3E]'
                }`}
              >
                <FileText className="w-4 h-4 text-[#D97706]" />
                <span>6. Executive Brief</span>
              </button>

              {/* Show Zomato Tab ONLY if the user explicitly loaded Zomato Bangalore Restaurants */}
              {profile.is_restaurant_dataset && (
                <button
                  onClick={() => setActiveTab('zomato')}
                  id="tab-zomato-hub"
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs md:text-sm font-medium transition-all cursor-pointer whitespace-nowrap ${
                    activeTab === 'zomato'
                      ? 'bg-white text-[#EA580C] shadow-xs font-bold border border-[#E2E8F0]'
                      : 'text-[#476D6D] hover:text-[#0F3D3E]'
                  }`}
                >
                  <Utensils className="w-4 h-4 text-[#EA580C]" />
                  <span>Zomato Hub</span>
                </button>
              )}
            </div>

            {/* TAB CONTENT SECTIONS */}
            {activeTab === 'clean' && (
              <CleanStudio dataset={profile} onDatasetUpdated={setProfile} />
            )}

            {activeTab === 'predict' && (
              <MLStudio dataset={profile} />
            )}

            {activeTab === 'copilot' && (
              <div className="space-y-4">
                {/* Copilot Mode Sub-Toggle */}
                <div className="flex items-center gap-2 bg-white p-1 rounded-xl border border-[#EAE6DF] w-fit shadow-xs">
                  <button
                    onClick={() => setCopilotSubTab('chat')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                      copilotSubTab === 'chat'
                        ? 'bg-[#FAF8F5] text-[#EA580C] border border-[#FED7AA]'
                        : 'text-[#648484] hover:text-[#0F3D3E]'
                    }`}
                  >
                    Plain-English Q&A (Grounded)
                  </button>
                  <button
                    onClick={() => setCopilotSubTab('planner')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                      copilotSubTab === 'planner'
                        ? 'bg-[#FAF8F5] text-[#0F3D3E] border border-[#CBD5E1]'
                        : 'text-[#648484] hover:text-[#0F3D3E]'
                    }`}
                  >
                    Autonomous Goal Planner & Execution
                  </button>
                </div>

                {copilotSubTab === 'chat' ? (
                  <ChatAgent profile={profile} apiKey={apiKey} />
                ) : (
                  <AgentView dataset={profile} />
                )}
              </div>
            )}

            {activeTab === 'export' && (
              <ReportView dataset={profile} />
            )}

            {activeTab === 'review' && (
              <div className="space-y-6">
                {/* Health Overview & Pillar Gauges */}
                <DataHealthRadar profile={profile} />

                {/* Optional Expandable: Paginated Data Table */}
                <div className="bg-white rounded-3xl border border-[#EAE6DF] shadow-xs overflow-hidden">
                  <button
                    onClick={() => setShowTablePreview(!showTablePreview)}
                    className="w-full p-5 flex items-center justify-between text-left hover:bg-[#FAF8F5] transition-colors cursor-pointer"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-xl bg-[#F0FDFA] text-[#0D9488] flex items-center justify-center">
                        <TableIcon className="w-5 h-5" />
                      </div>
                      <div>
                        <h3 className="font-heading font-bold text-sm text-[#0F3D3E]">
                          Explore Data Table ({profile.metadata.rows.toLocaleString()} Records)
                        </h3>
                        <p className="text-xs text-[#648484]">
                          Search, paginate, and inspect raw rows with typed column badges
                        </p>
                      </div>
                    </div>
                    {showTablePreview ? <ChevronUp className="w-4 h-4 text-[#648484]" /> : <ChevronDown className="w-4 h-4 text-[#648484]" />}
                  </button>

                  {showTablePreview && (
                    <div className="p-5 border-t border-[#F0EDE6] bg-[#FAF8F5]">
                      <DataGridPreview profile={profile} />
                    </div>
                  )}
                </div>

                {/* Optional Expandable: Statistical Feature Dictionary */}
                <div className="bg-white rounded-3xl border border-[#EAE6DF] shadow-xs overflow-hidden">
                  <button
                    onClick={() => setShowFeatureDictionary(!showFeatureDictionary)}
                    className="w-full p-5 flex items-center justify-between text-left hover:bg-[#FAF8F5] transition-colors cursor-pointer"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-xl bg-[#EEF2FF] text-[#4F46E5] flex items-center justify-center">
                        <FileText className="w-5 h-5" />
                      </div>
                      <div>
                        <h3 className="font-heading font-bold text-sm text-[#0F3D3E]">
                          Statistical Feature Dictionary ({profile.columns.length} Features)
                        </h3>
                        <p className="text-xs text-[#648484]">
                          Inspect inferred data types, completeness, min, max, mean, and IQR ranges
                        </p>
                      </div>
                    </div>
                    {showFeatureDictionary ? <ChevronUp className="w-4 h-4 text-[#648484]" /> : <ChevronDown className="w-4 h-4 text-[#648484]" />}
                  </button>

                  {showFeatureDictionary && (
                    <div className="p-5 border-t border-[#F0EDE6] bg-[#FAF8F5] overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr className="border-b border-[#EAE6DF] text-[#476D6D]">
                            <th className="py-2 px-3 font-semibold">Column</th>
                            <th className="py-2 px-3 font-semibold">Inferred Type</th>
                            <th className="py-2 px-3 font-semibold">Completeness</th>
                            <th className="py-2 px-3 font-semibold">Unique Values</th>
                            <th className="py-2 px-3 font-semibold">Summary Stats</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#EAE6DF]">
                          {profile.columns.map((col, idx) => (
                            <tr key={idx} className="hover:bg-white transition-colors">
                              <td className="py-2.5 px-3 font-semibold text-[#0F3D3E]">{col.column}</td>
                              <td className="py-2.5 px-3">
                                <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                                  col.inferred_type === 'numeric'
                                    ? 'bg-[#EFF6FF] text-[#1D4ED8]'
                                    : col.inferred_type === 'datetime'
                                    ? 'bg-[#F0FDF4] text-[#15803D]'
                                    : col.inferred_type === 'boolean'
                                    ? 'bg-[#FEF3C7] text-[#B45309]'
                                    : 'bg-[#F3E8FF] text-[#7E22CE]'
                                }`}>
                                  {col.inferred_type}
                                </span>
                              </td>
                              <td className="py-2.5 px-3 text-[#476D6D]">
                                {100 - col.null_percentage}% ({col.null_count} nulls)
                              </td>
                              <td className="py-2.5 px-3 text-[#476D6D]">
                                {col.unique_count.toLocaleString()}
                              </td>
                              <td className="py-2.5 px-3 text-[#0F3D3E] font-mono text-[11px]">
                                {col.inferred_type === 'numeric' && col.mean !== undefined
                                  ? `Mean: ${col.mean} | Min: ${col.min} | Max: ${col.max}`
                                  : col.top_categories && col.top_categories.length > 0
                                  ? `Top: ${col.top_categories[0].value} (${col.top_categories[0].percentage}%)`
                                  : '-'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'charts' && (
              <VisualizerStudio profile={profile} />
            )}

            {activeTab === 'zomato' && profile.is_restaurant_dataset && (
              <ZomatoHub profile={profile} />
            )}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-[#EAE6DF] bg-white py-4 px-4 md:px-8 text-center text-xs text-[#648484]">
        <span>DataSense AI V2 • Built with React, FastAPI, Pandas & Plotly • 100% Mathematically Grounded Execution</span>
      </footer>

      {/* Modals */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onSuccess={handleUploadSuccess}
        samples={samples}
        onSelectSample={(sId) => {
          handleSelectSample(sId);
          setIsUploadOpen(false);
        }}
      />

      <ApiKeyModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        apiKey={apiKey}
        onSaveApiKey={handleSaveApiKey}
      />
    </div>
  );
};

export default App;
