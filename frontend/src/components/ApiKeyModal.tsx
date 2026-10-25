import React, { useState } from 'react';
import { Key, X, Shield, CheckCircle, Lock } from 'lucide-react';


interface Props {
  isOpen: boolean;
  onClose: () => void;
  apiKey: string;
  onSaveApiKey: (key: string) => void;
}

export const ApiKeyModal: React.FC<Props> = ({
  isOpen,
  onClose,
  apiKey,
  onSaveApiKey,
}) => {
  const [keyInput, setKeyInput] = useState(apiKey);
  const [savedSuccess, setSavedSuccess] = useState(false);

  if (!isOpen) return null;

  const handleSave = () => {
    onSaveApiKey(keyInput.trim());
    setSavedSuccess(true);
    setTimeout(() => {
      setSavedSuccess(false);
      onClose();
    }, 700);
  };

  const handleClear = () => {
    setKeyInput('');
    onSaveApiKey('');
    setSavedSuccess(true);
    setTimeout(() => {
      setSavedSuccess(false);
      onClose();
    }, 700);
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-[#0F3D3E]/40 backdrop-blur-xs flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg bg-white rounded-3xl border border-[#EAE6DF] shadow-xl overflow-hidden animate-in fade-in zoom-in-95 duration-150"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-5 border-b border-[#F0EDE6] flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-[#FFF7ED] text-[#EA580C] flex items-center justify-center">
              <Key className="w-5 h-5" />
            </div>
            <h3 className="font-heading font-bold text-xl text-[#0F3D3E]">
              API Key Configuration
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-full hover:bg-[#FAF8F5] text-[#648484] hover:text-[#0F3D3E] transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-4">
          {/* Active Mode Banner */}
          <div className="p-4 bg-[#FAF8F5] rounded-2xl border border-[#EAE6DF] space-y-2">
            <div className="flex items-center gap-2">
              <span className={`w-3 h-3 rounded-full ${keyInput ? 'bg-[#0D9488]' : 'bg-[#EA580C]'}`} />
              <strong className="text-sm font-bold text-[#0F3D3E]">
                Active Engine: {keyInput ? 'Google Gemini API Integration' : 'Zero-Key Grounded Heuristic Engine'}
              </strong>
            </div>
            <p className="text-sm text-[#476D6D] leading-relaxed">
              {keyInput
                ? 'Queries use Google Gemini to understand complex business phrasing, construct valid Pandas operations, and synthesize clear natural-language takeaways.'
                : 'DataSense AI is operating in standalone grounded mode. All calculations (aggregations, rankings, IQR outliers, Pearson correlations) are strictly executed in Pandas with zero hallucination!'}
            </p>
          </div>

          {/* Key Input */}
          <div className="space-y-2">
            <label className="text-sm font-bold text-[#476D6D] block">
              Google Gemini API Key (Optional)
            </label>
            <div className="relative">
              <Lock className="w-4 h-4 text-[#648484] absolute left-3.5 top-1/2 -translate-y-1/2" />
              <input
                type="password"
                placeholder="AIzaSy... (or set GEMINI_API_KEY in backend .env)"
                className="w-full bg-[#FAF8F5] border border-[#EAE6DF] rounded-xl pl-10 pr-4 py-2.5 text-sm text-[#0F3D3E] focus:outline-none focus:border-[#EA580C]"
                value={keyInput}
                onChange={(e) => setKeyInput(e.target.value)}
                id="gemini-api-key-input"
              />
            </div>
            <span className="text-xs text-[#648484] block">
              You can also set <code className="bg-[#FAF8F5] px-1 py-0.5 rounded font-mono">GEMINI_API_KEY</code> in the backend environment. Session keys are stored only in your local browser.
            </span>
          </div>

          {/* Zero-Hallucination Guarantee Notice */}
          <div className="p-3.5 bg-[#F0FDFA] rounded-2xl border border-[#CCECE6] flex items-start gap-2.5">
            <Shield className="w-5 h-5 text-[#0D9488] shrink-0 mt-0.5" />
            <div className="text-sm text-[#0F3D3E] leading-relaxed">
              <strong>100% Mathematical Grounding:</strong> Regardless of whether an LLM API key is present, all numbers, sums, counts, and correlation coefficients are calculated by Python's Pandas engine on real data rows.
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-[#FAF8F5] border-t border-[#EAE6DF] flex items-center justify-between">
          <div>
            {keyInput && (
              <button
                type="button"
                onClick={handleClear}
                className="text-sm text-[#B91C1C] hover:underline cursor-pointer font-medium"
              >
                Clear Key
              </button>
            )}
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={onClose}
              className="px-5 py-2.5 rounded-full border border-[#EAE6DF] text-sm font-semibold text-[#476D6D] hover:bg-white transition-all cursor-pointer"
            >
              Close
            </button>
            <button
              type="button"
              onClick={handleSave}
              id="save-api-key-btn"
              className="bg-[#EA580C] hover:bg-[#C2410C] text-white px-6 py-2.5 rounded-full text-sm font-bold transition-all shadow-xs cursor-pointer flex items-center gap-2"
            >
              {savedSuccess ? (
                <>
                  <CheckCircle className="w-4 h-4" />
                  <span>Saved!</span>
                </>
              ) : (
                'Save Settings'
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
