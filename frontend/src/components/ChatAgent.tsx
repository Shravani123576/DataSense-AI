import React, { useState } from 'react';
import {
  Send,
  Sparkles,
  Code2,
  CheckCircle2,
  Clock,
  ArrowRight,
  Table as TableIcon,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  AlertCircle
} from 'lucide-react';
import type { QueryResponse, ProfileReport } from '../types';
import { queryDataset } from '../api';
import { PlotlyChart } from './PlotlyChart';

interface Props {
  profile: ProfileReport;
  apiKey?: string;
  initialQuestion?: string;
}

interface MessageItem {
  id: string;
  sender: 'user' | 'agent';
  text?: string;
  response?: QueryResponse;
  timestamp: string;
}

export const ChatAgent: React.FC<Props> = ({ profile, apiKey }) => {
  const getInitialMessage = (p: ProfileReport): MessageItem => ({
    id: `welcome-${p.metadata.id}`,
    sender: 'agent',
    text: `Hello! I am **DataSense AI**, your beginner-friendly data science assistant. I am currently analyzing **${p.metadata.name}** (${p.metadata.rows.toLocaleString()} records, ${p.metadata.columns} features).\n\nAll calculations are 100% computed directly using Python Pandas on the real dataset without hallucination. Ask any question in plain English or click one of the suggested starter prompts below!`,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  });

  const [messages, setMessages] = useState<MessageItem[]>(() => [getInitialMessage(profile)]);
  const [inputQuestion, setInputQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [expandedCodeIds, setExpandedCodeIds] = useState<Set<string>>(new Set());
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Automatically reset chat messages and display dynamic welcome whenever the dataset changes
  React.useEffect(() => {
    setMessages([getInitialMessage(profile)]);
    setInputQuestion('');
    setExpandedCodeIds(new Set());
  }, [profile.metadata.id, profile.metadata.name]);

  const toggleCode = (id: string) => {
    setExpandedCodeIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const copyCode = (id: string, code: string) => {
    navigator.clipboard.writeText(code);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleSend = async (questionText: string) => {
    const q = questionText.trim();
    if (!q || isLoading) return;

    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsgId = `user-${Date.now()}`;
    const agentMsgId = `agent-${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, sender: 'user', text: q, timestamp: timeStr },
    ]);
    setInputQuestion('');
    setIsLoading(true);

    try {
      const resp = await queryDataset(profile.metadata.id, q, apiKey);
      setMessages((prev) => [
        ...prev,
        {
          id: agentMsgId,
          sender: 'agent',
          response: resp,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } catch (err: any) {
      const isUnconfigured = !apiKey;
      setMessages((prev) => [
        ...prev,
        {
          id: agentMsgId,
          sender: 'agent',
          text: isUnconfigured
            ? '⚠️ AI model not configured. Add GEMINI_API_KEY to backend/.env'
            : `⚠️ I encountered an issue analyzing that question: ${err.message || 'Unknown error'}. Please try rephrasing or select one of the suggested prompts below.`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col space-y-4 max-w-5xl mx-auto">
      {/* Unconfigured LLM Status Notice */}
      {!apiKey && (
        <div className="bg-[#FFF7ED] border border-[#FED7AA] rounded-2xl p-4 flex items-center justify-between gap-3 text-sm text-[#9A3412]">
          <div className="flex items-center gap-2.5">
            <AlertCircle className="w-5 h-5 text-[#EA580C] shrink-0" />
            <span className="font-semibold">AI model not configured. Add GEMINI_API_KEY to backend/.env</span>
          </div>
        </div>
      )}

      {/* Recommended Beginner Questions Row */}
      <div className="bg-white p-4 rounded-2xl border border-[#EAE6DF] shadow-xs">
        <div className="flex items-center gap-2 text-sm font-bold text-[#0F3D3E] mb-3">
          <Sparkles className="w-4 h-4 text-[#EA580C]" />
          <span>Recommended Questions:</span>
        </div>
        <div className="flex flex-wrap gap-2.5">
          {profile.suggested_questions.map((prompt, idx) => (
            <button
              key={idx}
              onClick={() => handleSend(prompt)}
              disabled={isLoading}
              className="text-sm bg-[#FAF8F5] hover:bg-[#FFF7ED] text-[#0F3D3E] hover:text-[#EA580C] px-4 py-2 rounded-full border border-[#EAE6DF] transition-all flex items-center gap-2 shadow-2xs cursor-pointer disabled:opacity-50 font-medium"
            >
              <span>{prompt}</span>
              <ArrowRight className="w-3.5 h-3.5 text-[#EA580C]/70" />
            </button>
          ))}
        </div>
      </div>

      {/* Messages Stream */}
      <div className="space-y-4 min-h-[320px]">
        {messages.map((msg) => {
          if (msg.sender === 'user') {
            return (
              <div key={msg.id} className="flex justify-end">
                <div className="max-w-xl bg-[#0F3D3E] text-white px-5 py-3 rounded-2xl rounded-tr-xs shadow-xs space-y-1">
                  <div className="text-base">{msg.text}</div>
                  <div className="text-xs text-white/60 text-right">{msg.timestamp}</div>
                </div>
              </div>
            );
          }

          // Agent Message
          const resp = msg.response;
          const isCodeExpanded = expandedCodeIds.has(msg.id);

          return (
            <div key={msg.id} className="flex items-start gap-3.5">
              <div className="w-9 h-9 rounded-full bg-[#EA580C] text-white flex items-center justify-center shrink-0 shadow-2xs mt-1">
                <Sparkles className="w-5 h-5 fill-current/20" />
              </div>

              <div className="flex-1 bg-white rounded-2xl border border-[#EAE6DF] shadow-xs p-6 space-y-4">
                {/* Header status bar */}
                {resp && (
                  <div className="flex items-center justify-between text-sm pb-3 border-b border-[#F0EDE6]">
                    <div className="flex items-center gap-1.5 text-[#047857] font-semibold bg-[#ECFDF5] px-3 py-1 rounded-full border border-[#A7F3D0]">
                      <CheckCircle2 className="w-4 h-4 text-[#047857]" />
                      <span>Grounded Pandas Result</span>
                    </div>
                    <div className="flex items-center gap-1.5 text-[#648484] text-xs">
                      <Clock className="w-3.5 h-3.5" />
                      <span>{resp.execution_time_ms} ms</span>
                    </div>
                  </div>
                )}

                {/* Plain English explanation with bold numbers */}
                <div className="text-base text-[#0F3D3E] leading-relaxed">
                  {resp ? (
                    <div
                      dangerouslySetInnerHTML={{
                        __html: resp.answer_text
                          .replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold text-[#EA580C] bg-[#FFF7ED] px-1.5 py-0.5 rounded">$1</strong>')
                          .replace(/`(.*?)`/g, '<code class="bg-[#F5F2EB] text-[#0F3D3E] px-1.5 py-0.5 rounded text-sm font-mono">$1</code>'),
                      }}
                    />
                  ) : (
                    <div
                      dangerouslySetInnerHTML={{
                        __html: (msg.text || '')
                          .replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold text-[#EA580C]">$1</strong>')
                          .replace(/`(.*?)`/g, '<code class="bg-[#F5F2EB] px-1.5 py-0.5 rounded text-sm font-mono">$1</code>'),
                      }}
                    />
                  )}
                </div>

                {/* Primary Metric Figures Strip */}
                {resp && Object.keys(resp.summary_figures).length > 0 && (
                  <div className="flex flex-wrap gap-2.5 pt-1">
                    {Object.entries(resp.summary_figures).map(([k, v], i) => (
                      <div
                        key={i}
                        className="bg-[#FAF8F5] border border-[#EAE6DF] px-3.5 py-2 rounded-xl text-sm flex items-center gap-2"
                      >
                        <span className="text-[#476D6D] font-medium capitalize">{k.replace(/_/g, ' ')}:</span>
                        <strong className="text-[#0F3D3E] font-bold">{String(v)}</strong>
                      </div>
                    ))}
                  </div>
                )}

                {/* Interactive Chart (Plotly) */}
                {resp && resp.chart_spec && (
                  <div className="bg-[#FAF8F5] p-3 rounded-2xl border border-[#EAE6DF]">
                    <PlotlyChart spec={resp.chart_spec} plotlySpec={resp.chart_spec.plotly_spec} height={300} />
                  </div>
                )}

                {/* Table Preview (if multiple rows returned) */}
                {resp && resp.table_result && resp.table_result.length > 0 && (
                  <div className="bg-[#FAF8F5] rounded-xl border border-[#EAE6DF] overflow-hidden">
                    <div className="px-4 py-2.5 bg-[#F5F2EB] border-b border-[#EAE6DF] text-sm font-semibold text-[#0F3D3E] flex items-center gap-2">
                      <TableIcon className="w-4 h-4 text-[#EA580C]" />
                      <span>Result Table ({resp.table_result.length} rows)</span>
                    </div>
                    <div className="overflow-x-auto max-h-60">
                      <table className="w-full text-sm text-left">
                        <thead>
                          <tr className="bg-[#FAF8F5] text-[#476D6D] border-b border-[#EAE6DF]">
                            {(resp.table_columns || Object.keys(resp.table_result[0])).map((col, i) => (
                              <th key={i} className="px-3.5 py-2.5 font-semibold">{col}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#EAE6DF]">
                          {resp.table_result.slice(0, 15).map((row, rIdx) => (
                            <tr key={rIdx} className="hover:bg-white transition-colors">
                              {(resp.table_columns || Object.keys(row)).map((col, cIdx) => (
                                <td key={cIdx} className="px-3.5 py-2 text-[#0F3D3E]">
                                  {row[col] !== null ? String(row[col]) : '-'}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Verified Python Code Accordion Drawer */}
                {resp && resp.code_executed && (
                  <div className="border border-[#EAE6DF] rounded-xl overflow-hidden bg-[#FAF8F5]">
                    <button
                      onClick={() => toggleCode(msg.id)}
                      className="w-full px-4 py-2.5 text-sm font-medium text-[#476D6D] flex items-center justify-between hover:bg-[#F5F2EB] transition-colors cursor-pointer"
                    >
                      <div className="flex items-center gap-2 text-[#0F3D3E]">
                        <Code2 className="w-4 h-4 text-[#EA580C]" />
                        <span className="font-semibold">Python Execution Code</span>
                      </div>
                      <div className="flex items-center gap-1 text-xs text-[#648484]">
                        {isCodeExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      </div>
                    </button>

                    {isCodeExpanded && (
                      <div className="p-3.5 bg-white border-t border-[#EAE6DF] space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-[#648484] font-mono">AST-Validated Sandbox</span>
                          <button
                            onClick={() => copyCode(msg.id, resp.code_executed)}
                            className="text-xs text-[#0F3D3E] hover:text-[#EA580C] flex items-center gap-1.5 bg-[#FAF8F5] px-2.5 py-1.5 rounded-lg border border-[#EAE6DF] cursor-pointer font-medium"
                          >
                            {copiedId === msg.id ? (
                              <>
                                <Check className="w-3.5 h-3.5 text-[#047857]" />
                                <span>Copied!</span>
                              </>
                            ) : (
                              <>
                                <Copy className="w-3.5 h-3.5" />
                                <span>Copy Code</span>
                              </>
                            )}
                          </button>
                        </div>
                        <pre className="p-3 bg-[#FAF8F5] rounded-xl text-xs font-mono text-[#0F3D3E] overflow-x-auto border border-[#EAE6DF]">
                          <code>{resp.code_executed}</code>
                        </pre>
                      </div>
                    )}
                  </div>
                )}

                {/* Dynamic Follow-Up Suggestions */}
                {resp && resp.follow_up_suggestions && resp.follow_up_suggestions.length > 0 && (
                  <div className="pt-2 border-t border-[#F0EDE6] space-y-2">
                    <span className="text-xs font-bold text-[#648484] uppercase tracking-wider">
                      Follow-Up Questions:
                    </span>
                    <div className="flex flex-wrap gap-2">
                      {resp.follow_up_suggestions.map((sug, sIdx) => (
                        <button
                          key={sIdx}
                          onClick={() => handleSend(sug)}
                          disabled={isLoading}
                          className="text-sm bg-[#FAF8F5] hover:bg-[#FFF7ED] text-[#0F3D3E] hover:text-[#EA580C] px-3.5 py-1.5 rounded-full border border-[#EAE6DF] transition-all flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                        >
                          <span>{sug}</span>
                          <ArrowRight className="w-3.5 h-3.5 text-[#EA580C]/70" />
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* Loading Spinner State */}
        {isLoading && (
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-[#EA580C] text-white flex items-center justify-center shrink-0 animate-pulse">
              <Sparkles className="w-4 h-4 fill-current/20" />
            </div>
            <div className="bg-white rounded-2xl border border-[#EAE6DF] px-4 py-3 text-sm text-[#476D6D] flex items-center gap-2 shadow-xs">
              <div className="w-4 h-4 rounded-full border-2 border-[#EA580C] border-t-transparent animate-spin" />
              <span>Analyzing dataset with sandboxed Pandas operations...</span>
            </div>
          </div>
        )}
      </div>

      {/* Input Box Footer */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSend(inputQuestion);
        }}
        className="sticky bottom-4 bg-white p-2.5 rounded-2xl border border-[#EAE6DF] shadow-md flex items-center gap-3"
      >
        <input
          type="text"
          value={inputQuestion}
          onChange={(e) => setInputQuestion(e.target.value)}
          placeholder={`Ask about ${profile.metadata.name}... (e.g. "Which cuisine has the highest rating?")`}
          disabled={isLoading}
          className="flex-1 bg-transparent px-4 py-2.5 text-base text-[#0F3D3E] placeholder-[#648484] focus:outline-none"
        />
        <button
          type="submit"
          disabled={!inputQuestion.trim() || isLoading}
          className="bg-[#EA580C] hover:bg-[#C2410C] text-white px-6 py-3 rounded-xl font-semibold text-base flex items-center gap-2 transition-all shadow-xs cursor-pointer disabled:opacity-40"
        >
          <span>Ask Agent</span>
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
};
