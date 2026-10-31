# DataSense AI — Graph AI Analyst Diagnosis Report

**Date:** 2026-10-26  
**Component:** Graph AI Analyst (`VisualizerStudio.tsx` <-> `/api/graph/ask` <-> `analyst_agent.py`)

---

## Executive Summary

The Graph AI Analyst was observed to become unresponsive or freeze in the Explore Data tab under multiple runtime conditions. An end-to-end trace from UI user interaction through network fetch, routing, agent execution, and UI rendering identified seven root causes spanning missing payload identifiers, unhandled React child object rendering, IPv6 localhost dev proxy conflicts, missing conversation persistence, and schema format discrepancies between backend and frontend.

---

## 1. Request Payload Incompleteness (Dataset ID, Version, Signature, Conversation ID)

### Diagnosis:
- **File Reference:** `frontend/src/components/VisualizerStudio.tsx` (Lines 201–214), `backend/models.py` (Lines 492–504)
- **Findings:**
  1. **Conversation ID:** The frontend previously did not generate or transmit a `conversation_id`. The backend `GraphIntelligenceAskRequest` had no `conversation_id` field. Without a persistent conversation identifier, multi-turn follow-ups could not be anchored or stored in an isolated session store.
  2. **Chart Signature:** The frontend computed an `activeChartSignature` (`${chart_type}:${x_axis}:${y_axis}:${aggregation}`), but never passed it to the backend. The backend could not detect when a client query referenced a changed chart layout.
  3. **Dataset Version:** The frontend passed `(metadata as any).version ?? 0`. However, `DatasetMetadata` on `ProfileReport` does not contain a `version` field; dataset versions are tracked in the cleaning lineage (`DATASET_VERSIONS_STORAGE`). As a result, requests were hardcoded to `version=0`, failing version invalidation checks when users cleaned data to `v1` or `v2`.
  4. **Column Case Mismatch:** If a chart specification passed lowercase column names while the underlying Pandas dataframe had TitleCase columns (e.g. `contract` vs `Contract`), `analyst_agent.py` failed with `Analysis Error: Plotted column 'contract' does not exist in dataset`.

### Root Cause Fix:
- Add `conversation_id: Optional[str]`, `chart_signature: Optional[str]`, and `dataset_version: Optional[int]` to `GraphIntelligenceAskRequest` (`backend/models.py`).
- Implement case-insensitive column resolution in the backend tool layer.
- Populate `conversation_id` and pass active lineage version from frontend state.

---

## 2. Route Exceptions & Silent Failure in UI

### Diagnosis:
- **File Reference:** `frontend/src/components/VisualizerStudio.tsx` (Lines 215–235, 575–605), `backend/main.py` (Line 1481–1492)
- **Findings:**
  - In `backend/analyst_agent.py`, the fallback engine produced an answer object. However, if any sub-calculation encountered unexpected datatypes (such as `np.bool_` or `NoneType` in formatting strings), the error handler caught the exception but returned a generic text or 500 error.
  - Crucially, in `VisualizerStudio.tsx` (Line 218 & 580), the code did:
    ```tsx
    content: resp.answer
    ...
    <div className="whitespace-pre-line text-xs">{msg.content}</div>
    ```
    When `resp.answer` was returned as a structured object `{finding, evidence, method, limitations, next_step}` per the specification contract instead of a plain string, React threw an unhandled runtime error:
    `Uncaught Error: Objects are not valid as a React child (found: object with keys {finding, evidence, method, limitations, next_step})`.
    This runtime exception crashed the React component tree and froze the chat drawer permanently in an unresponsive state.

### Root Cause Fix:
- Update `VisualizerStudio.tsx` to handle the structured `{finding, evidence, method, limitations, next_step}` response contract natively with fallback to string rendering if legacy text is passed.
- Ensure error states in `handleAskGraphAi` catch parsing/rendering errors and surface a clear error row with a retry button.

---

## 3. Gemini API Key & Fallback Mode Handling

### Diagnosis:
- **File Reference:** `backend/analyst_agent.py` (Lines 711–727, 814–819)
- **Findings:**
  - When `GEMINI_API_KEY` was missing from the environment, the code correctly entered `run_deterministic_analyst_fallback`.
  - However, if `GEMINI_API_KEY` was set to an invalid, expired, or non-functional key, `agent.run_sync` raised authentication or quota exceptions. The catch block logged the error and called fallback, but the returned structure differed from what the LLM path produced, causing inconsistent response keys.
  - The fallback engine did not include a structured execution trace table or step durations, so the UI could not render tool execution steps.

### Root Cause Fix:
- Implement a unified agent response pipeline that records structured traces (`AgentTraceStep`: `{step, tool, input, output_summary, duration_ms, error}`) for BOTH the LLM mode and the deterministic fallback mode.
- Explicitly return `mode: "llm"` or `mode: "fallback"` along with `grounded: bool` and `warnings: []`.

---

## 4. Response Contract Shape Discrepancy

### Diagnosis:
- **File Reference:** `backend/models.py` (Line 506–517), `frontend/src/types.ts` (Line 480–495)
- **Findings:**
  - The backend `GraphIntelligenceAskResponse` previously returned:
    `{ answer: str, findings: List[...], computed_evidence: Dict[...], methodology: str, limitations: List[...], suggested_next_step: str, ... }`
  - The target specification contract requires:
    ```json
    {
      "answer": {
        "finding": "...",
        "evidence": ["..."],
        "method": "...",
        "limitations": ["..."],
        "next_step": "..."
      },
      "trace": [
        {
          "step": 1,
          "tool": "...",
          "input": {},
          "output_summary": "...",
          "duration_ms": 12.3,
          "error": null
        }
      ],
      "mode": "llm" | "fallback",
      "grounded": true,
      "warnings": []
    }
    ```
  - Discrepancy between `answer: str` vs `answer: { finding, evidence, method, limitations, next_step }` and `tools_invoked: List[str]` vs `trace: List[AgentTraceStep]` caused typing discrepancies and potential UI breaks.

### Root Cause Fix:
- Standardize the response model across backend and frontend to the exact contract.
- Maintain backward-compatible accessors so any existing tests that read `resp["answer"]` or `resp.answer` continue to function without regression.

---

## 5. Loading and Streaming State Lifecycle

### Diagnosis:
- **File Reference:** `frontend/src/components/VisualizerStudio.tsx` (Lines 198, 235–237)
- **Findings:**
  - `isAskingAi` was set to `true` at the start of `handleAskGraphAi` and set to `false` in `finally`.
  - However, if the user triggered another suggested question while `isAskingAi` was true, the previous request continued in-flight without abort signal, leading to race conditions where stale promises overwrote the latest chat message.
  - Furthermore, if an error occurred in React rendering of the returned message, React threw an error before `setIsAskingAi(false)` took visual effect.
  - No retry button was provided on error.

### Root Cause Fix:
- Add error boundary protection in message rendering.
- Add an explicit `[ Retry ]` button on failed messages that re-executes the last query.
- Always clear the `isAskingAi` loading indicator on success and error.

---

## 6. Stale Chart Signature Blocking & Invalidation

### Diagnosis:
- **File Reference:** `frontend/src/components/VisualizerStudio.tsx` (Lines 172–180, 544–558)
- **Findings:**
  - In `VisualizerStudio.tsx`:
    ```ts
    if (!activeChartSignature) {
      setActiveChartSignature(sig);
    }
    ```
  - `activeChartSignature` was initialized on the first chart, but was never updated when the user legitimately switched features or selected a preset chart.
  - Consequently, once a user changed the X-axis or chart type, the alert banner *"Active graph updated to... Reset Thread"* appeared permanently, and follow-ups could reuse stale parameters unless the user manually clicked "Reset Thread".
  - The backend did not enforce chart signature checks, meaning if a client sent a question about a bar chart while passing an old time-series context, the backend attempted time-series calculations on categorical data.

### Root Cause Fix:
- Pass `chart_signature` with each request.
- When `chart_signature` changes in the backend for an existing `conversation_id`, the backend agent notes the context switch, refuses to reuse prior calculations, and initiates a clean analytical trace for the new chart.
- On the frontend, update `activeChartSignature` when a new investigation begins or when the user confirms the chart switch.

---

## 7. Vite Development Proxy & Network Binding (IPv4 vs IPv6)

### Diagnosis:
- **File Reference:** `frontend/vite.config.ts` (Lines 13–18)
- **Findings:**
  - `vite.config.ts` configured the proxy as:
    ```ts
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
    ```
  - On Windows with Node.js 18+, `localhost` resolves preferentially to IPv6 `::1`.
  - When Uvicorn is launched with `python -m uvicorn main:app --host 127.0.0.1 --port 8000`, it binds strictly to IPv4 `127.0.0.1`.
  - In certain environments, Vite's proxy fails to connect to `::1:8000`, returning `504 Gateway Timeout` or `ECONNREFUSED` without reaching FastAPI.

### Root Cause Fix:
- Change the proxy target in `frontend/vite.config.ts` from `http://localhost:8000` to `http://127.0.0.1:8000` to eliminate IPv6 resolution failures.

---

## 8. Detailed Trace & Empty Section Root Cause (GLB.Ts+dSST.csv)

### End-to-End Execution Trace Path
1. **Chat Input & Trigger:**
   - **File:** `frontend/src/components/VisualizerStudio.tsx` (`handleAskGraphAi`, Lines 192–215)
   - Captures user question `q`, packages chart specification `currentSpec`, calculates `currentSig`, and appends user message to `chatMessages`.
2. **Frontend API Call:**
   - **File:** `frontend/src/api.ts` (`askGraphAi`, Lines 180–205)
   - Sends `POST http://127.0.0.1:8000/api/graph/ask` with JSON payload:
     `{ dataset_id, chart_spec, x_column, y_column, chart_type, aggregation, question, conversation_history, chart_signature, dataset_version }`.
3. **Backend Route Ingestion:**
   - **File:** `backend/main.py` (`ask_graph_analyst`, Lines 1482–1510)
   - Validates `dataset_id` in `DATASET_STORAGE`, converts request to `GraphIntelligenceAskRequest`, and invokes `ask_graph_analyst` in `analyst_agent.py`.
4. **Analyst Agent & Dispatch:**
   - **File:** `backend/analyst_agent.py` (`ask_graph_analyst` and `run_deterministic_analyst_fallback`, Lines 455–850)
   - Resolves columns from dataframe (including MultiIndex level 0 `Year` support for GLB), executes deterministic tools via tool registry `default_registry.execute_tool(...)`.
5. **Tool Execution:**
   - **File:** `backend/tools/analysis_tools.py`:
     - `AnalyzeTimeSeriesTool` (Lines 865–945): Computes linear slope per decade (+0.0804 °C/decade), 95% confidence interval `[+0.0723, +0.0885]`, start value (-0.21 in 1880), end value (1.22 in 2025), largest year-to-year change (+0.55 °C from 2022 to 2023), and bipartite split (early mean -0.1647 vs late mean +0.3582, delta +0.5229).
     - `AnalyzeChartTool` (Lines 640–740): Computes continuous distribution metrics: N=146, mean=0.0968, median=-0.0100, std=0.3990, skewness=1.1166, min=-0.52 in 1912, max=1.41 in 2023, top and bottom anomaly years.
6. **Grounding Verification & Response Formatting:**
   - **File:** `backend/analyst_agent.py` (`verify_grounding`, Lines 410–445)
   - Verifies all numeric figures in narrative exist in trace records. Formats `StructuredAnalystAnswer` with `finding`, `evidence`, `method`, `limitations`, and `next_step`.
7. **Frontend Rendering:**
   - **File:** `frontend/src/components/VisualizerStudio.tsx` (Lines 700–760)
   - Renders 4 structured sections. If any section is missing or empty, renders `"Not computed: <reason>"`.

### Root Cause of Empty Section Bodies
1. **Key Mismatch & Template Stubbing:** In legacy fallback responses, `methodology` and `limitations` were sometimes returned at root level of response rather than inside `StructuredAnalystAnswer` (`answer.method`, `answer.limitations`), causing the frontend to see empty arrays.
2. **Missing Fallback Rendering:** The UI previously had `{msg.structuredAnswer.evidence && msg.structuredAnswer.evidence.length > 0 && (...)}`. If the list was empty, no body was rendered, leaving headers without content.
3. **Chart Invalidation Failure:** When switching charts, `activeChartSignature` was never updated after initial mount, leaving stale context banners permanently active.

### Exact Backend Response JSON (Example: "What is the main trend?")
```json
{
  "answer": {
    "finding": "Primary Time-Series Trend: Land-Ocean: Global Means exhibits a statistically significant upward linear trend advancing at +0.0804 °C per decade (95% CI: [+0.0723, +0.0885], p < 0.0001) across 146 years analyzed (1880 to 2025). The series progressed from -0.21 °C in 1880 to 1.22 °C in 2025, with the largest single year-to-year change of +0.55 °C occurring between 2022 and 2023. Note: This trend describes empirical historical trajectory across the observed period and does not independently prove an isolated causal mechanism.",
    "evidence": [
      "Decadal Linear Slope: +0.0804 °C per decade (95% CI: [+0.0723, +0.0885])",
      "Observations Analyzed: N = 146 years across temporal span 1880 to 2025",
      "Boundary Values: -0.21 °C (1880) to 1.22 °C (2025)",
      "Maximum Annual Delta: +0.55 °C from 2022 to 2023",
      "Statistical Significance: Ordinary Least Squares fit p-value < 0.0001 (statistically significant)"
    ],
    "method": "Ordinary Least Squares (OLS) linear trend regression over temporal horizon with 95% confidence intervals and rate-of-change delta analysis.",
    "limitations": [
      "Linear regression assumes steady decadal trajectory; decadal acceleration or volcanic/ENSO cyclical shifts may deviate from linearity.",
      "Observational historical series represents observed associations and cannot establish causal drivers in isolation.",
      "146 annual observations evaluated; 2 boundary values with missing records were excluded."
    ],
    "next_step": "Examine moving averages or decompose cyclical anomaly periods."
  },
  "trace": [
    {
      "step_number": 1,
      "tool_name": "analyze_time_series",
      "tool_input": {
        "dataset_id": "test_glb",
        "time_column": "Year",
        "value_column": "Land-Ocean: Global Means"
      },
      "output_summary": "SlopeDecade=+0.0804/decade, CI=[+0.0723, +0.0885], YearsUsed=146, Span=1880-2025, StartVal=-0.21, EndVal=+1.22, LargestChange=+0.55 (2022-2023), Direction=upward, p=0.0001.",
      "duration_ms": 14.2,
      "error": null
    }
  ],
  "mode": "fallback",
  "grounded": true,
  "warnings": []
}
```
