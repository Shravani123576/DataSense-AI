# DataSense AI — Graph AI Analyst Agent Architecture

**System:** Graph AI Analyst Contextual Agent  
**Framework:** Pydantic AI (Google Gemini Provider) + Validated Tool Registry + Deterministic Analytical Fallback  
**Security & Integrity:** Zero numerical hallucination, execution tracing, and observational causal boundaries.

---

## 1. Architectural Overview & Workflow Diagram

```
                 User Interaction (Explore Data Tab)
                                │
                        handleAskGraphAi()
        [dataset_id, dataset_version, chart_signature, conversation_id]
                                │
                    POST /api/graph/ask
                                │
                FastAPI Route (Thin Service Layer)
                                │
                 Analyst Agent Orchestration Loop
                                │
             ┌──────────────────┴──────────────────┐
     GEMINI_API_KEY Available?              No / Offline / Quota / Timeout
             │                                     │
             ▼                                     ▼
     [LLM Agent Mode]                      [Deterministic Fallback Mode]
   Pydantic AI Planner                     Rule-Based Intent Router
   (Bounded: max 6 tool steps)             (Intent -> Analytical Pipeline)
             │                                     │
             └──────────────────┬──────────────────┘
                                │
                   Validated Tool Registry
            (Single Source of Truth for In-App & MCP)
                                │
  ┌─────────────────────────────┼─────────────────────────────┐
  │                             │                             │
analyze_chart          analyze_time_series           run_statistical_test
compare_groups         compute_rate_of_change        investigate_anomalies
get_dataset_schema     compute_period_comparison     inspect_data_quality
  │                             │                             │
  └─────────────────────────────┬─────────────────────────────┘
                                │
                    Structured Execution Trace
               {step, tool, input, output_summary, duration_ms, error}
                                │
                        Grounding Check
        (Verifies all numbers in answer match trace outputs)
                                │
                    Structured Response Contract
    {answer: {finding, evidence[], method, limitations[], next_step},
     trace: [...], mode: "llm" | "fallback", grounded: bool, warnings: []}
                                │
                      VisualizerStudio UI
    (Five-part answer • Collapsed "Tools used (N)" • Small mode badge)
```

---

## 2. Core Agent Pillars

### A. Tool Registry as the Single Source of Truth
Every analytical function is declared once as a `BaseTool` subclass with:
1. `name`: Unique canonical string (e.g., `compute_rate_of_change`).
2. `input_schema`: Strongly-typed Pydantic model.
3. `output_schema`: Structured Pydantic result model.
4. `timeout_seconds`: Strict execution timeout (default 30.0s) enforced via thread pools.
5. `category` and `permission`: Access classification (`READ_ONLY`, `MUTATES_DATA`).

Both the in-process FastAPI application and the external Model Context Protocol (`mcp_server.py`) invoke tools exclusively through `default_registry.execute_tool()`. No analytical calculation is duplicated.

### B. Planning, Adaptation & Bounded Execution Loop
1. **Understand:** Resolves question intent (trend, rate, comparison, significance, anomaly, seasonality) and merges multi-turn history.
2. **Context Binding:** Evaluates `chart_signature` against `default_conversation_store`. If signature has changed, previous results are invalidated.
3. **Act & Observe:** Dispatches allowlisted tool calls against session dataframe context. Summarized metrics are returned to the agent context (raw dataframes are never serialized into LLM prompts).
4. **Adapt:** Executes optional secondary tool calls (e.g., statistical hypothesis testing following group comparison) up to a strict bound of 6 tool steps.
5. **Trace Generation:** Appends each step with duration, parameters, and result summary to `AgentTracer`.

### C. Grounding Check & Causal Boundaries
Before returning the final payload:
1. The Grounding Engine parses all numerical assertions (slopes, counts, percentages, p-values) from `finding` and `evidence`.
2. Numbers are verified against the collected tool outputs in `trace` and session dimensions.
3. If an ungrounded or fabricated number is detected, `grounded` is flagged as `False`, a warning is logged, and the violation is recorded.
4. Causal disclaimer: Observational associations cannot prove causation without experimental intervention.

### D. Deterministic Analytical Fallback
When offline, without API keys, or upon provider timeout/quota exhaustion, the router selects the exact registered tools matching the question intent, runs the same calculations, and generates the identical 5-part contract with `mode: "fallback"`.

---

## 3. Structured Response Contract

```json
{
  "answer": {
    "finding": "Primary empirical observation directly supported by the data.",
    "evidence": ["Calculated bullet points with exact numbers, slopes, or test stats."],
    "method": "Exact mathematical method or test applied with sample size N.",
    "limitations": ["Data caveats, missing value exclusions, and causal disclaimer."],
    "next_step": "Suggested follow-up analytical investigation."
  },
  "trace": [
    {
      "step": 1,
      "tool": "compute_rate_of_change",
      "input": {"time_column": "Date", "value_column": "Sales"},
      "output_summary": "Slope = +2.0000/day, Rate/Year = +730.50, p = 0.0001",
      "duration_ms": 14.2,
      "error": null
    }
  ],
  "mode": "llm | fallback",
  "grounded": true,
  "warnings": []
}
```
