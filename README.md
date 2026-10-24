# DataSense AI V2 — Intelligent Data Science Studio

> **Beginner-Friendly AI Data Science Assistant with 100% Grounded Pandas Execution & Interactive Plotly Visualizations**

DataSense AI V2 allows users to upload any tabular CSV or Excel dataset (scaling past **100,000+ rows**) or explore pre-configured sample datasets. It automatically profiles data quality and anomalies, builds interactive Plotly visualizations, and answers natural language questions in clear, accessible English with 100% mathematically grounded Pandas execution.

---

## The 4-Step Beginner-Friendly Flow

1. **Step 1: Upload Dataset**: Direct drag-and-drop dropzone supporting CSV and Excel spreadsheets up to 100,000+ rows, or explore sample datasets with 1 click.
2. **Step 2: Review Data**: Instant data quality score (0–100 & Grade A+/A/B/C), 4 pillars (Completeness, Uniqueness, Outlier Health, Consistency), with expandable accordions for the raw paginated data table and statistical feature dictionary.
3. **Step 3: Ask Questions**: Conversational AI assistant with context-aware welcome message and tailored starter prompts. Translates plain English into AST-sandboxed Pandas code.
4. **Step 4: View Results**: Verified statistics with numbers highlighted in bold, summary KPI cards, interactive Plotly charts, and copyable Python code drawers.

---

## Key V2 Upgrades & Analytical Accuracy

- **Zero Zomato Auto-Loading**: Starts in a clean, neutral onboarding state. Zomato is only loaded or mentioned if the user explicitly selects it as a sample dataset.
- **State Isolation & Clean Reset**: Switching datasets or uploading a new file immediately resets all chat history and cached state.
- **Honest Limitation & Ambiguity Handling**: If a user asks about a column or concept not present in the dataset (e.g. asking for "revenue" on Titanic), DataSense AI explains that the metric was not found and lists available columns rather than hallucinating an unrelated column.
- **Universal Numeric Coercion**: Disguised string numbers (currency `$`, `₹`, `€`, `%`, commas `"1,200"`, fractional ratings `"4.2/5"`) are automatically coerced to true numbers for accurate mathematical operations.
- **Rankings & Chronological Trends**: Supports both Top N and Bottom/Lowest N rankings, period-over-period percentage changes with zero-division guards, and Plotly line charts.

4. **Automated Data Profiler & Quality Radar**
   - Overall Quality Index (0–100 score with A+/A/B/C letter grades).
   - 4 Health Pillars: Completeness, Uniqueness, Outlier Health, and Type Consistency.
   - Automated detection of missing cells, duplicates, and 1.5x IQR outliers.
   - Interactive Pearson Correlation Heatmap with hover tooltips and relationship badges.

5. **Plotly Visualizations Studio**
   - Interactive charts built on Plotly: Bar, Line, Scatter, Histogram, and Donut/Pie charts.
   - Aggregations: Mean, Sum, Count, Min, Max.
   - Auto-discovered pattern presets generated upon dataset ingestion.
   - Export charts with hover inspect, pan, zoom, and high-res image download.

6. **Interactive Data Grid & Feature Dictionary**
   - Searchable, paginated table with semantic type tags (`numeric`, `categorical`, `datetime`, `boolean`).
   - Feature Dictionary with beginner-friendly tooltips explaining statistical concepts (Mean, Median, Std Dev, Skewness, IQR).

7. **Downloadable Reports & CSV Export**
   - **Executive Data Profile Report (CSV)**: Export column-level health, completeness, missing percentages, and summary statistics.
   - **Full Cleaned Raw Dataset (CSV)**: One-click raw dataset export.
   - **Executive Health Audit (Markdown)**: Complete data dictionary and correlation highlights.

---

## Design System

Inspired by clean, minimalist design principles:
- **Warm off-white background**: `#FAF8F5` and `#FFFFFF` rounded cards.
- **Dark teal typography**: `#0F3D3E` and `#1E5152` for crisp, high-contrast readability.
- **Terracotta orange accents**: `#EA580C` for primary action buttons, key metrics, and callouts.
- **Spacious layout**: Generous padding, rounded borders (`rounded-2xl`), subtle card elevation, and friendly icons.

---

## Architecture & Security Guardrails

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS v4, Plotly.js (`plotly.js-dist-min`), Lucide icons.
- **Backend**: FastAPI, Uvicorn, Python 3.12, pandas 3.0, numpy, openpyxl, google-generativeai, python-dotenv.
- **AST Security Sandbox**: User queries and generated Python scripts are parsed with Python's Abstract Syntax Tree (`ast`). Restricted names (`open`, `os`, `sys`, `subprocess`, `socket`, `eval`, `exec`) and import statements are strictly blocked.
- **Session-Only Data**: Session-based in-memory processing without third-party tracking, persistent account logins, or persistent chat storage.
- **API Key Privacy**: Optional Google Gemini API key is stored in browser session storage and transmitted solely to the local backend; keys are never exposed.

---

## Quickstart

### 1. Start Backend Server
```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 2. Start Frontend Dev Studio
```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 3000
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### 3. Run Backend Test Suite
```bash
cd backend
pytest test_backend.py -v
```
