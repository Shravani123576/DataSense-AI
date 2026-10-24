import os
import re
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from analysis_engine import execute_sandboxed_pandas, create_plotly_spec
from models import QueryResponse, ChartSpec

logger = logging.getLogger("datasense.llm")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import google.generativeai as genai
    HAS_GEMINI_PKG = True
except ImportError:
    HAS_GEMINI_PKG = False


SYNONYM_MAP = {
    "sales": ["sale", "revenue", "sales_amount", "total_sales", "turnover"],
    "revenue": ["revenue", "sales", "sales_amount", "turnover", "total_revenue"],
    "profit": ["profit", "net_profit", "margin", "earnings", "net_income"],
    "cost": ["cost", "price", "expense", "cost_for_two", "unit_price", "charges", "fare"],
    "price": ["price", "cost", "unit_price", "fare", "charges", "listing_price"],
    "rating": ["rating", "rate", "stars", "score", "overall_rating"],
    "discount": ["discount", "rebate", "discount_pct"],
    "quantity": ["quantity", "volume", "units", "count", "qty"],
    "age": ["age", "years", "tenure"],
    "churn": ["churn", "attrition", "cancelled"],
    "return": ["returned", "returns", "refund", "return_status"],
    "category": ["category", "sub_category", "segment", "department", "type"],
    "region": ["region", "location", "city", "country", "state", "area", "locality"],
    "date": ["order_date", "date", "created_at", "timestamp", "year", "month"]
}


def extract_columns_from_df(df: pd.DataFrame) -> Dict[str, str]:
    """Return map of column name to simplified semantic data type."""
    col_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_types[col] = "datetime"
        elif pd.api.types.is_bool_dtype(df[col]):
            col_types[col] = "boolean"
        else:
            col_types[col] = "categorical"
    return col_types


def match_column_fuzzy(token: str, available_cols: List[str]) -> Optional[str]:
    """Find the best column match for a token or phrase in the user's query."""
    t_lower = token.lower().strip()
    clean_t = re.sub(r"[^a-z0-9]", "", t_lower)
    if not clean_t:
        return None

    # 1. Exact match
    for col in available_cols:
        if col.lower() == t_lower:
            return col

    # 2. Clean alphanumeric match (ignoring spaces/underscores)
    for col in available_cols:
        clean_c = re.sub(r"[^a-z0-9]", "", col.lower())
        if clean_t == clean_c:
            return col

    # 3. Substring match
    for col in available_cols:
        clean_c = re.sub(r"[^a-z0-9]", "", col.lower())
        if len(clean_t) >= 4 and (clean_t in clean_c or clean_c in clean_t):
            return col

    # 4. Synonym match
    for syn_key, syn_list in SYNONYM_MAP.items():
        if clean_t in [re.sub(r"[^a-z0-9]", "", s) for s in syn_list]:
            for col in available_cols:
                clean_c = re.sub(r"[^a-z0-9]", "", col.lower())
                if any(re.sub(r"[^a-z0-9]", "", s) in clean_c for s in syn_list):
                    return col

    return None


def detect_mentioned_columns(q: str, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Scan query to detect explicitly mentioned numeric columns, categorical columns, and date columns."""
    q_lower = q.lower()
    col_types = extract_columns_from_df(df)
    
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t in ["categorical", "boolean"]]
    date_cols = [c for c, t in col_types.items() if t == "datetime"]

    matched_num = []
    matched_cat = []
    matched_date = []

    # 1. Match multi-word or single-word column names using whole-word boundary
    for col in df.columns:
        c_clean = re.sub(r"[_]", " ", col.lower()).strip()
        pattern = r"\b" + re.escape(c_clean) + r"\b"
        if re.search(pattern, q_lower):
            c_type = col_types.get(col, "categorical")
            if c_type == "numeric" and col not in matched_num:
                matched_num.append(col)
            elif c_type == "datetime" and col not in matched_date:
                matched_date.append(col)
            elif c_type in ["categorical", "boolean"] and col not in matched_cat:
                matched_cat.append(col)

    # 2. Check synonyms for un-matched tokens using whole-word matching (excluding stop words)
    stop_words = {"what", "which", "show", "give", "tell", "average", "mean", "total", "count", "maximum", "minimum", "highest", "lowest", "least", "each", "every", "over", "time", "date", "find", "best"}
    words = re.findall(r"\b[a-z]{3,}\b", q_lower)
    for word in words:
        if word in stop_words:
            continue
        matched_col = match_column_fuzzy(word, list(df.columns))
        if matched_col:
            c_type = col_types.get(matched_col, "categorical")
            if c_type == "numeric" and matched_col not in matched_num:
                matched_num.append(matched_col)
            elif c_type == "datetime" and matched_col not in matched_date:
                matched_date.append(matched_col)
            elif c_type in ["categorical", "boolean"] and matched_col not in matched_cat:
                matched_cat.append(matched_col)

    return matched_num, matched_cat, matched_date


ANALYTICAL_STOPWORDS = {
    "what", "which", "how", "show", "tell", "give", "list", "display", "calculate", "find", 
    "is", "are", "were", "was", "the", "a", "an", "of", "in", "for", "with", "by", "per", 
    "across", "between", "over", "under", "than", "to", "from", "on", "at", "and", "or", "not", 
    "any", "all", "some", "every", "each", "both", "top", "bottom", "highest", "lowest", 
    "max", "maximum", "min", "minimum", "mean", "average", "avg", "median", "sum", "total", 
    "count", "rows", "columns", "features", "shape", "size", "records", "dimensions", 
    "dataset", "data", "table", "missing", "null", "empty", "nan", "duplicate", "duplicates", 
    "outlier", "outliers", "clean", "quality", "health", "trend", "trends", "change", "growth", 
    "correlation", "correlations", "relationship", "distribution", "summary", "stats", 
    "statistics", "describe", "breakdown", "frequency", "categories", "preview", "head", 
    "first", "last", "more", "less", "greater", "above", "below", "where", "filter", "group", 
    "grouped", "best", "worst", "score", "scores", "rate", "rates", "ratio", "level", "value", 
    "values", "most", "least", "there", "have", "has", "can", "could", "would", "please", 
    "help", "me", "you", "about", "this", "that", "these", "those", "does", "did", "do",
    "many", "much", "time", "date", "sample", "record", "file", "csv", "info", "overview"
}


def detect_unmatchable_concept(q: str, df: pd.DataFrame) -> Optional[str]:
    """
    Detect if the user is asking about a specific concept, metric, or entity that doesn't exist in df.
    Prevents hallucinating a substitute column.
    """
    q_lower = q.lower()
    candidate_concepts = [
        "revenue", "profit", "salary", "wage", "sales", "cost", "price", 
        "discount", "churn", "rating", "age", "quantity", "inventory", 
        "budget", "spend", "gdp", "score", "temperature", "speed", "fare",
        "weather", "ceo", "retention", "satisfaction", "stock", "margin", "tax"
    ]
    for concept in candidate_concepts:
        if re.search(r"\b" + concept + r"\b", q_lower):
            matched = match_column_fuzzy(concept, list(df.columns))
            if not matched:
                return concept

    words = re.findall(r"\b[a-z]{3,}\b", q_lower)
    for word in words:
        if word not in ANALYTICAL_STOPWORDS:
            matched = match_column_fuzzy(word, list(df.columns))
            if not matched:
                return word

    return None


def heuristic_query_engine(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    """
    Guaranteed 100% mathematically grounded data analysis engine.
    Calculates exact answers via pandas without guessing or hallucinating metrics.
    """
    q = question.lower()
    cols = list(df.columns)
    col_types = extract_columns_from_df(df)
    all_numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    all_cat_cols = [c for c, t in col_types.items() if t in ["categorical", "boolean"]]
    all_date_cols = [c for c, t in col_types.items() if t == "datetime"]

    matched_num, matched_cat, matched_date = detect_mentioned_columns(question, df)

    # 1. Missing values, Quality & Duplicates
    if any(k in q for k in ["missing", "null", "empty", "quality", "clean", "nan", "duplicate", "duplicates"]):
        if any(k in q for k in ["duplicate", "duplicates"]):
            dup_count = int(df.duplicated().sum())
            code = "df.duplicated().sum()"
            exec_res = execute_sandboxed_pandas(df, code)
            if dup_count == 0:
                ans = f"Your dataset has **0 duplicate rows**. Every single one of the **{len(df):,} records** is unique."
            else:
                ans = f"We detected **{dup_count} duplicate row(s)** ({round(dup_count / len(df) * 100, 2)}% of records). Removing them can prevent inflated aggregations."
            return {
                "code": code,
                "exec_res": exec_res,
                "answer": ans,
                "suggestions": [
                    "Are there any missing values?",
                    "What are the column names and data types?",
                    "Show me the summary statistics of numeric columns"
                ]
            }

        code = "df.isna().sum()[df.isna().sum() > 0]"
        exec_res = execute_sandboxed_pandas(df, code)
        missing_count = int(df.isna().sum().sum())
        missing_pct = round(missing_count / (len(df) * len(df.columns)) * 100, 2) if len(df) > 0 else 0.0

        if missing_count == 0:
            ans = f"Great news! Your dataset has **0 missing values** across all **{len(df.columns)} columns**. All {len(df):,} records are completely filled."
        else:
            missing_cols = df.columns[df.isna().any()].tolist()
            ans = (
                f"We identified **{missing_count:,} missing cell(s)** ({missing_pct}% of total cells) "
                f"across columns: **{', '.join(missing_cols[:4])}**"
                f"{' and more' if len(missing_cols) > 4 else ''}. See the affected column breakdown below."
            )
        return {
            "code": code,
            "exec_res": exec_res,
            "answer": ans,
            "suggestions": [
                "Are there any duplicate rows?",
                "What is the total number of rows and columns?",
                "Show me summary statistics for numeric columns"
            ]
        }

    # 2a. Explicit Column / Feature listing
    if any(k in q for k in ["what are the columns", "list columns", "show columns", "column names", "available columns", "what columns", "features in", "features are", "which features"]):
        code = "pd.DataFrame({'Column': df.columns, 'Type': [str(df[c].dtype) for c in df.columns], 'Non-Null Rows': df.notna().sum().values})"
        exec_res = execute_sandboxed_pandas(df, code)
        cols_preview = ", ".join([f"**{c}**" for c in df.columns[:8]])
        ans = (
            f"The dataset contains **{len(df.columns)} columns (features)** across **{len(df):,} rows**: {cols_preview}"
            f"{' and more' if len(df.columns) > 8 else ''}. See the complete feature dictionary below."
        )
        return {
            "code": code,
            "exec_res": exec_res,
            "answer": ans,
            "suggestions": [
                f"What is the average {all_numeric_cols[0]}?" if all_numeric_cols else "Show me the first 5 rows",
                f"What is the breakdown of {all_cat_cols[0]}?" if all_cat_cols else "Are there missing values?",
                "What is the overall data quality score?"
            ]
        }

    # 2b. Shape, Row Count, Dimensions
    if any(k in q for k in ["how many rows", "dataset shape", "size of dataset", "how many records", "dimensions", "how many columns"]):
        code = "pd.Series({'Total Rows': len(df), 'Total Columns': len(df.columns), 'Duplicate Rows': int(df.duplicated().sum()), 'Missing Cells': int(df.isna().sum().sum())})"
        exec_res = execute_sandboxed_pandas(df, code)
        ans = (
            f"The dataset contains **{len(df):,} total records (rows)** and **{len(df.columns)} columns (features)**. "
            f"There are **{int(df.duplicated().sum())} duplicate rows** and **{int(df.isna().sum().sum())} missing cells**."
        )
        return {
            "code": code,
            "exec_res": exec_res,
            "answer": ans,
            "suggestions": [
                "Show me the first 5 rows",
                "What are the column names and data types?",
                "Are there any missing values in the data?"
            ]
        }

    # 3. Check for unmatchable missing column query (Prevents Hallucination!)
    missing_concept = detect_unmatchable_concept(question, df)
    if missing_concept and not matched_num and not matched_cat:
        available_sample = [f"'{c}'" for c in df.columns[:8]]
        code = "# Clarification: Column not found in dataset\nlist(df.columns)"
        exec_res = execute_sandboxed_pandas(df, code)
        ans = (
            f"I couldn't find a column related to **'{missing_concept}'** in this dataset (**{len(df):,} rows**, **{len(df.columns)} columns**).\n\n"
            f"The available columns in this dataset are: **{', '.join(available_sample)}**"
            f"{' ... and others' if len(df.columns) > 8 else ''}.\n\n"
            f"Would you like to ask a question about one of the available columns instead?"
        )
        return {
            "code": code,
            "exec_res": exec_res,
            "answer": ans,
            "suggestions": [
                f"What is the distribution of {all_numeric_cols[0]}?" if all_numeric_cols else "Show me the first 5 rows",
                f"What is the breakdown of {all_cat_cols[0]}?" if all_cat_cols else "What are the column names?",
                "Are there any missing values?"
            ]
        }

    # 4. Decoupled Zomato / Restaurant Queries (ONLY if culinary markers exist in active dataset!)
    is_rest_dataset = ("cuisine" in [c.lower() for c in df.columns] and any(c.lower() in ["cost_for_two", "restaurant_name", "location"] for c in df.columns))
    if is_rest_dataset and any(w in q for w in ["restaurant", "cuisine", "food", "dining", "zomato"]):
        cuisine_col = next((c for c in df.columns if "cuisine" in c.lower()), None)
        rating_col = next((c for c in df.columns if "rating" in c.lower() or "rate" in c.lower()), None)
        cost_col = next((c for c in df.columns if "cost" in c.lower()), None)
        loc_col = next((c for c in df.columns if "location" in c.lower() or "area" in c.lower()), None)
        votes_col = next((c for c in df.columns if "vote" in c.lower()), None)

        if ("cuisine" in q or "food" in q) and rating_col and cuisine_col:
            code = f"df.groupby('{cuisine_col}')['{rating_col}'].mean().round(2).sort_values(ascending=False).head(10)"
            exec_res = execute_sandboxed_pandas(df, code)
            summary = exec_res.get("summary_figures", {})
            top_cuisine = summary.get("top_item", "N/A")
            top_val = summary.get("max", "N/A")
            ans = f"The cuisine with the highest average rating is **{top_cuisine}** with an average score of **{top_val}**. See the complete ranking across top cuisines below."
            return {
                "code": code,
                "exec_res": exec_res,
                "answer": ans,
                "suggestions": [
                    "What is the correlation between cost for two and restaurant rating?",
                    "Which locations have the highest concentration of restaurants?",
                    "Show me the top 10 restaurants with the highest votes"
                ]
            }

        if ("location" in q or "neighborhood" in q or "area" in q) and loc_col:
            code = f"df['{loc_col}'].value_counts().head(10)"
            exec_res = execute_sandboxed_pandas(df, code)
            top_loc = df[loc_col].value_counts().index[0]
            top_cnt = df[loc_col].value_counts().iloc[0]
            ans = f"**{top_loc}** has the highest concentration of restaurants with **{top_cnt} recorded dining outlets**. See the neighborhood distribution below."
            return {
                "code": code,
                "exec_res": exec_res,
                "answer": ans,
                "suggestions": [
                    "Which cuisines have the highest average rating?",
                    "What is the correlation between cost for two and restaurant rating?",
                    "Show me the top 10 restaurants with the highest votes"
                ]
            }

        if ("vote" in q or "popular" in q) and votes_col and "Restaurant_Name" in df.columns:
            code = f"df.nlargest(10, '{votes_col}')[['Restaurant_Name', '{rating_col or votes_col}', '{votes_col}']]"
            exec_res = execute_sandboxed_pandas(df, code)
            top_rest = df.nlargest(1, votes_col)['Restaurant_Name'].iloc[0]
            max_v = df[votes_col].max()
            ans = f"The most popular restaurant is **{top_rest}** with **{max_v:,} customer votes**. Here are the top 10 most voted destinations."
            return {
                "code": code,
                "exec_res": exec_res,
                "answer": ans,
                "suggestions": [
                    "Which cuisines have the highest average rating?",
                    "What is the correlation between cost for two and restaurant rating?",
                    "Which locations have the highest concentration of restaurants?"
                ]
            }

    # 5. Trends Over Time & Percentage Change
    if (any(k in q for k in ["trend", "over time", "change", "percentage change", "growth", "growth rate", "year over year", "monthly", "by year"]) 
        and (matched_date or all_date_cols) and (matched_num or all_numeric_cols)):
        date_c = matched_date[0] if matched_date else all_date_cols[0]
        num_c = matched_num[0] if matched_num else all_numeric_cols[0]

        try:
            # Check date range span
            d_series = pd.to_datetime(df[date_c], errors="coerce").dropna()
            if len(d_series) > 2:
                span_days = (d_series.max() - d_series.min()).days
                period_str = "%Y" if span_days > 730 else "%Y-%m"
                period_label = "Year" if span_days > 730 else "Month"

                code = (
                    f"df.dropna(subset=['{date_c}', '{num_c}'])\n"
                    f"  .assign(_period=pd.to_datetime(df['{date_c}'], errors='coerce').dt.strftime('{period_str}'))\n"
                    f"  .groupby('_period')['{num_c}'].sum().round(2)"
                )
                exec_res = execute_sandboxed_pandas(df, code)
                
                # Calculate overall growth rate
                temp_df = df.dropna(subset=[date_c, num_c]).assign(
                    _period=pd.to_datetime(df[date_c], errors='coerce').dt.strftime(period_str)
                ).groupby('_period')[num_c].sum()

                if len(temp_df) >= 2:
                    first_val = float(temp_df.iloc[0])
                    last_val = float(temp_df.iloc[-1])
                    first_p = temp_df.index[0]
                    last_p = temp_df.index[-1]
                    
                    if first_val != 0:
                        pct_change = round(((last_val - first_val) / abs(first_val)) * 100, 2)
                        direction = "increase" if pct_change > 0 else "decrease"
                        ans = (
                            f"Analyzing the trend of **{num_c}** over time by {period_label}: "
                            f"Values changed from **{first_val:,}** ({first_p}) to **{last_val:,}** ({last_p}), "
                            f"representing a **{abs(pct_change)}% {direction}** over the recorded period. "
                            f"See the chronological progression in the chart below."
                        )
                    else:
                        ans = f"Analyzing **{num_c}** by {period_label}: Values progressed from **{first_val:,}** ({first_p}) to **{last_val:,}** ({last_p})."

                    # Generate line chart
                    plotly_fig = create_plotly_spec(
                        chart_type="line",
                        title=f"{num_c} Trend by {period_label}",
                        x_axis=period_label,
                        y_axis=f"Total {num_c}",
                        labels=[str(x) for x in temp_df.index],
                        values=[round(float(v), 2) for v in temp_df.values]
                    )
                    exec_res["chart_spec"] = ChartSpec(
                        chart_type="line",
                        title=f"{num_c} Trend by {period_label}",
                        x_axis=period_label,
                        y_axis=f"Total {num_c}",
                        labels=[str(x) for x in temp_df.index],
                        data=[{"x": str(k), "y": round(float(v), 2)} for k, v in temp_df.items()],
                        description=f"Chronological trend calculated across {len(temp_df)} {period_label.lower()}s.",
                        plotly_spec=plotly_fig,
                        data_stage="analysis_ready",
                        rows_used=len(df.dropna(subset=[date_c, num_c])),
                        transformations=f"Aggregated sum of '{num_c}' grouped by {period_label}",
                        exclusions=f"Excluded {int(df[date_c].isna().sum())} null dates" if df[date_c].isna().any() else None
                    )

                    return {
                        "code": code,
                        "exec_res": exec_res,
                        "answer": ans,
                        "suggestions": [
                            f"What is the average {num_c} by categories?",
                            f"What are the top 5 highest records of {num_c}?",
                            "Are there any outliers in this metric?"
                        ]
                    }
        except Exception as e:
            logger.warning(f"Trend calculation fallback: {e}")

    # 6. Correlation between numeric variables
    if any(k in q for k in ["correlation", "correlate", "relationship", "correlated"]):
        if len(matched_num) >= 2:
            c1, c2 = matched_num[0], matched_num[1]
        elif len(all_numeric_cols) >= 2:
            c1, c2 = all_numeric_cols[0], all_numeric_cols[1]
        else:
            c1, c2 = None, None

        if c1 and c2:
            code = f"round(df[['{c1}', '{c2}']].dropna().corr().iloc[0, 1], 3)"
            exec_res = execute_sandboxed_pandas(df, code)
            r_val = exec_res.get("summary_figures", {}).get("primary_value", 0.0)
            
            abs_r = abs(float(r_val) if r_val is not None else 0.0)
            sign = "positive" if (r_val or 0) > 0 else "negative"
            if abs_r >= 0.7:
                strength = f"a **strong {sign} correlation**"
            elif abs_r >= 0.35:
                strength = f"a **moderate {sign} correlation**"
            elif abs_r >= 0.15:
                strength = f"a **weak {sign} correlation**"
            else:
                strength = "**virtually no linear correlation**"

            ans = (
                f"The Pearson correlation between **{c1}** and **{c2}** is **{r_val}**. "
                f"This indicates {strength}. As {c1} changes, {c2} tends to move in the {sign} direction."
            )

            # Generate scatter plot
            sub_df = df[[c1, c2]].dropna()
            if len(sub_df) > 300:
                sub_df = sub_df.sample(300, random_state=42)
            pts = [{"x": float(row[c1]), "y": float(row[c2])} for _, row in sub_df.iterrows()]
            plotly_fig = create_plotly_spec(
                chart_type="scatter",
                title=f"{c2} vs {c1} (r = {r_val})",
                x_axis=c1,
                y_axis=c2,
                points=pts
            )
            exec_res["chart_spec"] = ChartSpec(
                chart_type="scatter",
                title=f"{c2} vs {c1}",
                x_axis=c1,
                y_axis=c2,
                data=pts,
                description=f"Scatter plot of {len(pts)} coordinate points showing Pearson correlation {r_val}.",
                plotly_spec=plotly_fig,
                data_stage="analysis_ready",
                rows_used=len(sub_df),
                transformations=f"Sampled {len(pts)} coordinate points across '{c1}' and '{c2}'",
                exclusions=f"Excluded {len(df) - len(sub_df)} rows with missing values" if (len(df) - len(sub_df)) > 0 else None
            )

            return {
                "code": code,
                "exec_res": exec_res,
                "answer": ans,
                "suggestions": [
                    f"What is the average {c1}?",
                    f"Show the distribution of {c2}",
                    "What other columns have strong correlations?"
                ]
            }

    # 7. Rankings: Top N or Bottom/Lowest N
    is_bottom = any(w in q for w in ["bottom", "lowest", "smallest", "least", "worst", "minimum 5", "min 5"])
    is_top = any(w in q for w in ["top", "highest", "most", "largest", "best", "leading"])

    if (is_top or is_bottom) and (matched_num or all_numeric_cols):
        target_num = matched_num[0] if matched_num else all_numeric_cols[0]
        match_n = re.search(r"\b(\d+)\b", q)
        n = int(match_n.group(1)) if match_n and int(match_n.group(1)) <= 50 else 5

        # Choose 2-4 clean columns to display
        display_cols = [c for c in (matched_cat + [target_num]) if c in df.columns]
        if len(display_cols) < 2:
            display_cols = [c for c in df.columns if c != target_num][:2] + [target_num]

        if is_bottom:
            code = f"df.nsmallest({n}, '{target_num}')[{display_cols}]"
            exec_res = execute_sandboxed_pandas(df, code)
            min_val = round(float(df[target_num].min()), 2)
            ans = (
                f"Here are the **bottom {n} records** ranked by lowest **{target_num}**. "
                f"The lowest recorded value in the dataset is **{min_val:,}**."
            )
        else:
            code = f"df.nlargest({n}, '{target_num}')[{display_cols}]"
            exec_res = execute_sandboxed_pandas(df, code)
            max_val = round(float(df[target_num].max()), 2)
            ans = (
                f"Here are the **top {n} records** ranked by highest **{target_num}**. "
                f"The highest recorded value in the dataset is **{max_val:,}**."
            )

        return {
            "code": code,
            "exec_res": exec_res,
            "answer": ans,
            "suggestions": [
                f"What is the average {target_num}?",
                f"Show the distribution and summary statistics of {target_num}",
                f"What is {target_num} grouped by category?"
            ]
        }

    # 7b. Numeric row filtering with comparison (e.g. Price > 500, Rating >= 4.0)
    for num_c in (matched_num or all_numeric_cols[:3]):
        filter_match = re.search(r"\b" + re.escape(num_c.lower()) + r"\b\s*(>=|<=|>|<|==|=|greater than|less than|above|below|exceeding)\s*(\d+(?:\.\d+)?)", q)
        if filter_match:
            op_str = filter_match.group(1)
            val_str = filter_match.group(2)
            op_map = {">": ">", ">=": ">=", "<": "<", "<=": "<=", "==": "==", "=": "==", "greater than": ">", "above": ">", "exceeding": ">", "less than": "<", "below": "<"}
            op = op_map.get(op_str, ">")
            val_float = float(val_str)
            code = f"df[df['{num_c}'] {op} {val_str}].head(10)"
            exec_res = execute_sandboxed_pandas(df, code)
            
            # Count exact matches
            if op == ">":
                match_count = int((df[num_c] > val_float).sum())
            elif op == ">=":
                match_count = int((df[num_c] >= val_float).sum())
            elif op == "<":
                match_count = int((df[num_c] < val_float).sum())
            elif op == "<=":
                match_count = int((df[num_c] <= val_float).sum())
            else:
                match_count = int((df[num_c] == val_float).sum())

            pct = round(match_count / len(df) * 100, 1) if len(df) > 0 else 0
            ans = f"There are **{match_count:,} records** ({pct}% of the dataset) where **{num_c} {op} {val_str}**. Here are the first matching records."
            return {
                "code": code,
                "exec_res": exec_res,
                "answer": ans,
                "suggestions": [
                    f"What is the average {num_c}?",
                    f"Show the distribution of {num_c}",
                    f"What are the top 5 records of {num_c}?"
                ]
            }

    # 8. Aggregation Logic: Grouped Aggregation vs. Single-Column Aggregation
    is_grouped_intent = bool(matched_cat) or any(w in q for w in [" by ", " per ", " each ", " across ", " grouped "])
    target_num = matched_num[0] if matched_num else (all_numeric_cols[0] if len(all_numeric_cols) == 1 and any(w in q for w in ["average", "mean", "total", "sum"]) else None)
    target_cat = matched_cat[0] if matched_cat else (all_cat_cols[0] if is_grouped_intent and len(all_cat_cols) == 1 else None)

    # Determine aggregation operation
    if any(w in q for w in ["total", "sum", "overall"]):
        agg_func = "sum"
        agg_word = "total"
    elif any(w in q for w in ["maximum", "max", "highest"]):
        agg_func = "max"
        agg_word = "maximum"
    elif any(w in q for w in ["minimum", "min", "lowest"]):
        agg_func = "min"
        agg_word = "minimum"
    elif any(w in q for w in ["count", "number of", "how many"]):
        agg_func = "count"
        agg_word = "count"
    else:
        agg_func = "mean"
        agg_word = "average"

    # If user asks for aggregation without specifying which column among multiple numeric columns:
    if not target_num and not is_grouped_intent and any(w in q for w in ["average", "mean", "total", "sum", "max", "min"]) and all_numeric_cols:
        available_sample = [f"'{c}'" for c in all_numeric_cols[:6]]
        code = "# Clarification: Ambiguous aggregation target\n[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]"
        exec_res = execute_sandboxed_pandas(df, code)
        ans = (
            f"You asked to calculate the **{agg_word}**, but did not specify which column. "
            f"The dataset contains these numeric columns: **{', '.join(available_sample)}**"
            f"{' ... and others' if len(all_numeric_cols) > 6 else ''}. Which one would you like to calculate?"
        )
        return {
            "code": code,
            "exec_res": exec_res,
            "answer": ans,
            "suggestions": [
                f"What is the {agg_word} of {c}?" for c in all_numeric_cols[:3]
            ]
        }

    # 8a. Explicit Grouped Aggregation
    if is_grouped_intent and target_cat and target_num and (target_cat in df.columns and target_num in df.columns):
        code = f"df.groupby('{target_cat}')['{target_num}'].{agg_func}().round(2).sort_values(ascending=False)"
        exec_res = execute_sandboxed_pandas(df, code)

        summary = exec_res.get("summary_figures", {})
        top_group = summary.get("top_item", "N/A")
        top_val = summary.get("max", "N/A")

        ans = (
            f"When calculating the **{agg_word} {target_num}** grouped by **{target_cat}**, "
            f"**{top_group}** leads with an {agg_word} of **{top_val}**. "
            f"See the complete comparison in the chart and table below."
        )
        return {
            "code": code,
            "exec_res": exec_res,
            "answer": ans,
            "suggestions": [
                f"What is the total sum of {target_num} by {target_cat}?",
                f"Show the distribution of {target_num}",
                f"What are the top 5 records for {target_num}?"
            ]
        }

    # 8b. Single Column Aggregation (e.g. "What is the average Price?" or "total Sales")
    if target_num and target_num in df.columns and any(w in q for w in ["average", "mean", "total", "sum", "min", "minimum", "max", "maximum", "count"]):
        if agg_func == "mean":
            calc_val = round(float(df[target_num].mean()), 2)
            code = f"round(df['{target_num}'].mean(), 2)"
        elif agg_func == "sum":
            calc_val = round(float(df[target_num].sum()), 2)
            code = f"round(df['{target_num}'].sum(), 2)"
        elif agg_func == "max":
            calc_val = round(float(df[target_num].max()), 2)
            code = f"round(df['{target_num}'].max(), 2)"
        elif agg_func == "min":
            calc_val = round(float(df[target_num].min()), 2)
            code = f"round(df['{target_num}'].min(), 2)"
        else:
            calc_val = int(df[target_num].count())
            code = f"int(df['{target_num}'].count())"

        exec_res = execute_sandboxed_pandas(df, code)
        min_v = round(float(df[target_num].min()), 2)
        max_v = round(float(df[target_num].max()), 2)
        med_v = round(float(df[target_num].median()), 2)

        ans = (
            f"The {agg_word} **{target_num}** across all **{len(df):,} records** is **{calc_val:,}** "
            f"(with recorded values ranging from **{min_v:,}** to **{max_v:,}** and a median of **{med_v:,}**)."
        )
        return {
            "code": code,
            "exec_res": exec_res,
            "answer": ans,
            "suggestions": [
                f"Show the full distribution of {target_num}",
                f"What are the top 5 records for {target_num}?",
                f"What is the {target_num} grouped by category?"
            ]
        }

    # 9. Single Column Distribution / Summary Statistics
    # ONLY executed if user specifically requested distribution/statistics OR explicitly named a numeric column
    if target_num and target_num in df.columns and (bool(matched_num) or any(w in q for w in ["distribution", "summary", "stats", "statistics", "describe", "spread", "variance", "std", "quantile", "percentile"])):
        code = f"round(df['{target_num}'].describe(), 2)"
        exec_res = execute_sandboxed_pandas(df, code)
        mean_val = round(float(df[target_num].mean()), 2)
        min_val = round(float(df[target_num].min()), 2)
        max_val = round(float(df[target_num].max()), 2)
        median_val = round(float(df[target_num].median()), 2)

        ans = (
            f"For **{target_num}**, recorded values range from **{min_val:,}** to **{max_val:,}**, "
            f"with an average (mean) of **{mean_val:,}** and a median of **{median_val:,}**. "
            f"The full statistical distribution is presented below."
        )
        return {
            "code": code,
            "exec_res": exec_res,
            "answer": ans,
            "suggestions": [
                f"What are the top 5 highest records of {target_num}?",
                f"Are there any outliers in {target_num}?",
                f"What is {target_num} grouped by category?"
            ]
        }

    # 10. Categorical Frequency Breakdown
    # ONLY executed if user specifically requested breakdown/categories OR explicitly named a categorical column
    if target_cat and target_cat in df.columns and (bool(matched_cat) or any(w in q for w in ["breakdown", "frequency", "categories", "distribution", "value counts", "most common", "popular"])):
        code = f"df['{target_cat}'].value_counts()"
        exec_res = execute_sandboxed_pandas(df, code)
        top_cat = df[target_cat].value_counts().index[0]
        top_count = int(df[target_cat].value_counts().iloc[0])
        pct = round(top_count / len(df) * 100, 1)

        ans = (
            f"In **{target_cat}**, the most frequent value is **{top_cat}** with **{top_count:,} occurrences** "
            f"({pct}% of the dataset). The complete category breakdown is visualized below."
        )
        return {
            "code": code,
            "exec_res": exec_res,
            "answer": ans,
            "suggestions": [
                "Show me the overall data health report",
                "Are there any duplicate rows?",
                "What other questions can I ask about this dataset?"
            ]
        }

    # 11. Explicit Limitation: The query cannot be answered from the available dataset
    code = "# Limitation: Question cannot be answered from available dataset\nlist(df.columns)"
    exec_res = execute_sandboxed_pandas(df, code)
    available_sample = [f"'{c}'" for c in df.columns[:8]]
    ans = (
        f"I cannot answer this question from the available data. The question asks about concepts or analysis not supported by the active dataset (**{len(df):,} rows**, **{len(df.columns)} columns**).\n\n"
        f"Available columns are: **{', '.join(available_sample)}**"
        f"{' ... and others' if len(df.columns) > 8 else ''}.\n\n"
        f"Please ask an analytical question grounded in the available columns."
    )
    return {
        "code": code,
        "exec_res": exec_res,
        "answer": ans,
        "suggestions": [
            f"What is the average of {all_numeric_cols[0]}?" if all_numeric_cols else "What are the column names?",
            f"What is the breakdown of {all_cat_cols[0]}?" if all_cat_cols else "Are there missing values?",
            "What is the overall data quality score?"
        ]
    }


def query_llm_gemini(
    df: pd.DataFrame,
    question: str,
    api_key: str
) -> Dict[str, Any]:
    """
    Use Google Gemini API to generate sandboxed pandas code,
    execute against the real dataset, and synthesize a plain-English explanation.
    """
    if not HAS_GEMINI_PKG:
        raise RuntimeError("google-generativeai package is not available.")

    genai.configure(api_key=api_key)

    col_summary = []
    for col in df.columns[:30]:
        dtype_str = str(df[col].dtype)
        unique_cnt = df[col].nunique()
        sample_vals = [str(x) for x in df[col].dropna().head(3).tolist()]
        col_summary.append(f"- '{col}' ({dtype_str}, {unique_cnt} unique): e.g. {sample_vals}")
    schema_str = "\n".join(col_summary)

    prompt_for_code = f"""You are DataSense AI, an expert, beginner-friendly data science assistant.
A user asked this analytical question about a dataset:
"{question}"

Dataset Schema ({len(df)} rows, {len(df.columns)} columns):
{schema_str}

Instructions:
1. Write a SINGLE Python expression or short script using `df`, `pd`, and `np` to calculate the exact answer.
2. Store the result in a variable named `result` (or write a single expression).
3. The code must be clean, safe pandas code (NO file I/O, NO imports, NO network).
4. If the question asks for a metric or column that DOES NOT exist in the schema, set:
   `result = "CLARIFICATION: Column not found"`
5. Return ONLY a valid JSON object in this exact schema:
{{
  "code": "Python pandas code string",
  "explanation_intent": "Brief 1-sentence note of what this calculates"
}}
"""

    model = genai.GenerativeModel("gemini-2.5-flash" if "gemini-2.5-flash" in [m.name for m in []] else "gemini-1.5-flash")
    
    response = model.generate_content(
        prompt_for_code,
        generation_config={"response_mime_type": "application/json"}
    )
    
    parsed = json.loads(response.text)
    code_to_exec = parsed.get("code", "df.head(5)")

    # Execute against real data
    exec_res = execute_sandboxed_pandas(df, code_to_exec)
    
    # If the LLM's code errored, fallback cleanly to heuristic engine
    if not exec_res.get("success"):
        logger.warning(f"LLM generated code failed: {exec_res.get('error')}. Falling back to heuristic engine.")
        return heuristic_query_engine(df, question)

    summary_numbers = exec_res.get("summary_figures", {})
    table_preview = exec_res.get("table_result", [])[:10]

    explain_prompt = f"""You are DataSense AI. Explain these analytical findings to a beginner business user.
User Question: "{question}"
Pandas Code Executed: `{code_to_exec}`
Exact Computed Figures: {json.dumps(summary_numbers)}
Data Sample: {json.dumps(table_preview)}

Rules:
1. Explain the results in simple, encouraging, plain English.
2. You MUST cite the exact calculated numbers provided above. DO NOT invent or hallucinate any numbers.
3. Keep the explanation to 2-3 concise, impactful sentences.
4. Suggest 3 relevant follow-up questions the user might want to ask next.

Return JSON:
{{
  "answer": "Clear explanation with exact numbers in bold",
  "suggestions": ["Follow-up Q1", "Follow-up Q2", "Follow-up Q3"]
}}
"""

    explain_resp = model.generate_content(
        explain_prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    explain_data = json.loads(explain_resp.text)

    return {
        "code": code_to_exec,
        "exec_res": exec_res,
        "answer": explain_data.get("answer", "Here are the computed results from your dataset."),
        "suggestions": explain_data.get("suggestions", ["What else would you like to explore?"])
    }


def process_user_query(
    df: pd.DataFrame,
    question: str,
    api_key: Optional[str] = None
) -> QueryResponse:
    """Main query dispatcher with graceful fallbacks and guaranteed grounded answers."""
    cleaned_q = question.strip()
    active_key = api_key or os.getenv("GEMINI_API_KEY")

    try:
        if active_key and HAS_GEMINI_PKG:
            result_dict = query_llm_gemini(df, cleaned_q, active_key)
        else:
            result_dict = heuristic_query_engine(df, cleaned_q)
    except Exception as e:
        logger.exception(f"Error in query dispatcher: {e}")
        result_dict = heuristic_query_engine(df, cleaned_q)

    exec_res = result_dict["exec_res"]
    
    return QueryResponse(
        question=cleaned_q,
        answer_text=result_dict["answer"],
        code_executed=result_dict["code"],
        result_type=exec_res.get("result_type", "scalar"),
        summary_figures=exec_res.get("summary_figures", {}),
        table_result=exec_res.get("table_result"),
        table_columns=exec_res.get("table_columns"),
        chart_spec=exec_res.get("chart_spec"),
        follow_up_suggestions=result_dict.get("suggestions", []),
        execution_time_ms=exec_res.get("execution_time_ms", 0.0),
        grounded=True,
        error=exec_res.get("error")
    )
