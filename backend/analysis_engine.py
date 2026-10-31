import ast
import time
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from models import ChartSpec, CustomChartRequest


DISALLOWED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals",
    "vars", "dir", "os", "sys", "subprocess", "shutil", "socket", "urllib",
    "requests", "httpx", "pathlib", "importlib", "builtins", "input"
}


def validate_python_code_ast(code_str: str) -> Tuple[bool, Optional[str]]:
    """Inspect abstract syntax tree to ensure only safe data transformations are executed."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Syntax error in generated analysis code: {e}"

    for node in ast.walk(tree):
        # Disallow any import statements
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "Import statements are disallowed in sandboxed execution."
        
        # Disallow dangerous names
        if isinstance(node, ast.Name) and node.id in DISALLOWED_NAMES:
            return False, f"Use of restricted identifier '{node.id}' is prohibited."
            
        # Disallow attribute access on dunder methods
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") and node.attr.endswith("__")):
            return False, f"Access to private dunder attribute '{node.attr}' is prohibited."

    return True, None


def sanitize_val(v: Any) -> Any:
    if v is None:
        return None
    # If a Series or array was passed, extract the scalar value safely
    if isinstance(v, (pd.Series, np.ndarray, list)):
        if hasattr(v, 'iloc') and len(v) > 0:
            v = v.iloc[0]
        elif len(v) > 0:
            v = v[0]
        else:
            return None
    try:
        is_na = pd.isna(v)
        if isinstance(is_na, (pd.Series, np.ndarray, list)):
            if any(is_na):
                return None
        elif bool(is_na):
            return None
    except Exception:
        pass
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(float(v), 3)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return str(v)
    return str(v)


def execute_sandboxed_pandas(df: pd.DataFrame, code_str: str) -> Dict[str, Any]:
    """Execute validated pandas code safely and return structured results and figures."""
    is_safe, error_msg = validate_python_code_ast(code_str)
    if not is_safe:
        return {
            "success": False,
            "error": error_msg,
            "result_type": "error",
            "summary_figures": {},
            "code_executed": code_str,
            "execution_time_ms": 0.0
        }

    # Prepare sandboxed namespace
    safe_globals = {
        "__builtins__": {
            "len": len,
            "range": range,
            "round": round,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "int": int,
            "float": float,
            "str": str,
            "dict": dict,
            "list": list,
            "set": set,
            "sorted": sorted,
            "zip": zip,
            "enumerate": enumerate,
            "True": True,
            "False": False,
            "None": None,
        },
        "pd": pd,
        "np": np,
    }

    # Local namespace with a shallow/defensive copy of df
    local_ns = {"df": df.copy(), "result": None}

    start_time = time.perf_counter()
    try:
        # Check if the code is a single expression or a multi-line script
        lines = [line for line in code_str.strip().splitlines() if line.strip() and not line.strip().startswith("#")]
        
        if len(lines) == 1 and not lines[0].startswith("result =") and not lines[0].startswith("def ") and "=" not in lines[0]:
            # Evaluate directly as an expression
            raw_result = eval(lines[0], safe_globals, local_ns)
        else:
            # Multi-line statement execution
            exec(code_str, safe_globals, local_ns)
            raw_result = local_ns.get("result")
            if raw_result is None and len(lines) > 0:
                # If 'result' was not assigned, try evaluating the last line
                last_line = lines[-1].strip()
                if not last_line.startswith("return") and "=" not in last_line:
                    raw_result = eval(last_line, safe_globals, local_ns)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "success": False,
            "error": f"Execution error: {str(exc)}",
            "result_type": "error",
            "summary_figures": {},
            "code_executed": code_str,
            "execution_time_ms": elapsed_ms
        }

    # Format result for UI
    return format_execution_result(raw_result, code_str, elapsed_ms)


def create_plotly_spec(
    chart_type: str,
    title: str,
    x_axis: Optional[str] = None,
    y_axis: Optional[str] = None,
    labels: Optional[List[str]] = None,
    values: Optional[List[Any]] = None,
    points: Optional[List[Dict[str, Any]]] = None,
    orientation: str = "v",
    is_subset: bool = False,
    total_categories: Optional[int] = None,
    displayed_categories: Optional[int] = None,
    low_frequency_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate high quality Plotly figure specification matching the light warm teal/orange theme."""
    color_palette = [
        "#EA580C", "#0D9488", "#F59E0B", "#6366F1", "#EC4899", 
        "#14B8A6", "#8B5CF6", "#10B981", "#3B82F6", "#F97316"
    ]
    
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    elif labels is not None:
        labels = list(labels)

    if hasattr(values, "tolist"):
        values = values.tolist()
    elif values is not None:
        values = list(values)

    traces = []
    has_labels = bool(labels is not None and len(labels) > 0)
    has_values = bool(values is not None and len(values) > 0)
    cat_count = len(labels) if has_labels else 0

    if chart_type == "scatter" and points:
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": [p.get("x") for p in points],
            "y": [p.get("y") for p in points],
            "marker": {
                "size": 8,
                "color": "#EA580C",
                "opacity": 0.8,
                "line": {"color": "#FFFFFF", "width": 1}
            },
            "name": title
        })
    elif chart_type == "pie" and has_labels and has_values:
        traces.append({
            "type": "pie",
            "labels": labels,
            "values": values,
            "hole": 0.42,
            "marker": {"colors": color_palette},
            "textinfo": "label+percent",
            "hoverinfo": "label+value+percent"
        })
    elif chart_type == "line" and has_labels and has_values:
        traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "x": labels,
            "y": values,
            "line": {"color": "#EA580C", "width": 3, "shape": "spline"},
            "marker": {"color": "#0F3D3E", "size": 6},
            "name": y_axis or "Value"
        })
    elif orientation == "h" and has_labels and has_values:
        # Horizontal categorical bar chart for superior readability with >=6 categories or long labels
        short_labels = [str(l)[:28] + ("..." if len(str(l)) > 28 else "") for l in labels]
        hover_texts = [str(l) for l in labels]
        traces.append({
            "type": "bar",
            "orientation": "h",
            "x": values,
            "y": short_labels,
            "hovertext": hover_texts,
            "hovertemplate": "<b>%{hovertext}</b><br>" + (y_axis or "Value") + ": %{x:,}<extra></extra>",
            "marker": {
                "color": "#EA580C",
                "opacity": 0.9,
                "line": {"color": "#C2410C", "width": 1}
            },
            "name": y_axis or "Value"
        })
    else:  # default vertical bar or histogram
        short_labels = [str(l)[:22] + ("..." if len(str(l)) > 22 else "") for l in (labels or [])]
        hover_texts = [str(l) for l in (labels or [])]
        traces.append({
            "type": "bar",
            "x": short_labels if has_labels else [p.get("x") for p in (points or [])],
            "y": values if has_values else [p.get("y") for p in (points or [])],
            "hovertext": hover_texts if has_labels else None,
            "hovertemplate": ("<b>%{hovertext}</b><br>" + (y_axis or "Value") + ": %{y:,}<extra></extra>") if has_labels else None,
            "marker": {
                "color": "#EA580C",
                "opacity": 0.9,
                "line": {"color": "#C2410C", "width": 1}
            },
            "name": y_axis or "Value"
        })

    # Title styling with subset indicator
    title_text = title
    if is_subset and total_categories and displayed_categories:
        title_text = f"{title}<br><sup style='color:#648484;font-size:11px'>Showing Top {displayed_categories} of {total_categories} categories</sup>"

    # Dynamic height calculation
    if orientation == "h" and cat_count > 0:
        layout_height = max(380, min(1400, cat_count * 28 + 120))
        left_margin = min(220, max(130, max((len(str(l)) for l in labels), default=10) * 7))
    else:
        layout_height = 380
        left_margin = 60

    layout = {
        "title": {
            "text": title_text,
            "font": {"family": "Inter, sans-serif", "size": 15, "color": "#0F3D3E"}
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "color": "#0F3D3E"},
        "height": layout_height,
        "margin": {"t": 55, "r": 24, "b": 60, "l": left_margin},
        "xaxis": {
            "title": {"text": (y_axis or "Value") if orientation == "h" else (x_axis or ""), "font": {"color": "#0F3D3E", "size": 12}},
            "tickfont": {"color": "#476D6D", "size": 11},
            "gridcolor": "#EFECE6",
            "showgrid": True,
            "zeroline": False,
            "automargin": True,
            "tickangle": 0 if orientation == "h" or cat_count <= 5 else -30
        },
        "yaxis": {
            "title": {"text": (x_axis or "") if orientation == "h" else (y_axis or ""), "font": {"color": "#0F3D3E", "size": 12}},
            "tickfont": {"color": "#476D6D", "size": 11},
            "gridcolor": "#EFECE6",
            "showgrid": True if orientation != "h" else False,
            "zeroline": False,
            "automargin": True,
            "autorange": "reversed" if orientation == "h" else True  # top item at top for descending reading
        },
        "hoverlabel": {
            "bgcolor": "#0F3D3E",
            "font": {"color": "#FFFFFF", "family": "Inter, sans-serif"}
        }
    }

    if low_frequency_note:
        layout["annotations"] = [{
            "text": f"ℹ️ {low_frequency_note}",
            "xref": "paper",
            "yref": "paper",
            "x": 0.0,
            "y": -0.18 if orientation == "h" else -0.22,
            "showarrow": False,
            "font": {"size": 11, "color": "#648484", "family": "Inter, sans-serif"},
            "align": "left"
        }]
    
    return {"data": traces, "layout": layout}


def format_execution_result(raw_result: Any, code_str: str, elapsed_ms: float) -> Dict[str, Any]:
    """Parse raw execution output into tables, metrics, or charts."""
    summary_figures: Dict[str, Any] = {}
    chart_spec: Optional[ChartSpec] = None
    table_result: Optional[List[Dict[str, Any]]] = None
    table_columns: Optional[List[str]] = None
    result_type = "scalar"

    # 1. Scalar output
    if isinstance(raw_result, (int, float, np.integer, np.floating, str, bool, np.bool_)):
        clean_val = sanitize_val(raw_result)
        summary_figures["primary_value"] = clean_val
        result_type = "scalar"

    # 2. Pandas Series output
    elif isinstance(raw_result, pd.Series):
        s = raw_result.dropna()
        table_columns = ["Index", "Value"]
        rows = []
        chart_labels = []
        chart_data = []
        
        for idx, val in s.head(30).items():
            c_val = sanitize_val(val)
            c_idx = str(idx)
            rows.append({"Index": c_idx, "Value": c_val})
            chart_labels.append(c_idx)
            chart_data.append(c_val if isinstance(c_val, (int, float)) else 0)

        table_result = rows
        result_type = "table"
        
        if pd.api.types.is_numeric_dtype(s):
            summary_figures["count"] = int(s.count())
            summary_figures["mean"] = sanitize_val(s.mean())
            summary_figures["max"] = sanitize_val(s.max())
            summary_figures["min"] = sanitize_val(s.min())
            summary_figures["top_item"] = str(s.idxmax()) if len(s) > 0 else "N/A"
            
            c_type = "bar" if len(chart_labels) <= 15 else "line"
            c_title = f"Analysis of {s.name or 'Values'}"
            plotly_figure = create_plotly_spec(
                chart_type=c_type,
                title=c_title,
                x_axis="Group",
                y_axis=str(s.name or "Value"),
                labels=chart_labels,
                values=chart_data
            )
            
            # Auto-generate chart spec
            chart_spec = ChartSpec(
                chart_type=c_type,
                title=c_title,
                labels=chart_labels,
                data=[{"x": lbl, "y": val} for lbl, val in zip(chart_labels, chart_data)],
                description=f"Computed across {len(s)} data groups.",
                plotly_spec=plotly_figure
            )
            result_type = "chart"

    # 3. Pandas DataFrame output
    elif isinstance(raw_result, pd.DataFrame):
        df_res = raw_result.head(50)
        table_columns = [str(c) for c in df_res.columns]
        table_result = []
        for _, row in df_res.iterrows():
            table_result.append({str(k): sanitize_val(v) for k, v in row.items()})

        result_type = "table"
        summary_figures["total_rows_returned"] = len(raw_result)
        summary_figures["total_columns"] = len(raw_result.columns)

        # Check if 2 columns (category + numeric) -> eligible for auto-chart
        if len(df_res.columns) == 2:
            col1, col2 = df_res.columns[0], df_res.columns[1]
            if pd.api.types.is_numeric_dtype(df_res[col2]):
                c_labels = [str(r[col1]) for r in table_result[:15]]
                c_values = [r[col2] for r in table_result[:15]]
                c_title = f"{col2} by {col1}"
                plotly_figure = create_plotly_spec(
                    chart_type="bar",
                    title=c_title,
                    x_axis=str(col1),
                    y_axis=str(col2),
                    labels=c_labels,
                    values=c_values
                )
                chart_spec = ChartSpec(
                    chart_type="bar",
                    title=c_title,
                    x_axis=str(col1),
                    y_axis=str(col2),
                    labels=c_labels,
                    data=[{"x": l, "y": v} for l, v in zip(c_labels, c_values)],
                    description=f"Comparison of {col2} grouped by {col1}.",
                    plotly_spec=plotly_figure
                )
                result_type = "chart"

    # 4. Dictionary or List output
    elif isinstance(raw_result, dict):
        result_type = "table"
        table_columns = ["Key", "Value"]
        table_result = [{"Key": str(k), "Value": sanitize_val(v)} for k, v in raw_result.items()]
    elif isinstance(raw_result, list):
        result_type = "table"
        table_columns = ["Item"]
        table_result = [{"Item": sanitize_val(v)} for v in raw_result[:50]]
    else:
        result_type = "scalar"
        summary_figures["primary_value"] = str(raw_result)

    return {
        "success": True,
        "error": None,
        "result_type": result_type,
        "summary_figures": summary_figures,
        "table_result": table_result,
        "table_columns": table_columns,
        "chart_spec": chart_spec,
        "code_executed": code_str,
        "execution_time_ms": elapsed_ms
    }


def generate_custom_chart(df: pd.DataFrame, req: CustomChartRequest) -> ChartSpec:
    """Generate interactive chart data strictly aggregated from the real DataFrame."""
    x_col = req.x_column
    y_col = req.y_column
    c_type = req.chart_type
    agg = req.aggregation or "mean"
    limit = req.limit or 20

    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' does not exist in dataset.")

    # 1. Histogram
    if c_type == "histogram":
        series = pd.to_numeric(df[x_col], errors="coerce").dropna()
        if len(series) == 0:
            raise ValueError(f"Column '{x_col}' has no numeric values to plot histogram.")
        
        counts, bin_edges = np.histogram(series, bins=15)
        labels = [f"{round(bin_edges[i], 1)} - {round(bin_edges[i+1], 1)}" for i in range(len(counts))]
        data = [{"x": labels[i], "y": int(counts[i])} for i in range(len(counts))]
        
        c_title = f"Distribution of {x_col}"
        plotly_figure = create_plotly_spec(
            chart_type="bar",
            title=c_title,
            x_axis=f"{x_col} Bins",
            y_axis="Frequency Count",
            labels=labels,
            values=[int(c) for c in counts]
        )
        
        return ChartSpec(
            chart_type="bar",
            title=c_title,
            x_axis=f"{x_col} Bins",
            y_axis="Frequency Count",
            labels=labels,
            data=data,
            description=f"Histogram with 15 bins across {len(series)} recorded points.",
            plotly_spec=plotly_figure,
            data_stage="analysis_ready",
            rows_used=len(series),
            transformations=f"Binned numeric values of '{x_col}' into 15 equal-width intervals",
            exclusions=f"Excluded {int(df[x_col].isna().sum())} missing/non-numeric cells" if df[x_col].isna().any() else None
        )

    # 2. Scatter plot
    if c_type == "scatter":
        if not y_col or y_col not in df.columns:
            raise ValueError("Scatter plot requires both an X and Y column.")
        
        if x_col == y_col:
            sub_series = df[x_col].dropna()
            if len(sub_series) > 500:
                sub_series = sub_series.sample(500, random_state=42)
            points = [{"x": sanitize_val(val), "y": sanitize_val(val)} for val in sub_series]
            sub_len = len(sub_series)
        else:
            sub_df = df[[x_col, y_col]].dropna()
            if len(sub_df) > 500:
                sub_df = sub_df.sample(500, random_state=42)
            points = []
            for _, row in sub_df.iterrows():
                points.append({
                    "x": sanitize_val(row[x_col]),
                    "y": sanitize_val(row[y_col])
                })
            sub_len = len(sub_df)
            
        c_title = f"{y_col} vs {x_col}"
        plotly_figure = create_plotly_spec(
            chart_type="scatter",
            title=c_title,
            x_axis=x_col,
            y_axis=y_col,
            points=points
        )
        
        return ChartSpec(
            chart_type="scatter",
            title=c_title,
            x_axis=x_col,
            y_axis=y_col,
            data=points,
            description=f"Scatter distribution with {len(points)} plotted coordinates.",
            plotly_spec=plotly_figure,
            data_stage="analysis_ready",
            rows_used=sub_len,
            transformations=f"Mapped pairwise values across '{x_col}' and '{y_col}' (sampled to {len(points)})",
            exclusions=f"Excluded {len(df) - sub_len} rows with missing values" if (len(df) - sub_len) > 0 else None
        )

    # 3. Bar or Pie Chart (Grouped / Aggregated)
    is_y_numeric = bool(y_col and (y_col in df.columns) and pd.api.types.is_numeric_dtype(df[y_col]))
    if not y_col or y_col == x_col or not is_y_numeric:
        # Count frequency of x_col
        full_counts = df[x_col].value_counts(dropna=True)
        total_cats = len(full_counts)
        if total_cats == 0:
            raise ValueError(f"Column '{x_col}' has no valid values to chart.")

        # Sorting
        sort_ord = getattr(req, "sort_order", "desc") or "desc"
        if sort_ord == "asc":
            counts_sorted = full_counts.sort_values(ascending=True)
        elif sort_ord == "none":
            unique_ordered = [v for v in df[x_col].dropna().unique() if v in full_counts.index]
            counts_sorted = full_counts.reindex(unique_ordered)
        else:  # desc default
            counts_sorted = full_counts.sort_values(ascending=False)

        # Top N filtering
        top_n_val = req.top_n if req.top_n is not None else req.limit
        if top_n_val is not None and top_n_val > 0 and total_cats > top_n_val:
            display_series = counts_sorted.head(top_n_val)
            is_subset = True
            displayed_cats = len(display_series)
        else:
            display_series = counts_sorted
            is_subset = False
            displayed_cats = total_cats

        labels = [str(k) for k in display_series.index]
        values = [int(v) for v in display_series.values]
        data = [{"x": str(k), "y": int(v)} for k, v in display_series.items()]

        # Low-frequency diagnostic (e.g. restaurant names where most have count = 1)
        num_ones = int((full_counts == 1).sum())
        low_freq_note = None
        if num_ones >= 3 and (num_ones / total_cats >= 0.35):
            pct_ones = round((num_ones / total_cats) * 100, 1)
            low_freq_note = f"{num_ones} of {total_cats} categories ({pct_ones}%) have a frequency count of 1. Many categories share this low frequency."

        # Orientation selection: prefer horizontal for >= 6 categories
        req_orient = getattr(req, "orientation", "auto") or "auto"
        if req_orient in ["h", "v"]:
            chart_orientation = req_orient
        else:
            chart_orientation = "h" if displayed_cats >= 6 and c_type in ["bar", "histogram"] else "v"

        subset_tag = f" (Top {displayed_cats} of {total_cats})" if is_subset else ""
        c_title = f"Frequency Count of {x_col}{subset_tag}"

        plotly_figure = create_plotly_spec(
            chart_type=c_type if c_type in ["bar", "pie"] else "bar",
            title=c_title,
            x_axis=x_col,
            y_axis="Record Count",
            labels=labels,
            values=values,
            orientation=chart_orientation,
            is_subset=is_subset,
            total_categories=total_cats,
            displayed_categories=displayed_cats,
            low_frequency_note=low_freq_note
        )

        return ChartSpec(
            chart_type=c_type if c_type in ["bar", "pie"] else "bar",
            title=c_title,
            x_axis=x_col,
            y_axis="Record Count",
            labels=labels,
            data=data,
            description=f"{'Top ' + str(displayed_cats) + ' of ' if is_subset else 'All '}{total_cats} categories by occurrence frequency ({sort_ord}ending)." + (f" Note: {low_freq_note}" if low_freq_note else ""),
            plotly_spec=plotly_figure,
            data_stage="analysis_ready",
            rows_used=int(df[x_col].notna().sum()),
            transformations=f"Calculated frequency count on '{x_col}', sorted {sort_ord}ending{' (top ' + str(displayed_cats) + ' subset)' if is_subset else ''}",
            exclusions=f"Excluded {int(df[x_col].isna().sum())} missing values" if df[x_col].isna().any() else None,
            orientation=chart_orientation,
            is_subset=is_subset,
            total_categories=total_cats,
            displayed_categories=displayed_cats,
            sort_order=sort_ord,
            low_frequency_note=low_freq_note
        )
    else:
        # Aggregate numeric y_col by categorical x_col
        clean_sub = df.dropna(subset=[x_col, y_col])
        if len(clean_sub) == 0:
            raise ValueError(f"No non-null overlapping records found between '{x_col}' and '{y_col}'.")

        if agg == "sum":
            grouped_all = clean_sub.groupby(x_col)[y_col].sum()
        elif agg == "max":
            grouped_all = clean_sub.groupby(x_col)[y_col].max()
        elif agg == "min":
            grouped_all = clean_sub.groupby(x_col)[y_col].min()
        elif agg == "count":
            grouped_all = clean_sub.groupby(x_col)[y_col].count()
        else:
            grouped_all = clean_sub.groupby(x_col)[y_col].mean()

        total_cats = len(grouped_all)

        # Sorting
        sort_ord = getattr(req, "sort_order", "desc") or "desc"
        if sort_ord == "asc":
            grouped_sorted = grouped_all.sort_values(ascending=True)
        elif sort_ord == "none":
            grouped_sorted = grouped_all
        else:  # desc default
            grouped_sorted = grouped_all.sort_values(ascending=False)

        # Top N filtering
        top_n_val = req.top_n if req.top_n is not None else req.limit
        if top_n_val is not None and top_n_val > 0 and total_cats > top_n_val:
            display_grouped = grouped_sorted.head(top_n_val)
            is_subset = True
            displayed_cats = len(display_grouped)
        else:
            display_grouped = grouped_sorted
            is_subset = False
            displayed_cats = total_cats

        labels = [str(k) for k in display_grouped.index]
        values = [sanitize_val(v) for v in display_grouped.values]
        data = [{"x": str(k), "y": sanitize_val(v)} for k, v in display_grouped.items()]

        req_orient = getattr(req, "orientation", "auto") or "auto"
        if req_orient in ["h", "v"]:
            chart_orientation = req_orient
        else:
            chart_orientation = "h" if displayed_cats >= 6 and c_type in ["bar"] else "v"

        subset_tag = f" (Top {displayed_cats} of {total_cats})" if is_subset else ""
        c_title = f"{agg.capitalize()} of {y_col} by {x_col}{subset_tag}"

        plotly_figure = create_plotly_spec(
            chart_type=c_type if c_type in ["bar", "line", "pie"] else "bar",
            title=c_title,
            x_axis=x_col,
            y_axis=f"{agg.capitalize()} {y_col}",
            labels=labels,
            values=values,
            orientation=chart_orientation,
            is_subset=is_subset,
            total_categories=total_cats,
            displayed_categories=displayed_cats
        )

        return ChartSpec(
            chart_type=c_type if c_type in ["bar", "line", "pie"] else "bar",
            title=c_title,
            x_axis=x_col,
            y_axis=f"{agg.capitalize()} {y_col}",
            labels=labels,
            data=data,
            description=f"Calculated using pandas {agg}() aggregation over {len(clean_sub)} total rows ({sort_ord}ending).",
            plotly_spec=plotly_figure,
            data_stage="analysis_ready",
            rows_used=len(clean_sub),
            transformations=f"Grouped '{y_col}' by '{x_col}' using {agg}() and sorted {sort_ord}ending{' (top ' + str(displayed_cats) + ' subset)' if is_subset else ''}",
            exclusions=f"Excluded {len(df) - len(clean_sub)} rows with missing values" if (len(df) - len(clean_sub)) > 0 else None,
            orientation=chart_orientation,
            is_subset=is_subset,
            total_categories=total_cats,
            displayed_categories=displayed_cats,
            sort_order=sort_ord
        )

