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
    if v is None or pd.isna(v):
        return None
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
    points: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Generate high quality Plotly figure specification matching the light warm teal/orange theme."""
    color_palette = [
        "#EA580C", "#0D9488", "#F59E0B", "#6366F1", "#EC4899", 
        "#14B8A6", "#8B5CF6", "#10B981", "#3B82F6", "#F97316"
    ]
    
    traces = []
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
    elif chart_type == "pie" and labels and values:
        traces.append({
            "type": "pie",
            "labels": labels,
            "values": values,
            "hole": 0.42,
            "marker": {"colors": color_palette},
            "textinfo": "label+percent",
            "hoverinfo": "label+value+percent"
        })
    elif chart_type == "line" and labels and values:
        traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "x": labels,
            "y": values,
            "line": {"color": "#EA580C", "width": 3, "shape": "spline"},
            "marker": {"color": "#0F3D3E", "size": 6},
            "name": y_axis or "Value"
        })
    else:  # default bar or histogram
        traces.append({
            "type": "bar",
            "x": labels if labels else [p.get("x") for p in (points or [])],
            "y": values if values else [p.get("y") for p in (points or [])],
            "marker": {
                "color": "#EA580C",
                "opacity": 0.9
            },
            "name": y_axis or "Value"
        })

    layout = {
        "title": {
            "text": title,
            "font": {"family": "Inter, sans-serif", "size": 15, "color": "#0F3D3E"}
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "color": "#0F3D3E"},
        "margin": {"t": 45, "r": 20, "b": 60, "l": 60},
        "xaxis": {
            "title": {"text": x_axis or "", "font": {"color": "#0F3D3E", "size": 12}},
            "tickfont": {"color": "#476D6D", "size": 11},
            "gridcolor": "#EFECE6",
            "showgrid": True,
            "zeroline": False
        },
        "yaxis": {
            "title": {"text": y_axis or "", "font": {"color": "#0F3D3E", "size": 12}},
            "tickfont": {"color": "#476D6D", "size": 11},
            "gridcolor": "#EFECE6",
            "showgrid": True,
            "zeroline": False
        },
        "hoverlabel": {
            "bgcolor": "#0F3D3E",
            "font": {"color": "#FFFFFF", "family": "Inter, sans-serif"}
        }
    }
    
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
        
        sub_df = df[[x_col, y_col]].dropna()
        if len(sub_df) > 500:
            sub_df = sub_df.sample(500, random_state=42)
            
        points = []
        for _, row in sub_df.iterrows():
            points.append({
                "x": sanitize_val(row[x_col]),
                "y": sanitize_val(row[y_col])
            })
            
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
            rows_used=len(sub_df),
            transformations=f"Mapped pairwise values across '{x_col}' and '{y_col}' (sampled to {len(points)})",
            exclusions=f"Excluded {len(df) - len(sub_df)} rows with missing values" if (len(df) - len(sub_df)) > 0 else None
        )

    # 3. Bar or Pie Chart (Grouped / Aggregated)
    if not y_col or y_col == x_col or not pd.api.types.is_numeric_dtype(df.get(y_col, pd.Series())):
        # Count frequency of x_col
        counts = df[x_col].value_counts().head(limit)
        labels = [str(k) for k in counts.index]
        values = [int(v) for v in counts.values]
        data = [{"x": str(k), "y": int(v)} for k, v in counts.items()]
        
        c_title = f"Frequency Count of {x_col}"
        plotly_figure = create_plotly_spec(
            chart_type=c_type if c_type in ["bar", "pie"] else "bar",
            title=c_title,
            x_axis=x_col,
            y_axis="Record Count",
            labels=labels,
            values=values
        )
        
        return ChartSpec(
            chart_type=c_type if c_type in ["bar", "pie"] else "bar",
            title=c_title,
            x_axis=x_col,
            y_axis="Record Count",
            labels=labels,
            data=data,
            description=f"Top {len(labels)} categories by occurrence frequency.",
            plotly_spec=plotly_figure,
            data_stage="analysis_ready",
            rows_used=int(df[x_col].notna().sum()),
            transformations=f"Calculated frequency count on '{x_col}', sorted descending (top {len(labels)})",
            exclusions=f"Excluded {int(df[x_col].isna().sum())} missing values" if df[x_col].isna().any() else None
        )
    else:
        # Aggregate numeric y_col by categorical x_col
        clean_sub = df.dropna(subset=[x_col, y_col])
        if agg == "sum":
            grouped = clean_sub.groupby(x_col)[y_col].sum()
        elif agg == "max":
            grouped = clean_sub.groupby(x_col)[y_col].max()
        elif agg == "min":
            grouped = clean_sub.groupby(x_col)[y_col].min()
        elif agg == "count":
            grouped = clean_sub.groupby(x_col)[y_col].count()
        else:
            grouped = clean_sub.groupby(x_col)[y_col].mean()

        grouped = grouped.sort_values(ascending=False).head(limit)
        labels = [str(k) for k in grouped.index]
        values = [sanitize_val(v) for v in grouped.values]
        data = [{"x": str(k), "y": sanitize_val(v)} for k, v in grouped.items()]
        
        c_title = f"{agg.capitalize()} of {y_col} by {x_col}"
        plotly_figure = create_plotly_spec(
            chart_type=c_type if c_type in ["bar", "line", "pie"] else "bar",
            title=c_title,
            x_axis=x_col,
            y_axis=f"{agg.capitalize()} {y_col}",
            labels=labels,
            values=values
        )
        
        return ChartSpec(
            chart_type=c_type if c_type in ["bar", "line", "pie"] else "bar",
            title=c_title,
            x_axis=x_col,
            y_axis=f"{agg.capitalize()} {y_col}",
            labels=labels,
            data=data,
            description=f"Calculated using pandas {agg}() aggregation over {len(clean_sub)} total rows.",
            plotly_spec=plotly_figure,
            data_stage="analysis_ready",
            rows_used=len(clean_sub),
            transformations=f"Grouped '{y_col}' by '{x_col}' using {agg}() and sorted descending",
            exclusions=f"Excluded {len(df) - len(clean_sub)} rows with missing values" if (len(df) - len(clean_sub)) > 0 else None
        )

