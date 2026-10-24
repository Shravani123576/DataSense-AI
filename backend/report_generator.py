import datetime
from typing import Dict, List, Any, Optional
from models import (
    ProfileReport,
    CleaningAuditRecord,
    AutoMLResponse,
    ExecutiveReportResponse
)


def generate_executive_report(
    profile: ProfileReport,
    audit_history: Optional[List[CleaningAuditRecord]] = None,
    ml_response: Optional[AutoMLResponse] = None,
    objective: Optional[str] = None
) -> ExecutiveReportResponse:
    """
    Synthesize end-to-end data science lifecycle into an executive briefing report (Markdown and HTML).
    """
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%B %d, %Y - %H:%M UTC")
    meta = profile.metadata
    health = profile.health
    obj_text = objective or f"Comprehensive exploratory profiling, auditable data hygiene, and predictive assessment for {meta.name}."

    audit_history = audit_history or []

    # Markdown Generation
    md_lines = [
        f"# Executive Data Science & Analytical Intelligence Brief",
        f"**Dataset Name**: {meta.name}  ",
        f"**Source Filename**: `{meta.filename}`  ",
        f"**Report Generated**: {now_str}  ",
        f"**Analytical Objective**: {obj_text}  ",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary",
        f"- **Dataset Scale**: {meta.rows:,} records across {meta.columns} distinct features.",
        f"- **Data Completeness**: {100 - meta.missing_percentage:.1f}% ({meta.missing_cells:,} missing cell values).",
        f"- **Health Rating**: **{health.grade} ({health.overall_score}/100)**.",
        f"- **Audit Transformation Log**: {len(audit_history)} auditable data-cleaning action(s) applied.",
    ]

    if ml_response:
        md_lines.append(f"- **Machine Learning Best Model**: **{ml_response.best_model_name}** ({ml_response.task_type.capitalize()}) with holdout test score of **{ml_response.leaderboard[0].test_score}**.")

    md_lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Data Health & Quality Audit",
        f"| Component | Score | Status Assessment |",
        f"| :--- | :--- | :--- |",
        f"| **Overall Health** | **{health.overall_score}/100** | Grade: **{health.grade}** |",
        f"| Completeness | {health.completeness_score}/100 | {meta.missing_cells:,} missing values across features |",
        f"| Uniqueness | {health.uniqueness_score}/100 | {meta.duplicate_rows:,} duplicate records detected |",
        f"| Outlier Health | {health.outlier_score}/100 | Checked against 1.5x IQR statistical threshold |",
        f"| Consistency | {health.consistency_score}/100 | Type conformity and casing regularity |",
        f"",
        f"### Key Diagnostic Findings & Recommendations:",
    ])
    for rec in health.recommendations:
        md_lines.append(f"- {rec}")

    # Cleaning Audit Section
    md_lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. Auditable Data Transformations Log",
    ])
    if audit_history:
        md_lines.extend([
            f"| Timestamp | Rule | Column | Affected Rows | Strategy | Rationale & Transformation |",
            f"| :--- | :--- | :--- | :--- | :--- | :--- |"
        ])
        for record in audit_history:
            col_display = f"`{record.column}`" if record.column else "Dataset-Wide"
            md_lines.append(
                f"| {record.timestamp} | {record.rule} | {col_display} | "
                f"{record.affected_rows:,} ({record.affected_percentage}%) | {record.strategy_used} | {record.transformation_summary} |"
            )
    else:
        md_lines.append("_No consequential cleaning actions were applied; analyzing original raw dataset partition._")

    # Feature Dictionary
    md_lines.extend([
        f"",
        f"---",
        f"",
        f"## 4. Feature Profile & Key Distributions",
        f"| Feature Name | Inferred Dtype | Missing % | Unique | Min / Max | Outlier Count |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- |"
    ])
    for col in profile.columns[:25]:
        min_max = f"{col.min} to {col.max}" if col.min is not None else "N/A"
        out_cnt = f"{col.outlier_count} ({col.outlier_percentage}%)" if col.outlier_count is not None else "-"
        md_lines.append(f"| `{col.column}` | {col.inferred_type} | {col.null_percentage}% | {col.unique_count:,} | {min_max} | {out_cnt} |")

    # Machine Learning Leaderboard (if executed)
    if ml_response:
        md_lines.extend([
            f"",
            f"---",
            f"",
            f"## 5. Automated Machine Learning (AutoML) Evaluation",
            f"- **Target Variable**: `{ml_response.target_column}`",
            f"- **Task Type**: {ml_response.task_type.capitalize()}",
            f"- **Evaluation Methodology**: Strict train/test split ({ml_response.train_samples:,} train / {ml_response.test_samples:,} test) with isolated scaling & imputation pipeline.",
            f"",
            f"### Model Tournament Leaderboard",
            f"| Model Architecture | Algorithm Type | Holdout Test Score | Train Score | Overfitting Gap | Training Time |",
            f"| :--- | :--- | :--- | :--- | :--- | :--- |"
        ])
        for item in ml_response.leaderboard:
            base_badge = " *(Baseline)*" if item.is_baseline else ""
            md_lines.append(
                f"| **{item.model_name}{base_badge}** | {item.algorithm} | **{item.test_score}** | "
                f"{item.train_score} | {item.overfitting_gap:+.4f} | {item.training_time_ms} ms |"
            )

        md_lines.extend([
            f"",
            f"### Feature Importance Ranking",
            f"| Rank | Feature | Importance | Relative Share |",
            f"| :--- | :--- | :--- | :--- |"
        ])
        for idx, f_item in enumerate(ml_response.feature_importance, 1):
            md_lines.append(f"| #{idx} | `{f_item.feature}` | {f_item.importance:.4f} | {f_item.percentage}% |")

        md_lines.extend([
            f"",
            f"**Generalization Assessment**: {ml_response.overfitting_assessment}",
        ])

    # Limitations & Actionable Next Steps
    recs = [
        "Incorporate regular automated data ingestion validation checks to capture null spikes early.",
        "Maintain auditable cleaning checkpoints in ETL pipelines before downstream consumption.",
        "Prioritize the top identified predictive features when engineering business alerts or dashboards.",
    ]
    limits = [
        "Statistical correlation does not establish business causation.",
        f"Evaluation performed on historical snapshot of {meta.rows:,} records.",
        "Model predictions should be paired with domain subject-matter expert review.",
    ]

    if ml_response and ml_response.limitations:
        limits.extend(ml_response.limitations)

    md_lines.extend([
        f"",
        f"---",
        f"",
        f"## 6. Actionable Recommendations & Next Steps",
    ])
    for r in recs:
        md_lines.append(f"1. {r}")

    md_lines.extend([
        f"",
        f"## 7. Analytical Caveats & Constraints",
    ])
    for lim in limits:
        md_lines.append(f"- {lim}")

    markdown_content = "\n".join(md_lines)

    # HTML Generation with executive styling matching #FAF8F5 / #0F3D3E / #EA580C
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>DataSense AI — Executive Briefing: {meta.name}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background-color: #FAF8F5;
      color: #1E293B;
      line-height: 1.6;
      padding: 40px 20px;
      margin: 0;
    }}
    .container {{
      max-width: 900px;
      margin: 0 auto;
      background: #FFFFFF;
      padding: 48px;
      border-radius: 16px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
      border: 1px solid #E2E8F0;
    }}
    .header {{
      border-bottom: 2px solid #EA580C;
      padding-bottom: 24px;
      margin-bottom: 32px;
    }}
    .badge {{
      display: inline-block;
      background: #EA580C;
      color: #FFFFFF;
      font-size: 12px;
      font-weight: 700;
      padding: 4px 12px;
      border-radius: 20px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    h1 {{
      color: #0F3D3E;
      font-size: 28px;
      margin: 12px 0 6px 0;
    }}
    .meta-text {{
      color: #64748B;
      font-size: 14px;
    }}
    .grid-cards {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin: 24px 0;
    }}
    .card {{
      background: #FAF8F5;
      padding: 16px;
      border-radius: 12px;
      border: 1px solid #E2E8F0;
      text-align: center;
    }}
    .card .num {{
      font-size: 24px;
      font-weight: 800;
      color: #0F3D3E;
    }}
    .card .label {{
      font-size: 12px;
      color: #64748B;
      text-transform: uppercase;
      font-weight: 600;
      margin-top: 4px;
    }}
    h2 {{
      color: #0F3D3E;
      font-size: 20px;
      border-left: 4px solid #EA580C;
      padding-left: 12px;
      margin-top: 36px;
      margin-bottom: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid #E2E8F0;
    }}
    th {{
      background: #FAF8F5;
      color: #0F3D3E;
      font-weight: 600;
    }}
    ul, ol {{
      padding-left: 20px;
    }}
    li {{
      margin-bottom: 6px;
    }}
    .footer {{
      margin-top: 48px;
      padding-top: 20px;
      border-top: 1px solid #E2E8F0;
      text-align: center;
      font-size: 12px;
      color: #94A3B8;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <span class="badge">DataSense AI V2</span>
      <h1>Executive Data Science Intelligence Brief</h1>
      <div class="meta-text">
        <strong>Dataset:</strong> {meta.name} ({meta.filename}) | 
        <strong>Generated:</strong> {now_str}
      </div>
      <p style="margin-top: 12px; color: #334155;"><strong>Objective:</strong> {obj_text}</p>
    </div>

    <div class="grid-cards">
      <div class="card">
        <div class="num">{meta.rows:,}</div>
        <div class="label">Total Records</div>
      </div>
      <div class="card">
        <div class="num">{meta.columns}</div>
        <div class="label">Features</div>
      </div>
      <div class="card">
        <div class="num">{health.grade}</div>
        <div class="label">Health Score ({health.overall_score}/100)</div>
      </div>
      <div class="card">
        <div class="num">{len(audit_history)}</div>
        <div class="label">Audit Actions</div>
      </div>
    </div>

    <h2>1. Data Health & Integrity Audit</h2>
    <table>
      <thead>
        <tr><th>Dimension</th><th>Score</th><th>Status Assessment</th></tr>
      </thead>
      <tbody>
        <tr><td><strong>Completeness</strong></td><td>{health.completeness_score}/100</td><td>{meta.missing_cells:,} missing cell(s)</td></tr>
        <tr><td><strong>Uniqueness</strong></td><td>{health.uniqueness_score}/100</td><td>{meta.duplicate_rows:,} duplicate row(s)</td></tr>
        <tr><td><strong>Outlier Health</strong></td><td>{health.outlier_score}/100</td><td>1.5x IQR boundary inspection</td></tr>
        <tr><td><strong>Consistency</strong></td><td>{health.consistency_score}/100</td><td>Data type conformity and text padding</td></tr>
      </tbody>
    </table>

    <h2>2. Auditable Transformation History</h2>
    {'<table><thead><tr><th>Time</th><th>Rule</th><th>Feature</th><th>Affected</th><th>Action Summary</th></tr></thead><tbody>' + ''.join([f'<tr><td>{r.timestamp}</td><td>{r.rule}</td><td>{r.column or "Dataset"}</td><td>{r.affected_rows:,}</td><td>{r.transformation_summary}</td></tr>' for r in audit_history]) + '</tbody></table>' if audit_history else '<p style="color:#64748B;">No auditable transformations applied.</p>'}

    {'<h2>3. Machine Learning (AutoML) Leaderboard</h2>' + f'<p>Target: <strong>{ml_response.target_column}</strong> | Task: <strong>{ml_response.task_type.capitalize()}</strong></p>' + '<table><thead><tr><th>Model</th><th>Algorithm</th><th>Holdout Test Score</th><th>Train Score</th><th>Overfit Gap</th></tr></thead><tbody>' + ''.join([f'<tr><td><strong>{m.model_name}</strong>{" (Baseline)" if m.is_baseline else ""}</td><td>{m.algorithm}</td><td><strong>{m.test_score}</strong></td><td>{m.train_score}</td><td>{m.overfitting_gap:+.4f}</td></tr>' for m in ml_response.leaderboard]) + '</tbody></table>' if ml_response else ''}

    <h2>4. Actionable Next Steps</h2>
    <ol>
      {''.join([f'<li>{r}</li>' for r in recs])}
    </ol>

    <h2>5. Constraints & Analytical Caveats</h2>
    <ul>
      {''.join([f'<li>{l}</li>' for l in limits])}
    </ul>

    <div class="footer">
      Generated securely by DataSense AI V2 &bull; Confidential Data Briefing
    </div>
  </div>
</body>
</html>
"""

    return ExecutiveReportResponse(
        dataset_id=profile.metadata.id,
        dataset_name=meta.name,
        objective=obj_text,
        generated_at=now_str,
        markdown_content=markdown_content,
        html_content=html_content,
        executive_summary=f"Analyzed {meta.rows:,} records with health grade {health.grade} ({health.overall_score}/100). {len(audit_history)} auditable transformations executed.",
        recommendations=recs,
        limitations=limits
    )
