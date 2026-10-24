import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from models import (
    PlanStep,
    AgentPlan,
    AgentGoalRequest,
    AgentGoalResponse,
    ProfileReport,
    AutoMLRequest,
    AutoMLResponse,
    CleaningActionRequest,
)
from data_profiler import profile_dataframe
from data_cleaner import generate_cleaning_recommendations, apply_cleaning_action
from ml_engine import train_automl_pipeline, detect_task_type
from report_generator import generate_executive_report

logger = logging.getLogger("datasense.agent_orchestrator")


def decompose_goal_to_plan(goal: str, df: pd.DataFrame) -> Tuple[AgentPlan, bool, Optional[str], Optional[List[str]]]:
    """
    Analyzes user goal and breaks it down into a verifiable, step-by-step execution plan.
    Checks for ambiguous requirements and flags clarification if essential details are missing.
    """
    g_lower = goal.lower()
    columns = list(df.columns)
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]

    steps: List[PlanStep] = []
    step_num = 1

    # Step 1: Always Profile and Diagnose
    steps.append(PlanStep(
        id=f"step-{step_num}",
        step_number=step_num,
        title="Profile Dataset & Audit Quality",
        tool_name="data_profiling",
        parameters={},
        status="pending"
    ))
    step_num += 1

    # Check for Modeling / Prediction intent
    wants_ml = any(w in g_lower for w in ["predict", "model", "forecast", "classify", "automl", "train", "regression"])
    target_identified = None

    if wants_ml:
        # Check if user named any specific column
        for c in columns:
            if c.lower() in g_lower:
                target_identified = c
                break

        if not target_identified:
            # Need clarification from user to know what to predict!
            # Select reasonable candidates (non-unique identifiers, key numerics or categories)
            candidates = [c for c in columns if df[c].nunique() > 1 and df[c].nunique() < len(df) * 0.95][:6]
            clarification_q = f"Which column should we target for prediction? Please select from the candidate features below."
            
            # Create the remaining steps assuming target will be provided
            steps.append(PlanStep(
                id=f"step-{step_num}",
                step_number=step_num,
                title="Identify Target & Clean Feature Matrix",
                tool_name="cleaning_audit",
                parameters={},
                status="pending"
            ))
            step_num += 1
            steps.append(PlanStep(
                id=f"step-{step_num}",
                step_number=step_num,
                title="Train AutoML Tournament & Compare Baselines",
                tool_name="train_automl",
                parameters={"target_column": None},
                status="pending"
            ))
            step_num += 1
            steps.append(PlanStep(
                id=f"step-{step_num}",
                step_number=step_num,
                title="Synthesize Executive Findings & Explanations",
                tool_name="generate_report",
                parameters={},
                status="pending"
            ))
            
            plan = AgentPlan(goal=goal, steps=steps, current_step_index=0, is_completed=False, status="paused_for_clarification")
            return plan, True, clarification_q, candidates

    # Check for Cleaning intent
    wants_cleaning = any(w in g_lower for w in ["clean", "fix", "impute", "deduplicate", "outlier", "preprocess"])
    if wants_cleaning or wants_ml:
        steps.append(PlanStep(
            id=f"step-{step_num}",
            step_number=step_num,
            title="Evaluate Auditable Cleaning Rules & Prepare Features",
            tool_name="cleaning_audit",
            parameters={"auto_dedup": True},
            status="pending"
        ))
        step_num += 1

    # Check for Correlation / Exploration / Specific Column Analysis
    if not wants_ml:
        steps.append(PlanStep(
            id=f"step-{step_num}",
            step_number=step_num,
            title="Compute Statistical Distributions & Key Relationships",
            tool_name="exploratory_analysis",
            parameters={"goal": goal},
            status="pending"
        ))
        step_num += 1

    # If user wanted ML and specified target:
    if wants_ml and target_identified:
        steps.append(PlanStep(
            id=f"step-{step_num}",
            step_number=step_num,
            title=f"Train AutoML Tournament for '{target_identified}'",
            tool_name="train_automl",
            parameters={"target_column": target_identified},
            status="pending"
        ))
        step_num += 1

    # Final Step: Synthesis & Reporting
    steps.append(PlanStep(
        id=f"step-{step_num}",
        step_number=step_num,
        title="Synthesize Findings, Model Explanations & Next Steps",
        tool_name="generate_report",
        parameters={},
        status="pending"
    ))

    plan = AgentPlan(goal=goal, steps=steps, current_step_index=0, is_completed=False, status="created")
    return plan, False, None, None


def execute_agent_workflow(
    df: pd.DataFrame,
    dataset_id: str,
    dataset_name: str,
    filename: str,
    plan: AgentPlan,
    clarification_response: Optional[str] = None
) -> Tuple[AgentPlan, List[str], List[Dict[str, Any]], str]:
    """
    Executes each planned step sequentially, logging intermediate evidence and handling failures gracefully.
    """
    findings: List[str] = []
    artifacts: List[Dict[str, Any]] = []
    current_df = df.copy()
    ml_result: Optional[AutoMLResponse] = None
    profile_result: Optional[ProfileReport] = None

    # If clarification provided, inject into step parameters
    if clarification_response:
        for s in plan.steps:
            if s.tool_name == "train_automl" and not s.parameters.get("target_column"):
                s.parameters["target_column"] = clarification_response.strip()
                s.title = f"Train AutoML Tournament for '{clarification_response.strip()}'"

    for idx, step in enumerate(plan.steps):
        plan.current_step_index = idx
        step.status = "running"
        t0 = time.perf_counter()

        try:
            if step.tool_name == "data_profiling":
                profile_result = profile_dataframe(current_df, dataset_id, dataset_name, filename)
                meta = profile_result.metadata
                health = profile_result.health
                step.result_summary = (
                    f"Profiled {meta.rows:,} rows and {meta.columns} columns. Overall Health Grade: {health.grade} ({health.overall_score}/100). "
                    f"Identified {meta.missing_cells:,} missing cells and {meta.duplicate_rows:,} duplicate rows."
                )
                findings.append(f"Data Health: Grade {health.grade} ({health.overall_score}/100) across {meta.rows:,} records.")
                step.status = "completed"

            elif step.tool_name == "cleaning_audit":
                recs = generate_cleaning_recommendations(current_df)
                dup_recs = [r for r in recs if r.rule == "deduplicate"]
                applied_count = 0
                if dup_recs and dup_recs[0].affected_rows > 0:
                    current_df, _ = apply_cleaning_action(
                        current_df,
                        CleaningActionRequest(dataset_id=dataset_id, rule="deduplicate", strategy="keep_first")
                    )
                    applied_count += 1
                
                step.result_summary = (
                    f"Diagnosed {len(recs)} potential cleaning opportunity/opportunities. "
                    f"Deduplicated dataset (removed exact redundant rows); preserved {len(current_df):,} distinct records."
                )
                findings.append(f"Cleaning Hygiene: Evaluated {len(recs)} rules; verified deduplication.")
                step.status = "completed"

            elif step.tool_name == "exploratory_analysis":
                # Compute notable correlations or distributions
                num_df = current_df.select_dtypes(include=["number"])
                if num_df.shape[1] >= 2:
                    corr = num_df.corr().abs()
                    unstack = corr.unstack()
                    unstack = unstack[unstack < 0.999].sort_values(ascending=False)
                    if len(unstack) > 0:
                        top_pair = unstack.index[0]
                        top_r = round(float(unstack.iloc[0]), 3)
                        findings.append(f"Strongest statistical relationship: '{top_pair[0]}' and '{top_pair[1]}' (r = {top_r}).")
                
                step.result_summary = f"Analyzed {len(num_df.columns)} numeric variables and {len(current_df.columns) - len(num_df.columns)} categorical attributes."
                step.status = "completed"

            elif step.tool_name == "train_automl":
                target = step.parameters.get("target_column")
                if not target or target not in current_df.columns:
                    raise ValueError(f"Target column '{target}' is not specified or does not exist in dataset.")

                req = AutoMLRequest(dataset_id=dataset_id, target_column=target, task_type="auto")
                ml_result = train_automl_pipeline(current_df, req)
                
                best_m = ml_result.best_model_name
                best_s = ml_result.leaderboard[0].test_score
                top_f = ml_result.feature_importance[0].feature if ml_result.feature_importance else "N/A"

                step.result_summary = (
                    f"Trained tournament with {len(ml_result.leaderboard)} models. Best Architecture: {best_m} "
                    f"(Holdout Test Score: {best_s}). Primary predictive driver: '{top_f}'."
                )
                findings.append(f"AutoML Victory: '{best_m}' achieved holdout test score of {best_s} for target '{target}'.")
                findings.append(f"Top Predictive Factor: '{top_f}' accounts for the highest feature influence.")
                artifacts.append({
                    "type": "automl_leaderboard",
                    "target": target,
                    "task_type": ml_result.task_type,
                    "best_model": best_m,
                    "best_score": best_s,
                    "features": [f.feature for f in ml_result.feature_importance[:5]]
                })
                step.status = "completed"

            elif step.tool_name == "generate_report":
                if not profile_result:
                    profile_result = profile_dataframe(current_df, dataset_id, dataset_name, filename)
                
                rep = generate_executive_report(
                    profile=profile_result,
                    audit_history=[],
                    ml_response=ml_result,
                    objective=plan.goal
                )
                step.result_summary = f"Synthesized executive briefing report ({len(rep.markdown_content)} characters) and recommendations."
                findings.append(f"Executive Briefing generated with {len(rep.recommendations)} actionable recommendations.")
                artifacts.append({
                    "type": "executive_report",
                    "title": "Executive Briefing",
                    "summary": rep.executive_summary
                })
                step.status = "completed"

        except Exception as e:
            logger.exception(f"Error in step {step.step_number} ({step.title}): {e}")
            step.status = "failed"
            step.error = str(e)
            step.result_summary = f"Step halted with error: {str(e)}"
            plan.status = "failed"
            break

        step.execution_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Check if all completed
    if all(s.status == "completed" for s in plan.steps):
        plan.is_completed = True
        plan.status = "completed"

    summary_explanation = (
        f"Executed {sum(1 for s in plan.steps if s.status == 'completed')}/{len(plan.steps)} planned operations for objective: '{plan.goal}'. "
        + (" ".join(findings))
    )

    return plan, findings, artifacts, summary_explanation


def process_agent_goal(df: pd.DataFrame, req: AgentGoalRequest, dataset_name: str, filename: str) -> AgentGoalResponse:
    """
    Main entry point for autonomous agent workflows.
    """
    t_start = time.perf_counter()
    
    # 1. Decompose into plan or detect missing parameters
    plan, needs_clarification, clar_q, clar_opts = decompose_goal_to_plan(req.goal, df)

    # If clarification is required and none provided yet
    if needs_clarification and not req.clarification_response:
        dur_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return AgentGoalResponse(
            goal=req.goal,
            dataset_id=req.dataset_id,
            plan=plan,
            summary_explanation="The goal requires clarification to proceed reliably without guessing. Please select the target feature.",
            findings=[],
            artifacts=[],
            requires_clarification=True,
            clarification_question=clar_q,
            clarification_options=clar_opts,
            follow_up_suggestions=[f"Predict {c}" for c in (clar_opts or [])[:3]],
            execution_time_ms=dur_ms
        )

    # If clarification was provided or not needed, execute
    updated_plan, findings, artifacts, summary = execute_agent_workflow(
        df=df,
        dataset_id=req.dataset_id,
        dataset_name=dataset_name,
        filename=filename,
        plan=plan,
        clarification_response=req.clarification_response
    )

    dur_ms = round((time.perf_counter() - t_start) * 1000, 2)
    return AgentGoalResponse(
        goal=req.goal,
        dataset_id=req.dataset_id,
        plan=updated_plan,
        summary_explanation=summary,
        findings=findings,
        artifacts=artifacts,
        requires_clarification=False,
        follow_up_suggestions=[
            "Show me the interactive What-If feature simulator",
            "Download full executive report",
            "Examine feature correlations"
        ],
        execution_time_ms=dur_ms
    )
