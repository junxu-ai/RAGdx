from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from ragdx.core.compare import compare_results
from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult, OptimizationSession
from ragdx.storage.run_store import RunStore


DEMO_RESULT = EvaluationResult(
    retrieval={"context_precision": 0.63, "context_recall": 0.57, "context_entities_recall": 0.54, "hit_rate_at_k": 0.64},
    generation={"faithfulness": 0.79, "response_relevancy": 0.82, "noise_sensitivity": 0.31, "context_utilization": 0.61, "hallucination": 0.19},
    e2e={"answer_correctness": 0.68, "citation_accuracy": 0.71, "user_success_rate": 0.69},
    metadata={"dataset": "demo", "tools": ["ragas", "ragchecker"]},
)


def _flatten(result: EvaluationResult) -> pd.DataFrame:
    rows = []
    for layer_name, metrics in [("retrieval", result.retrieval), ("generation", result.generation), ("e2e", result.e2e)]:
        for metric, value in metrics.items():
            rows.append({"layer": layer_name, "metric": metric, "score": value})
    return pd.DataFrame(rows)


def _trials_df(session: OptimizationSession) -> pd.DataFrame:
    rows = []
    for t in session.trials:
        row = {
            "trial_id": t.trial_id,
            "experiment": t.experiment_name,
            "tool": t.tool,
            "strategy": t.strategy,
            "status": t.status,
            "utility": t.utility,
            "pareto_front": t.pareto_front,
            "config_path": t.config_path,
        }
        row.update({f"metric::{k}": v for k, v in t.objective_scores.items()})
        row.update({f"param::{k}": v for k, v in t.parameters.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def load_result(uploaded_file) -> EvaluationResult:
    if uploaded_file is None:
        return DEMO_RESULT
    payload = json.load(uploaded_file)
    return EvaluationResult(**payload)


def main() -> None:
    st.set_page_config(page_title="ragdx dashboard", layout="wide")
    st.title("ragdx: RAG evaluation, diagnosis, and optimization")
    st.caption("Inspect normalized metrics, diagnose root causes, define optimization plans, and monitor optimization sessions. For live monitoring, keep this page open and use Refresh session after or during CLI optimization runs.")

    store = RunStore()
    uploaded = st.sidebar.file_uploader("Upload evaluation JSON", type=["json"])
    use_saved = st.sidebar.checkbox("Use latest saved run", value=False)

    if use_saved and store.latest() is not None:
        current_run = store.latest()
        result = current_run.evaluation
        report = current_run.diagnosis
        plan = current_run.optimization_plan
    else:
        result = load_result(uploaded)
        report = RAGDiagnosisEngine().diagnose(result)
        plan = OptimizationPlanner().build_plan(report, result=result)
        current_run = None

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Scores", "Diagnosis", "Optimization Plan", "Optimization Sessions", "Compare", "Raw JSON"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval metrics", len(result.retrieval))
        c2.metric("Generation metrics", len(result.generation))
        c3.metric("E2E metrics", len(result.e2e))
        df = _flatten(result)
        fig = px.bar(df, x="metric", y="score", color="layer", barmode="group", title="Metric overview")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader("Summary")
        st.write(report.summary)
        if plan.rationale:
            st.subheader("Why this plan was selected")
            for item in plan.rationale:
                st.write(f"- {item}")
        st.subheader("Metric gaps")
        if report.metric_gaps:
            gap_df = pd.DataFrame([{"metric": k, "gap": v} for k, v in report.metric_gaps.items()])
            st.dataframe(gap_df, use_container_width=True)
        else:
            st.info("No gaps against configured thresholds.")
        st.subheader("Hypotheses")
        for idx, hyp in enumerate(report.hypotheses, start=1):
            with st.expander(f"{idx}. {hyp.root_cause}", expanded=(idx == 1)):
                st.write(f"**Component:** {hyp.component}")
                st.write(f"**Severity:** {hyp.severity}")
                st.write(f"**Confidence:** {hyp.confidence:.2f}")
                st.write("**Evidence**")
                for item in hyp.evidence:
                    st.write(f"- {item}")
                st.write("**Recommended actions**")
                for item in hyp.recommended_actions:
                    st.write(f"- {item}")
        st.subheader("Priority actions")
        for item in report.priority_actions:
            st.write(f"- {item}")

    with tab3:
        if not plan.experiments:
            st.info("No optimization experiment proposed at the current thresholds.")
        else:
            for exp in plan.experiments:
                with st.expander(f"{exp.name} | {exp.tool} | {exp.search_strategy}", expanded=True):
                    st.write(exp.description)
                    st.write("**Objectives**")
                    st.json(exp.objectives)
                    st.write("**Search space**")
                    st.json(exp.search_space)
                    st.write(f"**Trial budget:** {exp.max_trials}")

    with tab4:
        refresh = st.button("Refresh session")
        sessions = store.list_sessions()
        if not sessions:
            st.info("No optimization sessions found. Run `ragdx optimize ...` first.")
        else:
            options = {f"{s.session_id} | {s.status} | {s.strategy}": s for s in sessions}
            selected = st.selectbox("Optimization session", list(options.keys()))
            session = options[selected]
            progress = session.completed_trials / max(session.total_trials, 1)
            st.progress(progress)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", session.status)
            c2.metric("Strategy", session.strategy)
            c3.metric("Trials", f"{session.completed_trials}/{session.total_trials}")
            c4.metric("Pareto front", len(session.pareto_front_ids))
            if session.best_trial_id:
                st.write(f"**Best trial by scalar utility:** `{session.best_trial_id}`")

            trials_df = _trials_df(session)
            if not trials_df.empty:
                st.dataframe(trials_df, use_container_width=True)
                metric_cols = [c for c in trials_df.columns if c.startswith("metric::")]
                if metric_cols:
                    metric_choice = st.selectbox("Metric to plot", metric_cols)
                    fig = px.line(trials_df.reset_index(), x="index", y=metric_choice, color="experiment", markers=True, title="Objective progression")
                    st.plotly_chart(fig, use_container_width=True)
                if "metric::answer_correctness" in trials_df.columns and "metric::citation_accuracy" in trials_df.columns:
                    fig2 = px.scatter(
                        trials_df,
                        x="metric::answer_correctness",
                        y="metric::citation_accuracy",
                        color="pareto_front",
                        symbol="experiment",
                        hover_data=["trial_id", "utility"],
                        title="Pareto view: answer_correctness vs citation_accuracy",
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                pareto_df = trials_df[trials_df["pareto_front"] == True]
                if not pareto_df.empty:
                    st.subheader("Pareto-front trials")
                    st.dataframe(pareto_df, use_container_width=True)
                selected_trial = st.selectbox("Inspect config", trials_df["trial_id"].tolist())
                row = trials_df[trials_df["trial_id"] == selected_trial].iloc[0]
                config_path = row.get("config_path")
                if isinstance(config_path, str) and config_path:
                    st.code(open(config_path, "r", encoding="utf-8").read(), language="yaml")
                log_path = row.get("log_path")
                if isinstance(log_path, str) and log_path:
                    st.subheader("Runner log")
                    try:
                        st.code(open(log_path, "r", encoding="utf-8").read(), language="text")
                    except Exception:
                        st.info("Log file not readable yet.")
                output_path = row.get("output_path")
                if isinstance(output_path, str) and output_path:
                    st.subheader("Runner output")
                    try:
                        st.code(open(output_path, "r", encoding="utf-8").read(), language="json")
                    except Exception:
                        st.info("Output file not readable yet.")

    with tab5:
        saved_runs = store.list_runs()
        if len(saved_runs) < 2 and current_run is None:
            st.info("Save two runs first, or use the CLI compare command for ad hoc comparison.")
        else:
            options = {f"{r.run_id} | {r.name}": r for r in saved_runs}
            selected_current = st.selectbox("Current run", list(options.keys())) if options else None
            selected_base = st.selectbox("Baseline run", list(options.keys()), index=1 if len(options) > 1 else 0) if options else None
            if selected_current and selected_base:
                current = options[selected_current]
                baseline = options[selected_base]
                cmp_df = pd.DataFrame([c.model_dump() for c in compare_results(current.evaluation, baseline.evaluation)])
                st.dataframe(cmp_df, use_container_width=True)

    with tab6:
        st.code(result.model_dump_json(indent=2), language="json")


if __name__ == "__main__":
    main()
