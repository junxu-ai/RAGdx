from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ragdx.core.compare import compare_results
from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult
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


def load_result(uploaded_file) -> EvaluationResult:
    if uploaded_file is None:
        return DEMO_RESULT
    payload = json.load(uploaded_file)
    return EvaluationResult(**payload)


def main() -> None:
    st.set_page_config(page_title="ragdx dashboard", layout="wide")
    st.title("ragdx: RAG evaluation, diagnosis, and optimization")
    st.caption("Inspect normalized metrics, compare saved runs, diagnose root causes, and stage optimization experiments.")

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
        plan = OptimizationPlanner().build_plan(report)
        current_run = None

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Scores", "Diagnosis", "Optimization", "Compare", "Raw JSON"])

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
            plan_df = pd.DataFrame([exp.model_dump() for exp in plan.experiments])
            st.dataframe(plan_df, use_container_width=True)

    with tab4:
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

    with tab5:
        st.code(result.model_dump_json(indent=2), language="json")


if __name__ == "__main__":
    main()
