from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import typer
from rich import print
from rich.table import Table

from ragdx.core.compare import compare_results
from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.core.evaluator import UnifiedEvaluator
from ragdx.engines.llm_diagnosis import LLMDiagnosisExplainer
from ragdx.optim.executor import OptimizationExecutor
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult
from ragdx.storage.run_store import RunStore

app = typer.Typer(add_completion=False)


def _load_eval(path: str | Path) -> EvaluationResult:
    with open(path, "r", encoding="utf-8") as f:
        return EvaluationResult(**json.load(f))


def _build_engine(use_llm: bool = False, use_both: bool = False) -> RAGDiagnosisEngine:
    if not use_llm and not use_both:
        return RAGDiagnosisEngine()
    try:
        from openai import OpenAI
    except Exception as exc:
        raise typer.BadParameter(
            "LLM diagnosis requires the openai extra. Install with: pip install -e '.[openai]'"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise typer.BadParameter("OPENAI_API_KEY is required when using --use-llm or --use-both.")

    model = os.environ.get("RAGDX_OPENAI_MODEL", "")
    client = OpenAI(api_key=api_key)

    def llm_callable(prompt: str) -> str:
        response = client.responses.create(model=model, input=prompt)
        return response.output_text

    return RAGDiagnosisEngine(llm_explainer=LLMDiagnosisExplainer(llm_callable=llm_callable))


def _diagnose_and_plan(
    result: EvaluationResult,
    use_llm: bool = False,
    use_both: bool = False,
    strategy: str = "bayesian",
    budget: int = 12,
):
    engine = _build_engine(use_llm=use_llm, use_both=use_both)
    report = engine.diagnose(result, use_llm=use_llm, use_both=use_both)
    plan = OptimizationPlanner().build_plan(report, result=result, strategy=strategy, budget=budget)
    return report, plan


@app.command()
def diagnose(
    eval_json: str,
    save: bool = False,
    name: str = "",
    baseline_run_id: str = "",
    use_llm: bool = typer.Option(False, help="Use LLM diagnosis instead of rule-based diagnosis."),
    use_both: bool = typer.Option(False, help="Run rule-based diagnosis, run LLM diagnosis, then summarize both with the LLM."),
):
    if use_llm and use_both:
        raise typer.BadParameter("Use either --use-llm or --use-both, not both.")
    result = _load_eval(eval_json)
    report, plan = _diagnose_and_plan(result, use_llm=use_llm, use_both=use_both)
    if save:
        run = RunStore().save_run(result, report, plan, name=name or None, baseline_run_id=baseline_run_id or None)
        print(f"Saved run: {run.run_id}")
    print(report.model_dump_json(indent=2))


@app.command()
def plan(
    eval_json: str,
    strategy: str = typer.Option("bayesian", help="Search strategy: bayesian or pareto_evolutionary."),
    budget: int = typer.Option(12, help="Trial budget to distribute across experiments."),
):
    result = _load_eval(eval_json)
    report, plan = _diagnose_and_plan(result, strategy=strategy, budget=budget)
    print(plan.model_dump_json(indent=2))


@app.command()
def optimize(
    eval_json: str,
    strategy: str = typer.Option("bayesian", help="Search strategy: bayesian or pareto_evolutionary."),
    budget: int = typer.Option(12, help="Trial budget to distribute across experiments."),
    mode: str = typer.Option("simulate", help="Execution mode: simulate, prepare_only, or execute."),
    save_run: bool = typer.Option(True, help="Save the run, diagnosis, plan, and optimization session."),
    name: str = "",
    use_llm: bool = typer.Option(False, help="Use LLM diagnosis instead of rule-based diagnosis."),
    use_both: bool = typer.Option(False, help="Run rule-based diagnosis, run LLM diagnosis, then summarize both with the LLM."),
):
    if use_llm and use_both:
        raise typer.BadParameter("Use either --use-llm or --use-both, not both.")
    if strategy not in {"bayesian", "pareto_evolutionary"}:
        raise typer.BadParameter("strategy must be bayesian or pareto_evolutionary")
    if mode not in {"simulate", "prepare_only", "execute"}:
        raise typer.BadParameter("mode must be simulate, prepare_only, or execute")

    result = _load_eval(eval_json)
    report, plan = _diagnose_and_plan(result, use_llm=use_llm, use_both=use_both, strategy=strategy, budget=budget)
    store = RunStore()
    run = None
    if save_run:
        run = store.save_run(result, report, plan, name=name or None)
    session = OptimizationExecutor().execute_plan(plan, baseline=result, strategy=strategy, mode=mode, run_id=run.run_id if run else None)
    store.save_session(session)
    if run is not None:
        store.update_run_latest_session(run.run_id, session.session_id)
    print(session.model_dump_json(indent=2))


@app.command()
def compare(current_eval_json: str, baseline_eval_json: str):
    current = _load_eval(current_eval_json)
    baseline = _load_eval(baseline_eval_json)
    comparisons = compare_results(current, baseline)
    table = Table(title="Metric comparison")
    table.add_column("Metric")
    table.add_column("Current")
    table.add_column("Baseline")
    table.add_column("Delta")
    table.add_column("Direction")
    for c in comparisons:
        table.add_row(c.metric, f"{c.current:.4f}", f"{c.baseline:.4f}", f"{c.delta:+.4f}", c.direction)
    print(table)


@app.command()
def save(
    eval_json: str,
    name: str = "",
    tags: str = "",
    notes: str = "",
    baseline_run_id: str = "",
    use_llm: bool = typer.Option(False, help="Use LLM diagnosis instead of rule-based diagnosis."),
    use_both: bool = typer.Option(False, help="Run rule-based diagnosis, run LLM diagnosis, then summarize both with the LLM."),
):
    if use_llm and use_both:
        raise typer.BadParameter("Use either --use-llm or --use-both, not both.")
    result = _load_eval(eval_json)
    report, plan = _diagnose_and_plan(result, use_llm=use_llm, use_both=use_both)
    run = RunStore().save_run(
        result,
        report,
        plan,
        name=name or None,
        tags=[x.strip() for x in tags.split(",") if x.strip()],
        notes=notes,
        baseline_run_id=baseline_run_id or None,
    )
    print(run.model_dump_json(indent=2))


@app.command()
def runs():
    store = RunStore()
    rows = store.list_runs()
    table = Table(title="Saved ragdx runs")
    table.add_column("Run ID")
    table.add_column("Created")
    table.add_column("Name")
    table.add_column("Baseline")
    table.add_column("Latest session")
    table.add_column("Tags")
    for r in rows:
        table.add_row(r.run_id, r.created_at, r.name, r.baseline_run_id or "", r.latest_session_id or "", ", ".join(r.tags))
    print(table)


@app.command()
def sessions():
    rows = RunStore().list_sessions()
    table = Table(title="Saved ragdx optimization sessions")
    table.add_column("Session ID")
    table.add_column("Created")
    table.add_column("Run ID")
    table.add_column("Strategy")
    table.add_column("Mode")
    table.add_column("Status")
    table.add_column("Progress")
    for s in rows:
        table.add_row(s.session_id, s.created_at, s.run_id or "", s.strategy, s.mode, s.status, f"{s.completed_trials}/{s.total_trials}")
    print(table)


@app.command()
def export_report(run_id: str, output_md: str):
    path = RunStore().export_markdown(run_id, output_md)
    print(f"Wrote {path}")


@app.command()
def normalize_tools(ragas_json: str = "", ragchecker_json: str = "", output_json: str = "normalized_evaluation.json"):
    evaluator = UnifiedEvaluator()
    ragas_scores = json.loads(Path(ragas_json).read_text(encoding="utf-8")) if ragas_json else None
    ragchecker_scores = json.loads(Path(ragchecker_json).read_text(encoding="utf-8")) if ragchecker_json else None
    result = evaluator.evaluate([], ragas_scores=ragas_scores, ragchecker_scores=ragchecker_scores)
    Path(output_json).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote {output_json}")


@app.command()
def dashboard():
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "ui" / "dashboard.py")], check=False)


if __name__ == "__main__":
    app()


@app.command()
def monitor_session(session_id: str, show_logs: bool = typer.Option(False, help="Show per-trial logs.")):
    session = RunStore().load_session(session_id)
    print(session.model_dump_json(indent=2) if show_logs else json.dumps({
        "session_id": session.session_id,
        "status": session.status,
        "strategy": session.strategy,
        "mode": session.mode,
        "completed_trials": session.completed_trials,
        "total_trials": session.total_trials,
        "best_trial_id": session.best_trial_id,
        "pareto_front_ids": session.pareto_front_ids,
    }, indent=2))
