from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
from rich import print
from rich.table import Table

from ragdx.core.compare import compare_results
from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.core.evaluator import UnifiedEvaluator
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult
from ragdx.storage.run_store import RunStore

app = typer.Typer(add_completion=False)


def _load_eval(path: str | Path) -> EvaluationResult:
    with open(path, "r", encoding="utf-8") as f:
        return EvaluationResult(**json.load(f))


@app.command()
def diagnose(eval_json: str, save: bool = False, name: str = "", baseline_run_id: str = ""):
    result = _load_eval(eval_json)
    report = RAGDiagnosisEngine().diagnose(result)
    plan = OptimizationPlanner().build_plan(report)
    if save:
        run = RunStore().save_run(result, report, plan, name=name or None, baseline_run_id=baseline_run_id or None)
        print(f"Saved run: {run.run_id}")
    print(report.model_dump_json(indent=2))


@app.command()
def plan(eval_json: str):
    result = _load_eval(eval_json)
    report = RAGDiagnosisEngine().diagnose(result)
    plan = OptimizationPlanner().build_plan(report)
    print(plan.model_dump_json(indent=2))


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
def save(eval_json: str, name: str = "", tags: str = "", notes: str = "", baseline_run_id: str = ""):
    result = _load_eval(eval_json)
    report = RAGDiagnosisEngine().diagnose(result)
    plan = OptimizationPlanner().build_plan(report)
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
    table.add_column("Tags")
    for r in rows:
        table.add_row(r.run_id, r.created_at, r.name, r.baseline_run_id or "", ", ".join(r.tags))
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
