from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from ragdx.schemas.models import DiagnosisReport, EvaluationResult, OptimizationPlan, OptimizationSession, SavedRun


class RunStore:
    def __init__(self, root: str | Path = ".ragdx"):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.sessions_dir = self.root / "optimization" / "sessions"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _run_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def save_run(
        self,
        evaluation: EvaluationResult,
        diagnosis: DiagnosisReport,
        plan: OptimizationPlan,
        name: str | None = None,
        tags: Optional[list[str]] = None,
        notes: str = "",
        baseline_run_id: str | None = None,
        latest_session_id: str | None = None,
    ) -> SavedRun:
        ts = datetime.now(timezone.utc).isoformat()
        run = SavedRun(
            run_id=uuid4().hex[:12],
            created_at=ts,
            name=name or evaluation.metadata.get("dataset", "unnamed-run"),
            tags=tags or [],
            notes=notes,
            baseline_run_id=baseline_run_id,
            latest_session_id=latest_session_id,
            evaluation=evaluation,
            diagnosis=diagnosis,
            optimization_plan=plan,
        )
        self._run_path(run.run_id).write_text(run.model_dump_json(indent=2), encoding="utf-8")
        return run

    def update_run_latest_session(self, run_id: str, session_id: str) -> SavedRun:
        run = self.load_run(run_id)
        run.latest_session_id = session_id
        self._run_path(run_id).write_text(run.model_dump_json(indent=2), encoding="utf-8")
        return run

    def load_run(self, run_id: str) -> SavedRun:
        return SavedRun.model_validate_json(self._run_path(run_id).read_text(encoding="utf-8"))

    def list_runs(self) -> List[SavedRun]:
        runs = []
        for path in sorted(self.runs_dir.glob("*.json"), reverse=True):
            try:
                runs.append(SavedRun.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    def latest(self) -> SavedRun | None:
        runs = self.list_runs()
        return runs[0] if runs else None

    def save_session(self, session: OptimizationSession) -> OptimizationSession:
        self._session_path(session.session_id).write_text(session.model_dump_json(indent=2), encoding="utf-8")
        return session

    def load_session(self, session_id: str) -> OptimizationSession:
        return OptimizationSession.model_validate_json(self._session_path(session_id).read_text(encoding="utf-8"))

    def list_sessions(self) -> List[OptimizationSession]:
        sessions = []
        for path in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            try:
                sessions.append(OptimizationSession.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return sorted(sessions, key=lambda s: s.created_at, reverse=True)

    def latest_session(self) -> OptimizationSession | None:
        sessions = self.list_sessions()
        return sessions[0] if sessions else None

    def export_markdown(self, run_id: str, output_path: str | Path) -> Path:
        run = self.load_run(run_id)
        lines = [
            f"# ragdx run report: {run.name}",
            "",
            f"- Run ID: `{run.run_id}`",
            f"- Created at: {run.created_at}",
            f"- Baseline run: `{run.baseline_run_id or 'none'}`",
            f"- Tags: {', '.join(run.tags) if run.tags else 'none'}",
            f"- Latest optimization session: `{run.latest_session_id or 'none'}`",
            "",
            "## Summary",
            run.diagnosis.summary,
            "",
            "## Retrieval metrics",
        ]
        for k, v in run.evaluation.retrieval.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("\n## Generation metrics")
        for k, v in run.evaluation.generation.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("\n## End-to-end metrics")
        for k, v in run.evaluation.e2e.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("\n## Hypotheses")
        for h in run.diagnosis.hypotheses:
            lines.append(f"- **{h.root_cause}** ({h.component}, severity={h.severity}, confidence={h.confidence:.2f})")
        lines.append("\n## Planned experiments")
        for e in run.optimization_plan.experiments:
            lines.append(f"- **{e.name}** via `{e.tool}` targeting `{e.target_component}`: {e.description}")
        if run.latest_session_id:
            try:
                session = self.load_session(run.latest_session_id)
                lines.append("\n## Latest optimization session")
                lines.append(f"- Session ID: `{session.session_id}`")
                lines.append(f"- Strategy: `{session.strategy}`")
                lines.append(f"- Mode: `{session.mode}`")
                lines.append(f"- Status: `{session.status}`")
                lines.append(f"- Completed trials: {session.completed_trials}/{session.total_trials}")
            except Exception:
                pass
        out = Path(output_path)
        out.write_text("\n".join(lines), encoding="utf-8")
        return out
