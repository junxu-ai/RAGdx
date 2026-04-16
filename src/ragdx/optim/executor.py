from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import yaml

from ragdx.core.thresholds import LOWER_IS_BETTER
from ragdx.optim.autorag_adapter import AutoRAGAdapter
from ragdx.optim.dspy_adapter import DSPyAdapter
from ragdx.schemas.models import (
    EvaluationResult,
    ExecutionMode,
    OptimizationExperiment,
    OptimizationPlan,
    OptimizationSession,
    OptimizationTrial,
    SearchStrategy,
)
from ragdx.storage.run_store import RunStore


class OptimizationExecutor:
    def __init__(self, root: str | Path = ".ragdx"):
        self.root = Path(root)
        self.dspy = DSPyAdapter()
        self.autorag = AutoRAGAdapter()
        self.store = RunStore(self.root)

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _checkpoint(self, session: OptimizationSession) -> None:
        self.store.save_session(session)

    def _enumerate_candidates(self, search_space: Dict[str, List[Any]], limit: int) -> List[Dict[str, Any]]:
        keys = list(search_space.keys())
        values = [search_space[k] for k in keys]
        products = list(itertools.product(*values))
        candidates = [dict(zip(keys, combo)) for combo in products]
        if len(candidates) <= limit:
            return candidates
        rng = random.Random(7)
        rng.shuffle(candidates)
        return candidates[: limit * 4]

    def _hash_score(self, payload: Dict[str, Any]) -> float:
        text = json.dumps(payload, sort_keys=True)
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def _simulate_objectives(self, baseline: EvaluationResult, experiment: OptimizationExperiment, params: Dict[str, Any]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        noise = self._hash_score({"exp": experiment.name, "params": params})
        for metric, weight in experiment.objectives.items():
            base = baseline.score(metric, 0.65) or 0.65
            gain = 0.0
            if experiment.tool == "autorag":
                if metric in {"context_recall", "answer_correctness"}:
                    gain += 0.03 if params.get("retriever") == "hybrid" else 0.01
                    gain += 0.02 if params.get("top_k", 0) in {6, 8} else 0.0
                    gain += 0.015 if params.get("chunk_size", 0) in {384, 512, 768} else -0.005
                if metric in {"context_precision", "citation_accuracy"}:
                    gain += 0.025 if params.get("reranker") != "none" else -0.005
                    gain += 0.01 if params.get("context_ordering") != "retrieval_score" else 0.0
            elif experiment.tool == "dspy":
                if metric in {"faithfulness", "citation_accuracy", "answer_correctness"}:
                    gain += 0.03 if params.get("optimizer") == "MIPROv2" else 0.015
                    gain += 0.02 if params.get("prompt_style") in {"citation_first", "claim_then_evidence"} else 0.0
                    gain += 0.01 if params.get("decomposition") else 0.0
                if metric == "response_relevancy":
                    gain += 0.015 if params.get("fewshot_count", 0) in {2, 4} else 0.0
                    gain += 0.01 if params.get("temperature", 0.0) <= 0.2 else -0.005
            else:
                if metric == "answer_correctness":
                    gain += 0.025 if params.get("verifier") == "claim_checker" else 0.0
                    gain += 0.01 if params.get("retrieval_profile") == "balanced" else 0.0
                if metric in {"faithfulness", "context_recall", "citation_accuracy"}:
                    gain += 0.015 if params.get("generator_profile") == "citation_first" else 0.005

            perturb = (noise - 0.5) * 0.02
            candidate = base + gain + perturb
            if metric in LOWER_IS_BETTER:
                candidate = max(0.0, min(1.0, base - gain - perturb))
            else:
                candidate = max(0.0, min(1.0, candidate))
            metrics[metric] = round(candidate, 4)
        return metrics

    def _utility(self, metrics: Dict[str, float], objectives: Dict[str, float]) -> float:
        score = 0.0
        total = 0.0
        for metric, weight in objectives.items():
            val = metrics.get(metric, 0.0)
            if metric in LOWER_IS_BETTER:
                val = 1.0 - val
            score += weight * val
            total += weight
        return round(score / max(total, 1e-9), 4)

    def _dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        better_or_equal = True
        strictly_better = False
        for metric in set(a) | set(b):
            av = a.get(metric, 0.0)
            bv = b.get(metric, 0.0)
            if metric in LOWER_IS_BETTER:
                if av > bv:
                    better_or_equal = False
                if av < bv:
                    strictly_better = True
            else:
                if av < bv:
                    better_or_equal = False
                if av > bv:
                    strictly_better = True
        return better_or_equal and strictly_better

    def _pareto_front(self, trials: List[OptimizationTrial]) -> List[str]:
        front: List[str] = []
        done_trials = [t for t in trials if t.objective_scores]
        for trial in done_trials:
            dominated = False
            for other in done_trials:
                if other.trial_id == trial.trial_id:
                    continue
                if self._dominates(other.objective_scores, trial.objective_scores):
                    dominated = True
                    break
            if not dominated:
                front.append(trial.trial_id)
        return front

    def _sample_bayesian(self, candidates: List[Dict[str, Any]], trials: List[OptimizationTrial]) -> Dict[str, Any]:
        observed = {json.dumps(t.parameters, sort_keys=True): t for t in trials}
        remaining = [c for c in candidates if json.dumps(c, sort_keys=True) not in observed]
        if not remaining:
            return candidates[0]
        if not trials:
            return remaining[0]
        value_stats: Dict[tuple[str, Any], List[float]] = {}
        for t in trials:
            if t.utility is None:
                continue
            for k, v in t.parameters.items():
                value_stats.setdefault((k, v), []).append(t.utility)
        def score_candidate(candidate: Dict[str, Any]) -> float:
            total = 0.0
            count = 0
            for k, v in candidate.items():
                vals = value_stats.get((k, v), [])
                if vals:
                    total += sum(vals) / len(vals)
                    count += 1
            prior = self._hash_score(candidate) * 0.1
            return (total / count if count else 0.55) + prior
        ranked = sorted(remaining, key=score_candidate, reverse=True)
        return ranked[0]

    def _sample_pareto(self, candidates: List[Dict[str, Any]], trials: List[OptimizationTrial]) -> Dict[str, Any]:
        observed = {json.dumps(t.parameters, sort_keys=True): t for t in trials}
        remaining = [c for c in candidates if json.dumps(c, sort_keys=True) not in observed]
        if not remaining:
            return candidates[0]
        if len(trials) < 2:
            return remaining[0]
        front_ids = set(self._pareto_front(trials))
        front_trials = [t for t in trials if t.trial_id in front_ids]
        if not front_trials:
            return remaining[0]
        parent = random.Random(len(trials)).choice(front_trials)
        child = dict(parent.parameters)
        rng = random.Random(len(trials) * 17)
        mutable_keys = list(child.keys())
        for key in rng.sample(mutable_keys, k=max(1, min(2, len(mutable_keys)))):
            options = list({cand[key] for cand in candidates})
            child[key] = rng.choice(options)
        child_key = json.dumps(child, sort_keys=True)
        if child_key not in observed:
            return child
        return remaining[0]

    def _write_config(self, session_id: str, experiment: OptimizationExperiment, parameters: Dict[str, Any], trial_id: str) -> str:
        out_dir = self.root / "optimization" / session_id / "configs"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{trial_id}_{experiment.tool}.yaml"
        if experiment.tool == "autorag":
            payload = self.autorag.run(experiment, parameters).payload
        elif experiment.tool == "dspy":
            payload = self.dspy.run(experiment, parameters).payload
        else:
            payload = {
                "framework": "ragdx-manual",
                "objective_metric": "answer_correctness",
                "objectives": experiment.objectives,
                "search_parameters": parameters,
            }
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return str(path)

    def _runner_template(self, tool: str) -> str | None:
        mapping = {
            "dspy": os.environ.get("RAGDX_DSPY_RUNNER_CMD"),
            "autorag": os.environ.get("RAGDX_AUTORAG_RUNNER_CMD"),
            "manual": os.environ.get("RAGDX_MANUAL_RUNNER_CMD"),
        }
        return mapping.get(tool)

    def _parse_output_metrics(self, output_path: Path, objectives: Dict[str, float]) -> Dict[str, float]:
        if not output_path.exists():
            raise FileNotFoundError(f"Expected output file not found: {output_path}")
        data = json.loads(output_path.read_text(encoding="utf-8"))
        if "objective_scores" in data and isinstance(data["objective_scores"], dict):
            return {k: float(v) for k, v in data["objective_scores"].items()}
        metrics: Dict[str, float] = {}
        for bucket in ("retrieval", "generation", "e2e"):
            if isinstance(data.get(bucket), dict):
                metrics.update({k: float(v) for k, v in data[bucket].items()})
        if not metrics and isinstance(data.get("metrics"), dict):
            metrics = {k: float(v) for k, v in data["metrics"].items()}
        selected = {k: metrics[k] for k in objectives.keys() if k in metrics}
        if not selected:
            raise ValueError(f"No objective metrics found in output file {output_path}")
        return selected

    def _execute_external_trial(self, session_id: str, experiment: OptimizationExperiment, trial: OptimizationTrial, baseline: EvaluationResult) -> None:
        template = self._runner_template(experiment.tool)
        if not template:
            fallback = os.environ.get("RAGDX_FALLBACK_SIMULATE_ON_MISSING_RUNNER", "1")
            if fallback == "1":
                metrics = self._simulate_objectives(baseline, experiment, trial.parameters)
                trial.objective_scores = metrics
                trial.utility = self._utility(metrics, experiment.objectives)
                trial.status = "done"
                trial.logs.append(f"No external runner configured for tool={experiment.tool}; fell back to simulated scoring.")
                return
            trial.status = "failed"
            trial.logs.append(f"No runner template configured for tool={experiment.tool}. Set the corresponding RAGDX_*_RUNNER_CMD environment variable.")
            trial.notes = "Missing runner command template."
            return
        trial_dir = self.root / "optimization" / session_id / "outputs"
        trial_dir.mkdir(parents=True, exist_ok=True)
        output_path = trial_dir / f"{trial.trial_id}_metrics.json"
        log_path = trial_dir / f"{trial.trial_id}.log"
        command = template.format(
            config=trial.config_path,
            output=str(output_path),
            workdir=str(trial_dir),
            trial_id=trial.trial_id,
            session_id=session_id,
            tool=experiment.tool,
        )
        trial.runner_command = command
        trial.output_path = str(output_path)
        trial.log_path = str(log_path)
        trial.logs.append(f"Launching external runner: {command}")
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.run(shlex.split(command), stdout=lf, stderr=subprocess.STDOUT, text=True)
        trial.return_code = proc.returncode
        if proc.returncode != 0:
            trial.status = "failed"
            trial.logs.append(f"External runner failed with return code {proc.returncode}.")
            return
        metrics = self._parse_output_metrics(output_path, experiment.objectives)
        trial.objective_scores = metrics
        trial.utility = self._utility(metrics, experiment.objectives)
        trial.status = "done"
        trial.logs.append(f"Loaded objective scores from {output_path}: {metrics}")

    def _update_front_and_best(self, session: OptimizationSession) -> None:
        front = self._pareto_front(session.trials)
        session.pareto_front_ids = front
        for t in session.trials:
            t.pareto_front = t.trial_id in front
        completed_with_util = [t for t in session.trials if t.utility is not None]
        if completed_with_util:
            session.best_trial_id = max(completed_with_util, key=lambda t: t.utility or 0.0).trial_id

    def execute_plan(
        self,
        plan: OptimizationPlan,
        baseline: EvaluationResult,
        strategy: SearchStrategy = "bayesian",
        mode: ExecutionMode = "simulate",
        run_id: str | None = None,
    ) -> OptimizationSession:
        session_id = uuid4().hex[:12]
        total_trials = sum(exp.max_trials for exp in plan.experiments)
        initial_status = "running" if mode in {"simulate", "execute"} else "prepared"
        session = OptimizationSession(
            session_id=session_id,
            created_at=self._timestamp(),
            run_id=run_id,
            strategy=strategy,
            mode=mode,
            status=initial_status,
            plan=plan,
            total_trials=total_trials,
        )
        self._checkpoint(session)

        all_trials: List[OptimizationTrial] = []
        for exp in plan.experiments:
            session.current_experiment = exp.name
            self._checkpoint(session)
            candidates = self._enumerate_candidates(exp.search_space, exp.max_trials)
            exp_trials: List[OptimizationTrial] = []
            for _ in range(exp.max_trials):
                params = self._sample_bayesian(candidates, exp_trials) if exp.search_strategy == "bayesian" else self._sample_pareto(candidates, exp_trials)
                trial_id = uuid4().hex[:10]
                trial = OptimizationTrial(
                    trial_id=trial_id,
                    experiment_name=exp.name,
                    tool=exp.tool,
                    strategy=exp.search_strategy,
                    status="prepared" if mode == "prepare_only" else "running",
                    parameters=params,
                    started_at=self._timestamp(),
                )
                trial.config_path = self._write_config(session_id, exp, params, trial_id)
                trial.logs.append(f"Configuration written to {trial.config_path}")
                all_trials.append(trial)
                exp_trials.append(trial)
                session.trials = all_trials
                self._checkpoint(session)

                try:
                    if mode == "simulate":
                        metrics = self._simulate_objectives(baseline, exp, params)
                        trial.objective_scores = metrics
                        trial.utility = self._utility(metrics, exp.objectives)
                        trial.status = "done"
                        trial.logs.append(f"Simulated objective scores: {metrics}")
                    elif mode == "prepare_only":
                        trial.logs.append("Prepared configuration only. External execution is not started by ragdx in this mode.")
                    else:
                        self._execute_external_trial(session_id, exp, trial, baseline)
                except Exception as exc:
                    trial.status = "failed"
                    trial.notes = str(exc)
                    trial.logs.append(f"Execution error: {exc}")
                finally:
                    trial.completed_at = self._timestamp()
                    session.completed_trials += 1
                    self._update_front_and_best(session)
                    self._checkpoint(session)
            exp.status = "done" if mode in {"simulate", "execute"} else "planned"

        session.status = "completed" if mode in {"simulate", "execute"} else "prepared"
        session.current_experiment = None
        self._checkpoint(session)
        return session
