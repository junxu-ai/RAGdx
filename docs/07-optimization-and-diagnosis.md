# Diagnosis and Optimization Logic

## 1. Diagnosis pipeline

Diagnosis begins with an `EvaluationResult` and proceeds through one or more of the following layers:

1. rule-based analysis
2. causal graph construction and propagation
3. optional LLM refinement
4. optional rule plus LLM summary mode

## 2. Rule-based diagnosis

The rule-based analyzer uses metric levels, gaps, traces, and feedback to form hypotheses such as:

- corpus chunking defect
- retrieval recall defect
- retrieval precision defect
- context packing defect
- grounding defect
- citation binding defect
- judge or metric instability
- distribution shift

## 3. Layered causal graph

The diagnosis layer includes an explicit `CausalGraph` consisting of:

- `nodes`: `CausalSignal`
- `edges`: `CausalEdge`

The graph stores:

- priors
- posteriors
- evidence
- weighted causal relations

This supports more structured diagnosis than a flat list of heuristics.

## 4. Learned priors

The local store can maintain causal priors in:

- `.ragdx/causal/priors.json`

These priors can be updated from diagnosis reports and feedback, allowing some persistence across runs.

## 5. LLM diagnosis

When enabled, LLM diagnosis refines or summarizes the rule-based diagnosis. It should be understood as:

- a reasoning layer on top of structured evidence
- not a replacement for deterministic checks

The recommended production pattern is:

- run rule-based diagnosis by default
- use LLM diagnosis for ambiguous cases or richer summaries

## 6. Planning logic

The planner converts diagnosis into an `OptimizationPlan` with a set of experiments. Important planning ideas include:

- baseline-relative targets
- metric direction awareness
- stage-aware planning
- objective weights distinct from metric targets
- hard or near-hard constraints
- optional LLM planner refinement

## 7. Stages

The plan can include experiments in the following stages:

- `corpus`
- `retrieval`
- `generation`
- `orchestration`
- `joint`

Stage choice depends on the diagnosis.

## 8. Objectives vs targets vs constraints

This distinction is critical.

### Objective weights

Stored in `objectives` or `parameters.objective_weights`.

These are trade-off coefficients, not metric targets.

### Target thresholds and target specs

Stored in:

- `parameters.target_thresholds`
- `parameters.target_specs`

These specify what the optimizer should aim for relative to baseline.

### Constraint bounds

Stored in:

- `constraints`
- `parameters.constraint_bounds`

These define feasibility limits such as maximum hallucination or latency.

## 9. Multi-objective optimization

The optimizer handles multiple goals simultaneously, such as:

- answer correctness
- context recall
- context precision
- citation accuracy
- latency
- cost

This is important because real RAG changes often improve one dimension while hurting another.

## 10. Feasibility and Pareto logic

Each trial may have:

- `feasible`
- `constraint_violations`
- `feasibility_penalty`

Session state may include:

- `pareto_front_ids`
- `feasible_pareto_front_ids`
- `hypervolume`
- `feasible_hypervolume`

This allows the tool to separate globally strong candidates from candidates that are actually acceptable under constraints.

## 11. Search strategies

### Bayesian

A model-based search approach suitable when the search budget is limited and trial evaluations are expensive.

### Pareto evolutionary

A broader search approach suitable when trade-offs are complex and the search space is highly mixed or discrete.

## 12. Heavy BO backend

If configured, the system can attempt to use a heavier Bayesian optimization backend such as Ax.

## 13. LLM planner

The optional LLM planner takes into account:

- baseline metrics
- diagnosis summary
- metric gaps
- heuristic plan proposal

It can refine:

- objective weights
- target thresholds
- search-space focus
- trial budget
- rationale notes
