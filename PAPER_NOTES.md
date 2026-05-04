# Paper Notes: Social-Network Analysis of Conversational Agent Configurations for Process Mining Tasks

## Executive Summary

This repository implements and evaluates a selector-driven conversational multi-agent framework for answering process mining benchmark questions. The system is "agentic" in the sense that an LLM selector dynamically chooses among role-specialized persona agents before a final synthesis call. It is not a tool-using autonomous agent system, nor a fully decentralized multi-agent environment. The agents communicate indirectly through the accumulated chat transcript: each selected agent sees the original task and previous agent answers, then the final answer is produced from all prior contributions.

The main research scope is to test whether different agent configurations improve process mining task performance, and to use social network analysis (SNA) of the resulting agent traces to understand which agent roles and handoffs are productive. The process mining benchmark covers 57 tasks across eight categories, including event log interpretation, conformance and anomaly analysis, process model generation, querying, hypothesis generation, fairness, advanced process mining notations, and optimization.

The strongest observed configuration is `gpt-5.4-mini-ver-heavy-excl`, a verification-heavy setup with four underperforming handoff edges removed. Its final-answer judge mean is 6.981, compared with 6.718 for the no-agent baseline and 6.549 for the original verification-heavy configuration. However, the improvement over the no-agent baseline is modest and not robust under the simple paired normal approximation used here. Agentic configurations therefore appear useful as an experimental and diagnostic tool, but the current evidence does not show that agents are generally convenient as a default replacement for direct single-model answering.

Across the three balanced model runs with thinking disabled, `gpt-5.4-mini` clearly outperforms `grok-4.20` and `qwen3.6-35b-a3b` on final-answer quality. The social networks also differ strongly: the GPT run uses a sparse routing pattern with about 1.35 agent calls per question, while Grok and Qwen use substantially denser routing with lower node and final-answer scores. This suggests that more agent activity is not inherently better; routing selectivity is a central part of performance.

## Repository Evidence Used

The analysis in this document is based on:

- Framework implementation: `src/agents_sna/orchestrator.py`, `src/agents_sna/prompts.py`, `src/agents_sna/config.py`.
- Benchmark execution: `src/agents_sna/benchmark_runner.py`.
- LLM-as-judge evaluation: `src/agents_sna/agent_judge.py`.
- Social-network aggregation: `src/agents_sna/evaluation_network.py`.
- Agent configurations: `configs/*.json`.
- Benchmark artifacts: `benchmark_runs/<run>/requests`, `responses`, `traces`, `metadata`, and `summary.json`.
- Judge outputs and SNA summaries: `agent_evaluations/<run>/openaigpt-5.4/*.agent_evaluations.json` and `social_network_analysis.json`.

The final-answer metric used below is the `COMPLETE` node score assigned by the judge model `openai/gpt-5.4`. The SNA edge scores use the repository default, where a directed edge is scored with the target node's evaluation.

The generated social-network pictures are SVG files. Because every run uses the same basename, the full relative path is the useful identifier:

| Run | Social-network picture file |
|---|---|
| `gpt-5.4-mini-no-agent` | `agent_evaluations/gpt-5.4-mini-no-agent/openaigpt-5.4/social_network_analysis.svg` |
| `gpt-5.4-mini-artifact-pipeline` | `agent_evaluations/gpt-5.4-mini-artifact-pipeline/openaigpt-5.4/social_network_analysis.svg` |
| `gpt-5.4-mini-balanced` | `agent_evaluations/gpt-5.4-mini-balanced/openaigpt-5.4/social_network_analysis.svg` |
| `gpt-5.4-mini-verification-heavy` | `agent_evaluations/gpt-5.4-mini-verification-heavy/openaigpt-5.4/social_network_analysis.svg` |
| `gpt-5.4-mini-ver-heavy-excl` | `agent_evaluations/gpt-5.4-mini-ver-heavy-excl/openaigpt-5.4/social_network_analysis.svg` |
| `grok-4.20-balanced` | `agent_evaluations/grok-4.20-balanced/openaigpt-5.4/social_network_analysis.svg` |
| `qwen3.6-35b-a3b-balanced` | `agent_evaluations/qwen3.6-35b-a3b-balanced/openaigpt-5.4/social_network_analysis.svg` |

## Type of Agentic Framework

The implemented framework is best described as a centralized, selector-mediated conversational multi-agent framework.

Key properties:

- Conversational: all reasoning happens through chat-completion messages.
- Role-specialized: each agent is a named persona with a task description.
- Selector-driven: before each discussion step, a selector LLM decides which agent should contribute next, or returns `FINAL`.
- Sequential in the main process mining configurations: `single_agent_per_iteration` is `true`, so at most one agent is called per selector iteration.
- Transcript-mediated: agents do not directly message each other; they observe prior agent answers as assistant messages.
- Final synthesis based: the final answer is produced by a normal chat-completion call that sees the original prompt and prior agent answers.
- Non-tool-using: agents do not call external tools or manipulate a separate environment during benchmark execution.

The framework is therefore more conversational-deliberative than autonomous. It resembles a dynamic panel discussion with a moderator, not a swarm of independent agents with separate memory, tools, or environmental actions.

The main orchestration loop is:

1. Normalize the original prompt.
2. If agents exist, call the selector with available agents, previous answers, current iteration, and allowed handoffs.
3. Parse the selector response as JSON.
4. If the selector returns `FINAL`, stop discussion.
5. Otherwise, call the selected agent and append its answer to the shared transcript.
6. Repeat until `FINAL` or the iteration budget is exhausted.
7. Call the final synthesis prompt.

The no-agent baseline bypasses selector and agent discussion entirely and sends the original prompt directly to the final-answer call.

## Research Scope

The repository is scoped around the following empirical question:

> Do role-specialized conversational agents, and particular routing configurations among them, improve answer quality on process mining benchmark tasks compared with a direct no-agent baseline?

Secondary questions are:

- Which role decompositions work better for process mining tasks?
- Which agent roles are consistently useful, and which are weak?
- Which handoff edges are associated with low-quality downstream contributions?
- Can low-scoring handoffs be removed to improve the social network and final answers?
- Does the same agent configuration behave differently across base LLMs?
- Does increased agent activity justify its additional calls, latency, and cost?

This is not primarily a new process mining algorithm. It is an evaluation of LLM orchestration patterns for process mining question answering, with social-network analysis used as an explanatory layer over multi-agent traces.

## Experimental Setup

### Benchmark Corpus

The benchmark artifacts cover 57 questions from a local PM-LLM-Benchmark checkout. The question filenames reveal eight categories:

| Category | Count | Representative scope inferred from filenames |
|---|---:|---|
| `cat01` | 8 | Case ID inference, activity context, log construction, tables to logs |
| `cat02` | 9 | Conformance checking, anomaly detection, root cause analysis |
| `cat03` | 8 | Process tree, POWL, DECLARE, log skeleton, Petri net, temporal profile, discovery |
| `cat04` | 7 | Model descriptions, open questions, SQL-style filtering |
| `cat05` | 7 | Hypothesis and diagnostic question generation |
| `cat06` | 7 | Bias, fairness, mitigation, group comparisons |
| `cat07` | 6 | Object-centric and advanced process mining representations |
| `cat08` | 5 | Queue mining, instance spanning, transport optimization, resource assignment, scheduling |

### Common Execution Mechanics

All benchmark runs use OpenRouter-compatible chat completions. Each run writes:

- Full request artifacts under `benchmark_runs/<run>/requests`.
- Responses under `benchmark_runs/<run>/responses`.
- Short traces under `benchmark_runs/<run>/traces`.
- Metadata under `benchmark_runs/<run>/metadata`.
- A run summary under `benchmark_runs/<run>/summary.json`.

The LLM-as-judge stage reads saved request/response artifacts and emits one JSON list per question. For agentic runs, the judged nodes are:

- `START`: the first selector decision.
- One node per selected agent response.
- `COMPLETE`: the final synthesis response.

Later selector calls are not included as separate judged nodes. This is important: the SNA graph represents the judged contribution sequence, not every LLM API call.

The social network aggregation then computes:

- Node count, mean score, and population standard deviation by `agent_type`.
- Directed edge count, mean score, and population standard deviation for consecutive judged nodes.
- Average number of edge instantiations per evaluated question.
- Agent usage by benchmark category.

### Evaluation Caution

The benchmark quality metric is judge-based, not a human gold-standard score. The judge model is `openai/gpt-5.4`, and it scores nodes from 1.0 to 10.0 using a strict process-mining rubric. These results are useful for exploratory analysis, but a paper should describe them as initial or proxy results unless validated against independent human or reference-based evaluation.

The model-comparison condition states that `gpt-5.4-mini`, `grok-4.20`, and `qwen3.6-35b-a3b` were run with thinking disabled. The run summaries record model IDs but do not preserve the full OpenRouter `additional_payload`, so the exact disabled-thinking payload should be documented separately in the experiment log.

## Configurations Tried

### `no_agents`

File: `configs/no_agents.json`

- `max_iterations`: 1
- `agents`: none
- Purpose: direct single-model baseline.
- Behavior: no selector calls, no agent calls, one final-answer call per question.

### `pm_benchmark_artifact_pipeline`

File: `configs/pm_benchmark_artifact_pipeline.json`

- `max_iterations`: 50
- `single_agent_per_iteration`: true
- `excluded_handoffs`: none
- Intent: pipeline-like decomposition around artifact interpretation and answer auditing.

Agents:

| Agent | Role |
|---|---|
| `schema_and_artifact_mapper` | Extracts event log columns, objects, resources, timestamps, rules, schemas, and requested outputs |
| `trace_and_variant_reasoner` | Reconstructs variants, loops, case notions, and trace-level behavior |
| `control_flow_semantics_expert` | Checks procedural and declarative semantics |
| `performance_and_resource_analyst` | Identifies bottlenecks, handoffs, workload, and resource patterns |
| `data_question_designer` | Generates hypotheses, SQL-style investigations, filters, and evidence plans |
| `risk_fairness_and_compliance_reviewer` | Reviews fairness, risk, auditability, privacy, and side effects |
| `answer_precision_auditor` | Checks completeness, terminology, grounding, and unsupported claims |

Only five of these agents appear in the judged SNA node table for the stored evaluations. `performance_and_resource_analyst` and `data_question_designer` were not selected in the evaluated contribution sequence.

### `pm_benchmark_balanced`

File: `configs/pm_benchmark_balanced.json`

- `max_iterations`: 50
- `single_agent_per_iteration`: true
- `excluded_handoffs`: none
- Intent: broad process mining role coverage balanced across common task types.

Agents:

| Agent | Role |
|---|---|
| `event_log_interpreter` | Event logs, traces, timestamps, resources, missing case IDs |
| `process_modeler` | Process trees, POWL, DECLARE, log skeletons, temporal profiles, Petri nets |
| `conformance_anomaly_checker` | Deviations, mandatory steps, forbidden orders, unusual patterns |
| `query_and_sql_analyst` | SQL, querying, filtering, grouping, constraints |
| `hypothesis_generator` | Hypotheses, diagnostic questions, verification strategies |
| `fairness_and_ethics_reviewer` | Bias, protected groups, disparate impact, mitigation |
| `optimization_consultant` | Bottlenecks, rework, capacity, wait times, redesign |

This configuration was run with three base models:

- `openai/gpt-5.4-mini`
- `x-ai/grok-4.20`
- `qwen/qwen3.6-35b-a3b`

### `pm_benchmark_verification_heavy`

File: `configs/pm_benchmark_verification_heavy.json`

- `max_iterations`: 50
- `single_agent_per_iteration`: true
- `excluded_handoffs`: none
- Intent: emphasize parsing, verification, formal checking, adversarial review, and final scoring awareness.

Agents:

| Agent | Role |
|---|---|
| `artifact_parser` | Extracts exact entities, activities, timestamps, rules, schemas, and deliverables |
| `domain_reasoner` | Builds the primary process mining answer |
| `formal_semantics_checker` | Checks formal process model syntax and semantics |
| `counterexample_hunter` | Searches for counterexamples, edge cases, and overgeneralizations |
| `data_consistency_auditor` | Checks case IDs, attributes, resources, ordering, SQL/table logic |
| `bias_and_impact_auditor` | Reviews fairness, protected attributes, intervention side effects |
| `benchmark_answer_judge` | Predicts whether the answer would score well under the judge |

### `pm_benchmark_verification_heavy_excluding_low_edges`

File: `configs/pm_benchmark_verification_heavy_excluding_low_edges.json`

This configuration keeps the verification-heavy agent set but disables four handoffs:

| Excluded edge | Original count | Original mean edge score |
|---|---:|---:|
| `formal_semantics_checker -> domain_reasoner` | 3 | 4.967 |
| `artifact_parser -> benchmark_answer_judge` | 3 | 6.467 |
| `artifact_parser -> formal_semantics_checker` | 6 | 6.667 |
| `domain_reasoner -> benchmark_answer_judge` | 4 | 6.925 |

The first edge was clearly weak in the original SNA. The other three were lower or moderate rather than catastrophic. The experiment tests whether pruning these directed transitions improves downstream routing and final answers.

## Overall Results

All listed runs have 57 judged evaluation files, even where `summary.json` currently reports many questions as `skipped_existing_response`. That status means the benchmark runner found pre-existing answer files during a later invocation, not that the evaluation is missing.

| Run | Model | Config | Judged questions | Final mean | Final sd | Avg LLM calls/question | Avg agent calls/question | Avg judged edges/question | Graph edges |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `gpt-5.4-mini-no-agent` | `openai/gpt-5.4-mini` | no agents | 57 | 6.718 | 2.158 | 1.000 | 0.000 | 0.000 | 0 |
| `gpt-5.4-mini-artifact-pipeline` | `openai/gpt-5.4-mini` | artifact pipeline | 57 | 6.737 | 2.126 | 5.895 | 1.947 | 2.947 | 14 |
| `gpt-5.4-mini-balanced` | `openai/gpt-5.4-mini` | balanced | 57 | 6.746 | 1.812 | 4.702 | 1.351 | 2.351 | 29 |
| `gpt-5.4-mini-verification-heavy` | `openai/gpt-5.4-mini` | verification heavy | 57 | 6.549 | 1.994 | 6.316 | 2.158 | 3.158 | 25 |
| `gpt-5.4-mini-ver-heavy-excl` | `openai/gpt-5.4-mini` | verification heavy with excluded edges | 57 | 6.981 | 1.699 | 6.140 | 2.070 | 3.070 | 20 |
| `grok-4.20-balanced` | `x-ai/grok-4.20` | balanced | 57 | 5.325 | 2.160 | 11.684 | 4.842 | 5.842 | 54 |
| `qwen3.6-35b-a3b-balanced` | `qwen/qwen3.6-35b-a3b` | balanced | 57 | 5.388 | 1.829 | 8.421 | 3.211 | 4.211 | 48 |

Main observations:

- The best raw final-answer mean is `gpt-5.4-mini-ver-heavy-excl` at 6.981.
- The no-agent baseline is strong at 6.718.
- The artifact-pipeline and balanced GPT configurations barely improve over no-agent in raw mean.
- The original verification-heavy configuration underperforms no-agent.
- Grok and Qwen balanced runs score far below the GPT balanced run while using many more LLM calls.
- More agent calls do not imply better final answers. The two densest balanced networks, Grok and Qwen, are also the weakest model runs.

## Paired Comparisons Against No-Agent Baseline

The table below compares GPT-based agent configurations against the GPT no-agent baseline on the same 57 questions. The `p approx` column is a simple two-sided normal approximation on paired differences. It should be treated as descriptive, not as a definitive statistical test.

| Agent configuration minus no-agent | Mean difference | Wins | Losses | Ties | p approx |
|---|---:|---:|---:|---:|---:|
| `gpt-5.4-mini-artifact-pipeline` | +0.019 | 24 | 27 | 6 | 0.929 |
| `gpt-5.4-mini-balanced` | +0.028 | 22 | 30 | 5 | 0.867 |
| `gpt-5.4-mini-verification-heavy` | -0.168 | 19 | 32 | 6 | 0.422 |
| `gpt-5.4-mini-ver-heavy-excl` | +0.263 | 27 | 27 | 3 | 0.155 |

The only GPT agent configuration with a meaningful raw improvement is `gpt-5.4-mini-ver-heavy-excl`, but even there the win/loss count is balanced and the paired approximation does not support a strong significance claim. This is an important result: the current data do not justify a blanket claim that agents are better than direct answering.

## Category-Level Final Answer Results

| Run | cat01 | cat02 | cat03 | cat04 | cat05 | cat06 | cat07 | cat08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpt-5.4-mini-no-agent` | 7.300 | 8.167 | 4.450 | 7.014 | 7.929 | 6.529 | 4.067 | 8.140 |
| `gpt-5.4-mini-artifact-pipeline` | 7.438 | 8.044 | 5.763 | 6.371 | 7.114 | 7.400 | 3.300 | 8.000 |
| `gpt-5.4-mini-balanced` | 6.975 | 7.744 | 4.612 | 6.914 | 7.943 | 6.800 | 5.083 | 8.000 |
| `gpt-5.4-mini-verification-heavy` | 7.150 | 7.700 | 4.275 | 7.471 | 7.000 | 6.886 | 3.800 | 8.060 |
| `gpt-5.4-mini-ver-heavy-excl` | 7.837 | 8.011 | 6.062 | 6.971 | 7.143 | 6.900 | 4.350 | 8.280 |
| `grok-4.20-balanced` | 4.800 | 6.078 | 4.800 | 5.600 | 6.329 | 5.971 | 3.233 | 5.460 |
| `qwen3.6-35b-a3b-balanced` | 5.325 | 6.044 | 4.075 | 5.786 | 5.971 | 6.243 | 3.417 | 6.200 |

Category observations:

- `cat03` and `cat07` are consistently difficult. These categories include formal model generation/discovery and advanced process mining representations, where exact semantics matter.
- `cat08` is consistently strong for GPT runs, including the no-agent baseline.
- `gpt-5.4-mini-ver-heavy-excl` improves strongly over the original verification-heavy run on `cat03` (+1.787), `cat01` (+0.687), `cat07` (+0.550), and `cat08` (+0.220), but loses on `cat04` (-0.500).
- The no-agent baseline remains best or near-best on `cat02`, `cat04`, `cat05`, and `cat08`, which limits the case for agents as a default.
- Grok and Qwen underperform GPT in every category under the balanced configuration.

## Node-Level Social Network Results

### GPT Artifact Pipeline

Social-network picture: `agent_evaluations/gpt-5.4-mini-artifact-pipeline/openaigpt-5.4/social_network_analysis.svg`.

| Node | Count | Mean | sd | Interpretation |
|---|---:|---:|---:|---|
| `answer_precision_auditor` | 56 | 7.529 | 1.708 | Strong and heavily used; the main successful node in this configuration |
| `risk_fairness_and_compliance_reviewer` | 5 | 7.400 | 1.249 | Strong but category-limited, mainly fairness tasks |
| `control_flow_semantics_expert` | 7 | 7.157 | 1.968 | Useful on formal/control-flow tasks, but variable |
| `schema_and_artifact_mapper` | 41 | 6.580 | 1.531 | Frequently used first-stage parser, moderate score |
| `trace_and_variant_reasoner` | 2 | 5.350 | 2.750 | Weak but too rare for strong conclusions |
| `COMPLETE` | 57 | 6.737 | 2.126 | Final-answer quality close to no-agent |

This configuration behaves like a parse-then-audit pipeline. The edge `schema_and_artifact_mapper -> answer_precision_auditor` appears 37 times with mean 7.727, and `answer_precision_auditor -> COMPLETE` appears 55 times. The configuration depends heavily on the auditor. It performs well on fairness and several event-log categories but poorly on `cat07`.

Underperforming edges with count at least 2 include:

- `START -> trace_and_variant_reasoner`: count 2, mean 5.350.
- `schema_and_artifact_mapper -> control_flow_semantics_expert`: count 3, mean 5.400.
- `START -> answer_precision_auditor`: count 5, mean 5.800.
- `trace_and_variant_reasoner -> answer_precision_auditor`: count 2, mean 5.850.

### GPT Balanced

Social-network picture: `agent_evaluations/gpt-5.4-mini-balanced/openaigpt-5.4/social_network_analysis.svg`.

| Node | Count | Mean | sd | Interpretation |
|---|---:|---:|---:|---|
| `hypothesis_generator` | 6 | 7.817 | 0.825 | Best node, especially relevant for `cat05` |
| `query_and_sql_analyst` | 7 | 7.314 | 1.327 | Strong on query/question-generation tasks |
| `conformance_anomaly_checker` | 12 | 7.217 | 1.195 | Strong on conformance and anomaly tasks |
| `fairness_and_ethics_reviewer` | 7 | 7.143 | 1.278 | Strong and targeted to fairness category |
| `optimization_consultant` | 10 | 6.770 | 1.714 | Moderate; good final transitions but mixed node quality |
| `event_log_interpreter` | 13 | 6.746 | 1.639 | Moderate; useful for event logs but not clearly superior |
| `process_modeler` | 22 | 6.236 | 1.980 | Main underperforming node; high use and low mean |
| `COMPLETE` | 57 | 6.746 | 1.812 | Slightly above no-agent but practically tied |

The balanced GPT network is relatively sparse, with 1.351 agent calls per question. The selector often chooses a single specialized agent and then moves to final synthesis.

Underperforming edges with count at least 2 include:

- `START -> COMPLETE`: count 2, mean 4.850.
- `process_modeler -> optimization_consultant`: count 2, mean 5.100.
- `process_modeler -> COMPLETE`: count 15, mean 5.873.
- `event_log_interpreter -> COMPLETE`: count 9, mean 6.233.
- `START -> process_modeler`: count 18, mean 6.261.

The process-modeling role is both important and weak. Since `cat03` is one of the hardest categories, this suggests that future configurations should improve the formal modeling agent, add stricter syntax checks, or use dedicated process model validation.

### GPT Verification Heavy

Social-network picture: `agent_evaluations/gpt-5.4-mini-verification-heavy/openaigpt-5.4/social_network_analysis.svg`.

| Node | Count | Mean | sd | Interpretation |
|---|---:|---:|---:|---|
| `counterexample_hunter` | 10 | 8.160 | 0.539 | Best node; adversarial review is useful |
| `bias_and_impact_auditor` | 4 | 7.825 | 0.521 | Strong but category-limited |
| `data_consistency_auditor` | 7 | 7.786 | 0.783 | Strong consistency-checking contribution |
| `formal_semantics_checker` | 12 | 7.025 | 1.434 | Moderate; useful but not enough to lift final score |
| `domain_reasoner` | 28 | 6.932 | 1.320 | Frequently used, moderate |
| `artifact_parser` | 55 | 6.885 | 1.793 | Nearly always first, moderate |
| `benchmark_answer_judge` | 7 | 6.729 | 0.997 | Lowest non-special node |
| `COMPLETE` | 57 | 6.549 | 1.994 | Worse than no-agent |

This network almost always begins with `artifact_parser` (`START -> artifact_parser`, count 55). The strongest specialized nodes are review-oriented: `counterexample_hunter`, `data_consistency_auditor`, and `bias_and_impact_auditor`. However, the final answer score is lower than no-agent. This suggests the extra verification process sometimes adds friction or fails to translate into better synthesis.

Underperforming edges with count at least 2 include:

- `formal_semantics_checker -> domain_reasoner`: count 3, mean 4.967.
- `formal_semantics_checker -> COMPLETE`: count 6, mean 5.467.
- `artifact_parser -> COMPLETE`: count 13, mean 6.100.
- `domain_reasoner -> COMPLETE`: count 12, mean 6.383.
- `artifact_parser -> benchmark_answer_judge`: count 3, mean 6.467.
- `artifact_parser -> formal_semantics_checker`: count 6, mean 6.667.

### GPT Verification Heavy With Excluded Low Edges

Social-network picture: `agent_evaluations/gpt-5.4-mini-ver-heavy-excl/openaigpt-5.4/social_network_analysis.svg`.

| Node | Count | Mean | sd | Interpretation |
|---|---:|---:|---:|---|
| `bias_and_impact_auditor` | 3 | 7.833 | 0.732 | Strong but low count |
| `counterexample_hunter` | 13 | 7.815 | 1.137 | Strong and more frequent than before |
| `formal_semantics_checker` | 10 | 7.640 | 1.070 | Improved mean after edge exclusion |
| `benchmark_answer_judge` | 2 | 7.150 | 0.050 | Better but very rare |
| `domain_reasoner` | 34 | 6.821 | 1.677 | Frequent, moderate |
| `artifact_parser` | 55 | 6.535 | 1.992 | Nearly always first, lower than before |
| `data_consistency_auditor` | 1 | 5.800 | 0.000 | Too rare for conclusions |
| `COMPLETE` | 57 | 6.981 | 1.699 | Best final-answer score among all runs |

This is the best raw configuration. The excluded edges are absent in the new SNA, and final-answer quality improves from 6.549 to 6.981 relative to the original verification-heavy run. The graph also becomes slightly smaller, from 25 to 20 observed edges.

The improvement does not mean every node improved. `artifact_parser` drops from 6.885 to 6.535, and `domain_reasoner` drops from 6.932 to 6.821. The main gain is likely routing-level: the model avoids several weak transitions and reaches better final synthesis paths, especially via `domain_reasoner -> COMPLETE`, `domain_reasoner -> formal_semantics_checker`, and `counterexample_hunter -> COMPLETE`.

Remaining lower-scoring edges with count at least 2 include:

- `artifact_parser -> COMPLETE`: count 14, mean 6.464.
- `START -> artifact_parser`: count 55, mean 6.535.
- `formal_semantics_checker -> COMPLETE`: count 7, mean 6.686.
- `counterexample_hunter -> domain_reasoner`: count 3, mean 6.700.

### Grok Balanced

Social-network picture: `agent_evaluations/grok-4.20-balanced/openaigpt-5.4/social_network_analysis.svg`.

| Node | Count | Mean | sd | Interpretation |
|---|---:|---:|---:|---|
| `event_log_interpreter` | 32 | 5.150 | 2.391 | Best non-special node, but still low |
| `hypothesis_generator` | 46 | 5.109 | 1.790 | Frequently used, low-to-moderate |
| `query_and_sql_analyst` | 22 | 4.595 | 2.057 | Weak |
| `process_modeler` | 58 | 4.141 | 1.998 | Weak and overused |
| `fairness_and_ethics_reviewer` | 34 | 3.726 | 2.293 | Weak |
| `optimization_consultant` | 32 | 3.675 | 2.154 | Weak |
| `conformance_anomaly_checker` | 52 | 3.287 | 2.012 | Weakest node |
| `COMPLETE` | 57 | 5.325 | 2.160 | Low final-answer score |

Grok's balanced run creates a very dense network, with 4.842 agent calls and 5.842 judged edges per question. The selector frequently keeps routing among agents, but the extra discussion is low scoring and does not improve the final answer.

Prominent low-scoring edges include:

- `optimization_consultant -> query_and_sql_analyst`: count 2, mean 1.800.
- `event_log_interpreter -> fairness_and_ethics_reviewer`: count 2, mean 1.850.
- `optimization_consultant -> fairness_and_ethics_reviewer`: count 14, mean 2.400.
- `event_log_interpreter -> conformance_anomaly_checker`: count 9, mean 2.611.
- `process_modeler -> conformance_anomaly_checker`: count 20, mean 3.215.

### Qwen Balanced

Social-network picture: `agent_evaluations/qwen3.6-35b-a3b-balanced/openaigpt-5.4/social_network_analysis.svg`.

| Node | Count | Mean | sd | Interpretation |
|---|---:|---:|---:|---|
| `event_log_interpreter` | 14 | 6.721 | 1.660 | Best node, competitive with GPT on this role |
| `hypothesis_generator` | 20 | 5.515 | 1.840 | Moderate |
| `fairness_and_ethics_reviewer` | 20 | 5.165 | 2.573 | Moderate but variable |
| `conformance_anomaly_checker` | 33 | 5.045 | 2.199 | Moderate-to-low |
| `query_and_sql_analyst` | 12 | 4.742 | 1.688 | Weak |
| `process_modeler` | 45 | 4.320 | 1.663 | Weak and frequently used |
| `optimization_consultant` | 39 | 3.856 | 1.844 | Weakest node |
| `COMPLETE` | 57 | 5.388 | 1.829 | Low final-answer score |

Qwen is less extreme than Grok but still much denser than GPT, with 3.211 agent calls per question. It has one reasonably strong role, `event_log_interpreter`, but process modeling and optimization are weak.

Prominent low-scoring edges include:

- `process_modeler -> hypothesis_generator`: count 2, mean 2.300.
- `event_log_interpreter -> optimization_consultant`: count 3, mean 3.000.
- `event_log_interpreter -> process_modeler`: count 7, mean 3.214.
- `process_modeler -> fairness_and_ethics_reviewer`: count 3, mean 3.233.
- `process_modeler -> optimization_consultant`: count 13, mean 3.662.

## Comparison of the Three Balanced Models

All three model runs use the same `pm_benchmark_balanced` agent configuration and the same judge model.

| Comparison | Mean final-score difference | Wins | Losses | Ties | p approx |
|---|---:|---:|---:|---:|---:|
| GPT balanced minus Grok balanced | +1.421 | 46 | 10 | 1 | <0.001 |
| GPT balanced minus Qwen balanced | +1.358 | 45 | 10 | 2 | <0.001 |
| Grok balanced minus Qwen balanced | -0.063 | 29 | 28 | 0 | 0.822 |

Interpretation:

- `gpt-5.4-mini` performs clearly better than both `grok-4.20` and `qwen3.6-35b-a3b` under the balanced agent configuration.
- Grok and Qwen are essentially tied on final-answer quality.
- The social networks differ substantially. GPT uses a sparse network; Grok and Qwen use dense networks with many more handoffs and lower-scoring agent contributions.

Balanced SNA comparison:

| Model run | Final mean | Avg agent calls/question | Avg judged edges/question | Observed graph edges | Approx directed graph density |
|---|---:|---:|---:|---:|---:|
| `gpt-5.4-mini-balanced` | 6.746 | 1.351 | 2.351 | 29 | 0.403 |
| `grok-4.20-balanced` | 5.325 | 4.842 | 5.842 | 54 | 0.750 |
| `qwen3.6-35b-a3b-balanced` | 5.388 | 3.211 | 4.211 | 48 | 0.667 |

The density values use 9 observed nodes, including `START` and `COMPLETE`, and divide observed directed edges by 9 * 8 possible non-self edges. These are descriptive graph summaries, not inferential network tests.

The practical difference between the networks is large. The weaker model runs make many more agent calls, revisit more roles, and instantiate more handoff types. This suggests that the selector's ability to stop early and choose targeted expertise is model-dependent. In these experiments, denser deliberation correlates with lower final quality.

Strictly speaking, the repository does not contain repeated independent network samples per model/configuration sufficient for a formal significance test over graph topology. The final-answer paired differences are significant for GPT versus the other two balanced models, but "significant differences between the social networks" should be stated as observed structural differences unless a permutation/bootstrap network test is added.

## Effectiveness of Removing Underperforming Edges

The edge-exclusion experiment compares:

- Original: `gpt-5.4-mini-verification-heavy`
- Edge-pruned: `gpt-5.4-mini-ver-heavy-excl`

The excluded edges are fully absent in the new SNA:

| Edge | Original count | Original mean | Pruned count |
|---|---:|---:|---:|
| `formal_semantics_checker -> domain_reasoner` | 3 | 4.967 | 0 |
| `artifact_parser -> benchmark_answer_judge` | 3 | 6.467 | 0 |
| `artifact_parser -> formal_semantics_checker` | 6 | 6.667 | 0 |
| `domain_reasoner -> benchmark_answer_judge` | 4 | 6.925 | 0 |

Final-answer effect:

| Metric | Original verification-heavy | Edge-pruned verification-heavy | Difference |
|---|---:|---:|---:|
| Final mean | 6.549 | 6.981 | +0.432 |
| Final sd | 1.994 | 1.699 | -0.295 |
| Avg LLM calls/question | 6.316 | 6.140 | -0.176 |
| Avg agent calls/question | 2.158 | 2.070 | -0.088 |
| Observed graph edges | 25 | 20 | -5 |

By raw mean, the edge-pruned configuration is better than the original verification-heavy configuration and better than every other run in the current repository. The improvement is especially visible in `cat03`, where final mean increases from 4.275 to 6.062.

However, the conclusion should be phrased carefully:

- The edge-pruned model performs better in this run.
- The result is consistent with the hypothesis that low-performing handoff edges can damage multi-agent performance.
- It does not prove that edge pruning alone caused the improvement, because the runs are stochastic and there is no repeated-trial estimate.
- The improvement over no-agent is still modest: +0.263 mean final score, with 27 wins, 27 losses, and 3 ties.
- The pruned configuration still costs about 6.14 LLM calls per question versus 1.0 for no-agent.

In paper language, this should be treated as promising evidence for SNA-guided configuration refinement, not as conclusive evidence that pruning always improves agentic systems.

## Are Agents Convenient Compared With No Agents?

The current evidence gives a cautious answer: not generally, at least not without careful routing and pruning.

Arguments against agents as a default:

- The no-agent baseline is strong: final mean 6.718.
- GPT artifact-pipeline and balanced configurations improve the mean by only +0.019 and +0.028, respectively.
- The original verification-heavy configuration is worse than no-agent by -0.168.
- Agent configurations multiply API calls: GPT balanced averages 4.702 calls per question, artifact pipeline 5.895, and verification-heavy around 6.1 to 6.3, compared with exactly 1.0 for no-agent.
- Grok and Qwen balanced runs show that adding agents can make things substantially worse when the model routes too much or produces weak role contributions.

Arguments for agents in selected circumstances:

- The pruned verification-heavy configuration has the best raw score and improves difficult formal/modeling categories.
- The SNA artifacts make the internal process inspectable. Without agents, there are no handoff or role diagnostics.
- Specialized reviewers such as `counterexample_hunter`, `data_consistency_auditor`, `bias_and_impact_auditor`, and `answer_precision_auditor` often score well as individual nodes.
- Agents may be useful for high-risk process mining tasks where interpretability, checking, or fairness review matter more than minimizing calls.

Practical conclusion:

Agents are not automatically convenient for this benchmark. They are convenient only if the goal includes diagnostic traceability, role-level analysis, or targeted improvement on hard task classes. For pure average answer quality per API call, the no-agent baseline remains difficult to beat. The best current direction is not "more agents", but "selective agents with SNA-guided routing constraints."

## Paper-Ready Claims and How Strong They Are

Strongly supported by current artifacts:

- The implemented system is a centralized conversational multi-agent framework with selector-mediated routing.
- The main empirical scope is process mining benchmark answering and configuration effectiveness.
- GPT balanced outperforms Grok balanced and Qwen balanced under the judge metric.
- Grok and Qwen instantiate denser social networks than GPT under the same balanced configuration.
- `gpt-5.4-mini-ver-heavy-excl` is the best raw final-answer run currently present.
- More agent calls do not guarantee higher final-answer quality.

Moderately supported:

- SNA can identify weak handoff edges that are plausible targets for configuration pruning.
- Removing selected weak edges improves the verification-heavy configuration in this repository.
- Verification/review agents can be useful, but only when routing is controlled.

Weak or not yet supported:

- Agents are generally better than no agents.
- The edge-pruning improvement is statistically significant against the no-agent baseline.
- The social networks are statistically significantly different in a formal graph-theoretic sense.
- The LLM judge scores perfectly reflect process mining correctness.

## Suggested Paper Structure

1. Introduction
   - Motivation: LLMs increasingly answer process mining questions, but role decomposition and agent orchestration are underexplored.
   - Problem: multi-agent systems add cost and complexity; we need evidence that they help and diagnostics for when they do not.
   - Contribution: a selector-driven conversational multi-agent framework, benchmark evaluation across process mining tasks, and SNA-based analysis of agent roles and handoffs.

2. Background
   - Process mining question answering.
   - LLM-as-a-judge evaluation.
   - Conversational multi-agent orchestration.
   - Social network analysis for interaction traces.

3. Framework
   - Configurable agent personas.
   - Selector and final synthesis prompts.
   - Single-agent-per-iteration routing.
   - Handoff exclusion mechanism.
   - Artifact logging.

4. Experimental Design
   - PM-LLM-Benchmark task categories.
   - Configurations: no-agent, artifact pipeline, balanced, verification-heavy, edge-pruned verification-heavy.
   - Model comparison: GPT, Grok, Qwen balanced runs with thinking disabled.
   - Judge model and scoring rubric.
   - SNA construction: nodes, edges, target-node edge scoring.

5. Results
   - Overall final-answer scores.
   - Category-level results.
   - Node-level performance.
   - Edge-level performance.
   - Model comparison.
   - Edge-pruning experiment.

6. Discussion
   - When agents help.
   - When agents hurt.
   - Role of routing selectivity.
   - Why dense networks may indicate indecision rather than depth.
   - Cost/convenience trade-off.

7. Threats to Validity
   - Single run per condition.
   - Judge-model dependence.
   - No human correctness labels.
   - Missing full payload capture for thinking-disabled condition.
   - Edge scores inherit target-node scores.
   - No token/cost normalization yet.

8. Conclusion
   - Agents are diagnostically valuable but not automatically quality-improving.
   - SNA-guided pruning is the most promising finding.
   - Future work should replicate, add human evaluation, and optimize routing.

## Recommended Follow-Up Analyses

Before turning this into a paper, the following would strengthen the empirical claims:

- Repeat each configuration multiple times to estimate run-to-run variance.
- Record full OpenRouter request payloads, including thinking/reasoning settings.
- Add token counts, latency, and cost per answer.
- Compare judge scores against a human or reference-based evaluation subset.
- Run a formal paired test with correction for multiple comparisons.
- Bootstrap or permutation-test graph metrics if making formal SNA significance claims.
- Add ablations that isolate agent role changes from handoff exclusions.
- Try a fixed-route verification pipeline against the dynamic selector to separate role value from routing quality.
- Improve the process modeling agent, since process-model tasks are consistently weak.
- Test category-specific configurations, especially for `cat03` and `cat07`.
