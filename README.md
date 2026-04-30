# agents-sna

A small Python CLI for running a multi-agent LLM discussion through
OpenRouter before producing a final answer.

The default model is `openai/gpt-5.4-mini`. You can override it from the
CLI, and individual agents can also override it in the JSON config.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or install only the runtime dependency:

```bash
pip install -r requirements.txt
```

## Configure

Agent descriptions and the maximum iteration count live in JSON. Start
from [configs/agents.example.json](configs/agents.example.json):

```json
{
  "max_iterations": 4,
  "single_agent_per_iteration": false,
  "excluded_handoffs": [],
  "agents": [
    {
      "name": "planner",
      "description": "A pragmatic planning agent..."
    }
  ]
}
```

Use an empty `agents` list to skip the selector and agent-discussion
steps and send the prompt directly to final-answer generation. See
[configs/no_agents.json](configs/no_agents.json).

`max_iterations` includes the final synthesis call. For example, with
`max_iterations: 4`, the system can run up to three selector/agent
discussion iterations and then the final-answer request.

Set `single_agent_per_iteration` to `true` when each discussion round
should call at most one agent. In that mode, `excluded_handoffs` can
block directed transitions between agents:

```json
{
  "from": "artifact_parser",
  "to": "benchmark_answer_judge"
}
```

This prevents `benchmark_answer_judge` from being selected immediately
after `artifact_parser`; the selector prompt receives the currently
allowed next agents each iteration.

## Run

Set your OpenRouter key:

```bash
export OPENROUTER_API_KEY="..."
```

Run with the console script:

```bash
agents-sna --config configs/agents.example.json \
  "Design a migration plan for a legacy Flask app."
```

Or without installing the console script:

```bash
PYTHONPATH=src python3 -m agents_sna.cli --config configs/agents.example.json \
  "Design a migration plan for a legacy Flask app."
```

Useful options:

```bash
agents-sna --help
agents-sna --model openai/gpt-5.4-mini "Your prompt"
agents-sna --json "Your prompt"
agents-sna --quiet "Your prompt"
agents-sna --no-color "Your prompt"
agents-sna --request-inputs-file reports/requests.json \
  --conversation-trace-file reports/trace.json \
  "Your prompt"
```

By default, the CLI prints colored progress details to stderr while the
run is happening: the original prompt, each prompt bundle sent to the
LLM, selector responses, selected agents, and agent responses. The final
answer is printed to stdout in white. Use `--quiet` to suppress progress
output or `--no-color` to disable ANSI colors.

Both report files are optional and are written as JSON with indent `2`.
`--request-inputs-file` writes the full list of LLM request inputs, one
message list per OpenRouter call, in call order. Each message contains a
`role` such as `system`, `user`, or `assistant`, and its `content`.
`--conversation-trace-file` writes a shorter trace containing the
original prompt, selector choices, chosen-agent responses, and final
answer.

## Flow

For each discussion iteration before the final one:

1. The selector request includes a system prompt describing all configured
   agents, the original user prompt, previous agent answers as assistant
   messages, and a final user prompt asking for a JSON list of next agents
   or `["FINAL"]`.
2. Each selected agent receives its own persona as the system prompt, the
   original user prompt, previous agent answers as assistant messages, and
   a user prompt asking it to apply its persona.
3. When the selector returns `["FINAL"]`, or when the discussion budget is
   exhausted, the final request includes the original prompt, all previous
   agent answers, and a final synthesis prompt.

If `agents` is empty, the selector and discussion iterations are skipped
and only the final request is sent.

The implementation uses `requests.post` against OpenRouter's
`/chat/completions` endpoint.

## Test

```bash
python3 -m unittest discover -s tests
```

## Benchmark Runner

Run a config over every question in a local PM-LLM-Benchmark checkout:

```bash
python3 src/agents_sna/benchmark_runner.py gpt-5.4-mini-verification-heavy \
  --config configs/pm_benchmark_verification_heavy.json \
  --benchmark-dir ../pm-llm-benchmark \
  --model openai/gpt-5.4-mini
```

The positional name is the benchmark run alias used for answer filenames,
for example:

```text
../pm-llm-benchmark/answers/gpt-5.4-mini-verification-heavy_cat01_01_case_id_inference.txt
```

The actual OpenRouter model is controlled by `--model`.

Questions with an existing non-empty answer file are skipped unless
`--overwrite` is passed. If an OpenRouter request fails, the runner
retries the same request after 15 seconds by default; adjust this with
`--retry-delay` or cap retries with `--max-retries`. If a question still
fails at the orchestration level, the failure is recorded and the runner
continues with later questions by default. Use `--stop-on-error` to stop
after the first failed question.

Local artifacts are written under:

```text
benchmark_runs/<run-name>/requests/
benchmark_runs/<run-name>/responses/
benchmark_runs/<run-name>/traces/
benchmark_runs/<run-name>/metadata/
benchmark_runs/<run-name>/summary.json
```

PNG questions are sent as multimodal `image_url` messages by default. Use
`--skip-images` to process only textual questions.

## Agent Judge

Evaluate saved benchmark request/response artifacts with an
LLM-as-a-judge:

```bash
python3 src/agents_sna/agent_judge.py \
  benchmark_runs/gpt-5.4-mini-verification-heavy \
  --judge-model openai/gpt-5.4
```

You can also pass a `requests/` directory, individual
`*.requests.json` files, or glob patterns. Matching response files are
found from the sibling `responses/` directory by default.

Each output file is written under:

```text
agent_evaluations/<source-run>/<judge-model>/<question>.agent_evaluations.json
```

The output is a JSON list of nodes. `START` is always first and
represents the first selector decision, `COMPLETE` is always last, and
later selector calls are skipped:

```json
[
  { "agent_type": "START", "evaluation": 8.0, "explanation": "Good first choice" },
  { "agent_type": "artifact_parser", "evaluation": 8.5, "explanation": "Correct parsing" },
  { "agent_type": "COMPLETE", "evaluation": 9.0, "explanation": "Grounded final answer" }
]
```

## Evaluation Network

Aggregate a folder of judged agent-evaluation files into node and edge
statistics:

```bash
python3 src/agents_sna/evaluation_network.py \
  agent_evaluations/gpt-5.4-mini-verification-heavy/openaigpt-5.4
```

The default output is:

```text
agent_evaluations/<source-run>/<judge-model>/social_network_analysis.json
```

Nodes are grouped by `agent_type`. Directed edges are consecutive
`agent_type` pairs in each evaluation file. Since evaluation files score
nodes, not handoffs, each edge occurrence is scored with the target
node's evaluation by default. Use `--edge-score mean` to score an edge
with the mean of its source and target node scores.

The output contains `count`, `average`, and population `stddev` for each
node and edge:

```json
{
  "nodes": [
    { "agent_type": "artifact_parser", "count": 30, "average": 7.2, "stddev": 0.8 }
  ],
  "edges": [
    { "source": "START", "target": "artifact_parser", "count": 18, "average": 7.4, "stddev": 0.9 }
  ]
}
```
