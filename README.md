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
  "agents": [
    {
      "name": "planner",
      "description": "A pragmatic planning agent..."
    }
  ]
}
```

`max_iterations` includes the final synthesis call. For example, with
`max_iterations: 4`, the system can run up to three selector/agent
discussion iterations and then the final-answer request.

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
```

By default, the CLI prints colored progress details to stderr while the
run is happening: the original prompt, each prompt bundle sent to the
LLM, selector responses, selected agents, and agent responses. The final
answer is printed to stdout in white. Use `--quiet` to suppress progress
output or `--no-color` to disable ANSI colors.

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

The implementation uses `requests.post` against OpenRouter's
`/chat/completions` endpoint.

## Test

```bash
python3 -m unittest discover -s tests
```
