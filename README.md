# Prompt Playground

Streamlit sandbox for testing system/user prompts against OpenAI or a local Ollama model. Tune temperature, compare two variants side by side, and keep a JSONL history of runs.

## Features

- System + user prompt editors
- Provider switch: **OpenAI** or **Ollama**
- Temperature and max-tokens controls
- Single run and A/B compare modes
- Persistent run history in `data/history.jsonl`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY if using OpenAI
```

For local models, install and start [Ollama](https://ollama.com), then pull a model:

```bash
ollama pull llama3.2
```

## Run

```bash
streamlit run app.py
```

## Configuration

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | Default OpenAI model (`gpt-4o-mini`) |
| `OLLAMA_BASE_URL` | Ollama HTTP API (`http://localhost:11434`) |
| `OLLAMA_MODEL` | Default local model (`llama3.2`) |

History is stored at `data/history.jsonl` (gitignored).
