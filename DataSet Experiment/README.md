# LangFuse Multi-Turn Conversation Evaluation Framework

A framework for simulating and evaluating multi-turn LLM conversations using [Langfuse](https://langfuse.com/) for observability and experiment tracking. It uses an **LLM-as-a-Judge** pattern to automatically score conversation quality across configurable metrics.

---

## Overview

This project:

- **Simulates** realistic multi-turn conversations between a synthetic user (LLM-powered) and an AI chatbot (LLM-powered).
- **Evaluates** each conversation turn automatically using a judge LLM scoring relevance, accuracy, helpfulness, and clarity.
- **Tracks** all traces, sessions, scores, and experiments in Langfuse for analysis and comparison across runs.
- **Runs structured experiments** against named Langfuse datasets so results can be compared across runs.

All LLM inference is served locally via [Ollama](https://ollama.com/), making the framework fully self-contained with no external LLM API costs.

---

## Architecture

```
.env (LANGFUSE_* keys)
     │
     ▼
utils/utils.py ──── load_config() ──── config/config.json
                                              │
               ┌──────────────────────────────┤
               ▼                              ▼
        agents/chatbot.py           agents/simulated_user.py
        LlamaChatbot                SimulatedUser
        (llama3.1 via Ollama)       (mistral via Ollama)
               │                              │
               └─────── main.py ──────────────┘
                    simulate_single_turn()
                    simulate_continuous_conversation()
                    evaluate_and_score_traces()
                    run_task()
                         │
                         ▼
                  metrics/metric.py
                  EvaluationMetrics
                  (qwen3:14b via Ollama as judge)
                         │
                         ▼
                  Langfuse Cloud / Self-hosted
                  Traces / Sessions / Scores / Experiments
```

### Models Used

| Role | Model | Purpose |
|---|---|---|
| Chatbot | `llama3.1:latest` | Generates assistant responses |
| Simulated User | `mistral:latest` | Generates synthetic user messages |
| Judge | `qwen3:14b` | Evaluates and scores conversation turns |

---

## Project Structure

```
langfuse_project/
├── main.py                      # Primary entry point — runs experiments
├── main.ipynb                   # Notebook version for interactive development
├── requirements.txt             # Pinned Python dependencies
├── .env                         # Environment variables (not committed)
│
├── config/
│   └── config.json              # Model URLs, dataset name, evaluation metric definitions
│
├── agents/
│   ├── chatbot.py               # LlamaChatbot — the AI assistant being evaluated
│   └── simulated_user.py        # SimulatedUser — synthetic user that drives the conversation
│
├── metrics/
│   └── metric.py                # EvaluationMetrics — LLM judge + Langfuse evaluator factory
│
├── utils/
│   └── utils.py                 # Config loading, error handling, session/trace ID management
│
├── data/
│   └── evaluation_dataset.py    # Dataset definitions (personas and scenarios)
│
├── scripts/
│   ├── upload_dataset.py        # Upload evaluation dataset items to Langfuse
│   └── view_results.py          # CLI to inspect scores for a given Langfuse session
│
└── notebooks/
    └── dataset_experiment.ipynb # Exploratory notebook for dataset management
```

---

## Prerequisites

### 1. Ollama (Local LLM Server)

Install [Ollama](https://ollama.com/) and pull the required models:

```bash
ollama pull llama3.1:latest
ollama pull mistral:latest
ollama pull qwen3:14b
```

Ollama must be running on `http://localhost:11434` before executing any scripts.

### 2. Langfuse

Set up a [Langfuse](https://langfuse.com/) account (cloud or self-hosted). You will need your project's **Public Key** and **Secret Key**.

### 3. Python Environment

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # Optional; defaults to cloud
```

### `config/config.json`

Key configuration options:

| Section | Key | Description |
|---|---|---|
| `ollama` | `base_url` | Ollama API endpoint (default: `http://localhost:11434/v1`) |
| `models` | `answer_model` | Chatbot model name |
| `models` | `answer_model_alt` | Simulated user model name |
| `models` | `judge_model` | Evaluation judge model name |
| `dataset` | `name` | Langfuse dataset name |
| `dataset` | `max_turns` | Number of conversation turns per session |
| `evaluation_metrics` | — | Per-metric name, description, min/max scale, and score labels |

---

## Usage

### Step 1 — Upload the Dataset

Upload conversation scenarios (personas + situations) to Langfuse:

```bash
python scripts/upload_dataset.py
```

This is safe to re-run — it checks for existing items before uploading.

### Step 2 — Run an Experiment

Execute the full simulation + evaluation pipeline:

```bash
python main.py
```

This will:
1. Fetch items from the Langfuse dataset (`multi-turn-conversations2`).
2. For each item, simulate a multi-turn conversation (`max_turns = 6`).
3. Score each conversation turn with the judge LLM across all configured metrics.
4. Post all scores back to Langfuse and register the run under the experiment name `unified-metrics-template-v1`.

### Step 3 — View Results

Inspect scores for a specific session interactively:

```bash
python scripts/view_results.py
```

Enter a session ID when prompted to see all trace-level scores and comments.

Alternatively, open the [Langfuse dashboard](https://cloud.langfuse.com) to browse experiments, traces, and scores visually.

---

## Evaluation Metrics

Each conversation turn is scored by the judge LLM on a **1–5 scale** across four dimensions:

| Metric | Description |
|---|---|
| **Relevance** | How well the assistant's response addresses the user's question |
| **Accuracy** | Factual correctness of the response |
| **Helpfulness** | Practical usefulness of the response |
| **Clarity** | How clearly and understandably the response is written |

An **overall score** (normalised 0–1) is computed as the mean of all metric scores divided by the maximum score (5). Scores are posted to Langfuse at both the trace level (per turn) and the run level (aggregate).

---

## Code Flow

```
main.py __main__
  └─ langfuse.get_dataset("multi-turn-conversations2")
  └─ dataset.run_experiment(task=run_task, evaluators=[...], run_evaluators=[...])
       └─ run_task(item=<dataset_item>)
            ├─ simulate_continuous_conversation(persona, scenario, session_id, max_turns=6)
            │    └─ [loop N turns]
            │         └─ simulate_single_turn(...)
            │              ├─ SimulatedUser.generate_message()   → mistral
            │              └─ LlamaChatbot.chat()                → llama3.1
            └─ evaluate_and_score_traces(conversation_log, ...)
                 └─ [per turn] EvaluationMetrics.evaluate_turn() → qwen3:14b
                 └─ langfuse.create_score(trace_id, metric, value, ...)
       └─ [per item] create_evaluator(metric)(...)  → Langfuse Evaluation
       └─ [run level] create_average_quality_run_evaluator()(...) → avg_quality
  └─ langfuse.flush()
```

---

## Key Modules

### `agents/chatbot.py` — `LlamaChatbot`
Maintains conversation history and sends the full history to `llama3.1` on each turn. Each response is wrapped in a Langfuse `@observe` span for tracing.

### `agents/simulated_user.py` — `SimulatedUser`
Generates realistic user messages using `mistral`, guided by a persona and scenario defined in the dataset. Maintains its own conversation history to ensure coherent multi-turn flow.

### `metrics/metric.py` — `EvaluationMetrics`
Sends structured prompts to the judge model (`qwen3:14b`) requesting JSON-formatted scores. Implements the Langfuse evaluator interface (`create_evaluator`, `get_all_evaluators`, `create_average_quality_run_evaluator`) for seamless integration with `dataset.run_experiment()`.

### `utils/utils.py` — Utilities & `SessionManager`
Handles config loading with environment variable substitution, Langfuse connection validation, deterministic trace ID generation (`langfuse.create_trace_id(seed=...)`), and error handling that returns error messages as strings so conversations can continue despite individual turn failures.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `main.ipynb` | Interactive mirror of `main.py` for step-by-step development and testing |
| `notebooks/dataset_experiment.ipynb` | Dataset creation, upload, inspection, and experiment execution in a notebook environment |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ConnectionError` to Langfuse | Check `.env` keys and `LANGFUSE_HOST`. Run `validate_langfuse_connection()` in utils. |
| Ollama model not found | Run `ollama pull <model_name>` and ensure Ollama is running. |
| `LANGFUSE_PUBLIC_KEY not set` | Ensure `.env` file exists in the project root with the correct keys. |
| Scores not appearing in Langfuse | Call `langfuse.flush()` at the end of your script to ensure all events are sent. |

---

## Dependencies

Key packages (see `requirements.txt` for full pinned versions):

| Package | Version | Purpose |
|---|---|---|
| `langfuse` | 3.12.1 | LLM observability, tracing, dataset experiments |
| `openai` | 2.15.0 | OpenAI-compatible client for Ollama |
| `python-dotenv` | 1.2.1 | Load `.env` secrets |
| `pydantic` | 1.10.26 | Data validation |
| `tqdm` | 4.67.1 | Progress bars |
| `ipykernel` | 7.2.0 | Jupyter notebook support |
