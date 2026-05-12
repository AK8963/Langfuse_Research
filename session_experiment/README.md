# Langfuse Multi-Turn Conversation Evaluation Project

A comprehensive system for running multi-turn conversation experiments between AI agents with session-level evaluation and tracing using Langfuse.

## Overview

This project simulates realistic multi-turn conversations between a chatbot and a simulated user, evaluates the conversations using LLM-as-a-Judge methodology, and tracks all interactions with Langfuse for observability and analysis.

**Key Features:**
- Multi-turn conversation simulation with configurable personas and scenarios
- Session-level evaluation metrics (relevance, accuracy, helpfulness, clarity)
- Complete tracing and observability with Langfuse
- Structured experiment runs with dataset integration
- Local LLM support via Ollama
- Jupyter notebook support for interactive analysis

## Project Structure

```
Langfuse_project/
├── agents/                  # AI agent implementations
│   ├── chatbot.py          # LlamaChatbot for multi-turn conversations
│   └── simulator.py        # SimulatedUser for realistic user interactions
├── config/                 # Configuration files
│   └── config.json         # Ollama, Langfuse, and model settings
├── metrics/                # Evaluation metrics
│   ├── metrics.py          # Session-level evaluation metrics
│   └── metrics2.py         # Additional metrics utilities
├── data/                   # Data processing
│   └── data.py             # Data handling utilities
├── utils/                  # Utility functions
│   ├── utils.py            # Core utilities (SessionManager, logger)
│   └── utils2.py           # Additional utilities
├── notebooks/              # Jupyter notebooks for analysis
├── session.ipynb           # Main session-level evaluation notebook
├── session2.ipynb          # Secondary experiment notebook
├── test.py                 # Test utilities
├── requirements.txt        # Python dependencies
├── config.json            # Configuration (see config/config.json)
├── .env                   # Environment variables (secrets)
└── venv_py313/            # Python virtual environment
```

## Getting Started

### Prerequisites

- Python 3.13+
- Ollama (for local LLM inference)
- Langfuse account and API keys
- 8GB+ RAM recommended

### Installation

1. **Clone/navigate to the project directory:**
   ```bash
   cd Langfuse_project
   ```

2. **Create and activate the virtual environment:**
   ```bash
   # If not already created
   python -m venv venv_py313
   
   # Activate (Windows)
   venv_py313\Scripts\activate
   
   # Activate (macOS/Linux)
   source venv_py313/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   # Create .env file with your credentials
   LANGFUSE_PUBLIC_KEY=your_public_key
   LANGFUSE_SECRET_KEY=your_secret_key
   LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted URL
   ```

5. **Update configuration:**
   Edit `config/config.json` with your settings:
   - Ollama base URL and API key
   - Model names (answer_model, judge_model)
   - Dataset configuration
   - Evaluation parameters

### Starting Ollama

Before running experiments, start the Ollama service:

```bash
# Start Ollama (ensure it's listening on http://localhost:11434)
ollama serve
```

In another terminal, pull required models:
```bash
ollama pull llama3:latest
ollama pull mistral:latest
```

## Usage

### Running Session-Level Experiments

Execute the main experiment notebook or script:

```bash
# Using Jupyter
jupyter notebook session.ipynb

# Or run as Python script
python session.ipynb  # (if configured as executable)
```

**What it does:**
1. Loads a dataset from Langfuse with personas and scenarios
2. For each dataset item, simulates a multi-turn conversation (configurable turns)
3. Each turn is tracked as a separate trace with session grouping
4. Applies session-level evaluators to assess overall conversation quality
5. Logs results back to Langfuse for analysis

### Configuration Details

#### `config/config.json`

**Ollama Settings:**
```json
{
  "ollama": {
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama"
  }
}
```

**Langfuse Settings:**
```json
{
  "langfuse": {
    "public_key": "${LANGFUSE_PUBLIC_KEY}",
    "secret_key": "${LANGFUSE_SECRET_KEY}",
    "LANGFUSE_BASE_URL": "${LANGFUSE_HOST}"
  }
}
```

**Model Selection:**
```json
{
  "models": {
    "answer_model": "llama3:latest",      # Chatbot model
    "answer_model_alt": "llama3:latest",  # Simulated user model
    "judge_model": "mistral:latest"       # Evaluation model
  }
}
```

**Dataset & Evaluation:**
```json
{
  "dataset": {
    "name": "6-turn-travel-conversations",
    "max_turns": 6
  },
  "evaluation": {
    "retry_attempts": 3,
    "timeout_seconds": 30,
    "max_concurrency": 5
  }
}
```

**Session Evaluation Metrics:**
- **Session Relevance**: Ensures all responses stay on topic
- **Session Accuracy**: Validates factual correctness across turns
- **Session Helpfulness**: Measures overall usefulness to the user
- **Session Clarity**: Evaluates communication clarity throughout

## 🔧 Core Components

### Agents

#### `LlamaChatbot` (agents/chatbot.py)
- Maintains conversation history
- Integrates with Langfuse for tracing
- Supports session-level context propagation
- Generates responses using configured LLM model

```python
chatbot = LlamaChatbot(
    model="llama3:latest",
    ollama_config=config['ollama'],
    session_id="session-12345"
)
response = chatbot.chat("Hello!", turn_number=1)
```

#### `SimulatedUser` (agents/simulator.py)
- Generates realistic user messages based on persona and scenario
- Maintains conversation context
- Supports first-turn and follow-up message generation

```python
user = SimulatedUser(
    persona="A travel enthusiast interested in budgets",
    scenario="Planning a trip to Southeast Asia",
    model="llama3:latest",
    ollama_config=config['ollama']
)
message = user.generate_message(is_first_turn=True)
```

### Metrics

#### `EvaluationMetrics` (metrics/metrics.py)
Provides session-level evaluation functions:
- `create_session_relevance_evaluator()`: Checks topic consistency
- `create_session_accuracy_evaluator()`: Validates information accuracy
- `create_session_helpfulness_evaluator()`: Measures practical usefulness
- `create_session_clarity_evaluator()`: Assesses communication quality

### Utilities

#### `SessionManager` (utils/utils.py)
- `generate_session_id()`: Creates unique session identifiers
- `generate_trace_id()`: Creates trace IDs per turn
- `log_session_start()`: Records session initialization
- `log_session_complete()`: Records session completion

#### Logger
Configured logging for debugging and monitoring:
```python
from utils.utils import logger
logger.info("Starting experiment...")
```

## Experiment Flow

```
┌─────────────────────────────────────────┐
│  Load Dataset from Langfuse             │
│  (personas & scenarios)                 │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  For Each Dataset Item                  │
├─────────────────────────────────────────┤
│  1. Generate Unique Session ID          │
│  2. Initialize Chatbot & Simulated User │
│  3. For Each Turn (1 to max_turns):     │
│     a. User generates message           │
│     b. Chatbot responds                 │
│     c. Trace created with turn data     │
│     d. Append to conversation log       │
│  4. Apply Session Evaluators            │
│  5. Log Results to Langfuse             │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Aggregate Results                      │
│  Generate Experiment Report             │
└─────────────────────────────────────────┘
```

## Testing

Run basic tests:
```bash
python test.py
```

## Monitoring Results

### In Langfuse UI:
1. Navigate to **Dataset Runs**
2. Find your experiment run
3. View individual traces organized by session
4. Review evaluation scores and metrics
5. Compare runs and analyze trends

### Programmatic Access:
```python
from langfuse import get_client

client = get_client()
dataset = client.get_dataset("6-turn-travel-conversations")
runs = dataset.runs

for run in runs:
    print(f"Run: {run.name}")
    for trace in run.traces:
        print(f"  Trace: {trace.id}")
```

## Security

- **API Keys**: Store Langfuse credentials in `.env` (never commit!)
- **Ollama**: Runs locally; configure firewall as needed
- **Data Privacy**: Review Langfuse data retention policies

## Troubleshooting

### Ollama Connection Issues
```
Error: Failed to connect to http://localhost:11434
```
**Solution**: Ensure Ollama is running and listening on the correct port.

### Langfuse Authentication
```
Error: Langfuse authentication failed
```
**Solution**: Verify `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` in `.env`.

### Model Loading
```
Error: Model not found: llama3:latest
```
**Solution**: Pull the model first:
```bash
ollama pull llama3:latest
```

### Memory Issues
```
Error: CUDA out of memory or system memory exhausted
```
**Solution**: 
- Use smaller models (e.g., `mistral:7b`)
- Reduce `max_turns` in configuration
- Reduce `max_concurrency` in evaluation settings

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| langfuse | 3.12.1 | LLM observability & tracing |
| openai | 2.16.0 | LLM API client (compatible with Ollama) |
| pydantic | 2.12.5 | Data validation |
| python-dotenv | 1.2.1 | Environment variable management |
| opentelemetry | 1.39.1 | Distributed tracing |

See `requirements.txt` for complete list.

## Notebooks

- **session.ipynb**: Main experiment with session-level evaluation
- **session2.ipynb**: Alternative experiment configuration for testing

## Contributing

1. Create a feature branch
2. Make changes to agents, metrics, or utilities
3. Test with a subset of data
4. Submit results



## Support

For issues with:
- **Langfuse**: Check [langfuse.com/docs](https://langfuse.com/docs)
- **Ollama**: Visit [ollama.ai](https://ollama.ai)
- **OpenAI Python Client**: See [github.com/openai/openai-python](https://github.com/openai/openai-python)


