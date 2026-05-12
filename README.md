# Langfuse Research: Dual Experiment Workflow

## Overview

This project contains **two complementary experiments** for evaluating multi-turn LLM conversations using Langfuse:

1. **Dataset Experiment** - Turn-level evaluation with per-turn scoring
2. **Session Experiment** - Session-level evaluation with aggregate conversation metrics

Both experiments use local LLMs (via Ollama), structured dataset management, and Langfuse for complete observability.

---

## Experiment 1: Dataset Experiment

**Location:** `DataSet Experiment/`

### Purpose
Simulates multi-turn conversations with **per-turn evaluation**, scoring each individual exchange for relevance, accuracy, helpfulness, and clarity.

### Key Components
- **Chatbot** (`agents/chatbot.py`): LLM-powered assistant maintaining conversation history
- **Simulated User** (`agents/simulated_user.py`): Realistic user generating contextual messages
- **Metrics Engine** (`metrics/metric.py`): LLM-as-a-Judge scoring each turn independently
- **Main Script** (`main.py`): Orchestrates simulation and evaluation pipeline
- **Notebook** (`main.ipynb`): Interactive analysis and visualization

### Workflow
```
1. Load Langfuse Dataset
        ↓
2. For Each Dataset Item (persona + scenario)
        ↓
3. Generate First User Message
        ↓
4. Chatbot Responds (Turn 1)
        ↓
5. Evaluate Turn 1 (per-turn scores)
        ↓
6. Simulated User Generates Follow-up
        ↓
7. Repeat steps 4-6 for N turns
        ↓
8. Log All Turn-Level Scores to Langfuse
        ↓
9. Aggregate & Generate Run-Level Metrics
```

### Output
- Individual turn traces with per-turn evaluation scores
- Run-level aggregate metrics
- Experiment comparison capability in Langfuse UI

### Configuration
```json
{
  "models": {
    "answer_model": "llama3.1",
    "answer_model_alt": "mistral",
    "judge_model": "qwen3:14b"
  },
  "max_turns": 10
}
```

---

## Experiment 2: Session Experiment

**Location:** `session_experiment/`

### Purpose
Simulates multi-turn conversations with **session-level evaluation**, assessing overall conversation quality across all turns as a cohesive session.

### Key Components
- **Chatbot** (`agents/chatbot.py`): LLM-powered assistant with session tracking
- **Simulated User** (`agents/simulator.py`): Persona-driven message generation
- **Session Manager** (`utils/utils.py`): Handles session and trace ID generation
- **Metrics Engine** (`metrics/metrics.py`): Session-level evaluation only
- **Main Notebook** (`session.ipynb`): Experiment execution and analysis

### Workflow
```
1. Load Langfuse Dataset
        ↓
2. For Each Dataset Item (persona + scenario)
        ↓
3. Generate Unique Session ID
        ↓
4. Initialize Chatbot & Simulated User with Session Context
        ↓
5. For Each Turn (1 to max_turns):
        a. User generates message (context-aware)
        b. Chatbot responds (session-aware)
        c. Create separate trace per turn
        d. Tag all traces with same session_id
        ↓
6. Collect Full Conversation Log
        ↓
7. Apply Session-Level Evaluators:
        - Overall Relevance (all turns on-topic?)
        - Overall Accuracy (consistent factual correctness?)
        - Overall Helpfulness (useful throughout?)
        - Overall Clarity (clear communication?)
        ↓
8. Log Session-Level Scores to Langfuse
        ↓
9. Generate Run-Level Aggregate Metrics
```

### Output
- Multiple traces grouped by session_id (all turns visible)
- Session-level evaluation scores (single score per conversation)
- Run-level metrics across all sessions

### Configuration
```json
{
  "models": {
    "answer_model": "llama3:latest",
    "answer_model_alt": "llama3:latest",
    "judge_model": "mistral:latest"
  },
  "max_turns": 6,
  "evaluators": [
    "session_relevance",
    "session_accuracy",
    "session_helpfulness",
    "session_clarity"
  ]
}
```

---

## Key Differences

| Aspect | Dataset Experiment | Session Experiment |
|--------|-------------------|-------------------|
| **Evaluation Scope** | Per-turn | Per-session (whole conversation) |
| **Score Granularity** | Turn-level scores | Session-level scores |
| **Judge Model** | qwen3:14b | mistral:latest |
| **Traces** | One trace per turn, independent | Multiple traces per session, grouped by session_id |
| **Use Case** | Detailed turn quality analysis | Overall conversation quality assessment |
| **Metrics** | Turn-by-turn performance | Holistic conversation performance |

---

## Complementary Analysis

### When to Use Each

**Dataset Experiment:**
- Analyzing specific turn failures or successes
- Debugging chatbot responses at granular level
- Per-turn quality trends
- Identifying where conversations break down

**Session Experiment:**
- Evaluating overall conversation flow
- User satisfaction metrics
- Persona-scenario combination viability
- End-to-end conversation quality

### Combined Insights

Run both experiments on same dataset to:
1. Identify which turns cause session-level quality drops
2. Compare per-turn vs. overall quality metrics
3. Optimize chatbot for both individual responses AND conversation continuity
4. Detect if early turns impact later conversation quality

---

## Setup & Execution

### Prerequisites
```bash
# 1. Install Python 3.13+
# 2. Start Ollama
ollama serve

# 3. Pull required models
ollama pull llama3:latest
ollama pull llama3.1
ollama pull mistral:latest
ollama pull qwen3:14b
```

### Configuration
1. Create `.env` file:
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=http://localhost:3000
```

2. Update model names in respective `config/config.json` files

### Run Dataset Experiment
```bash
cd "DataSet Experiment"
pip install -r requirements.txt
python main.py
# or
jupyter notebook main.ipynb
```

### Run Session Experiment
```bash
cd session_experiment
pip install -r requirements.txt
jupyter notebook session.ipynb
```

---

## Langfuse Dashboard Navigation

### View Results

1. **Traces View**
   - Dataset Exp: Each trace = one turn
   - Session Exp: Multiple traces per session_id

2. **Experiments View**
   - Compare run scores across iterations
   - View aggregate metrics

3. **Dataset Runs**
   - Filter by experiment name
   - Compare per-turn vs. session-level results

4. **Session Grouping** (Session Experiment)
   - Click on session_id to see all turns together
   - View session-level evaluation scores

---

## Performance Considerations

| Factor | Dataset Exp | Session Exp |
|--------|------------|------------|
| **Trace Count** | N items × M turns | N items × M turns |
| **Evaluation Time** | Per-turn scores (faster) | Holistic scoring (slightly slower) |
| **Storage** | More granular data | More aggregated data |
| **Comparison Speed** | Per-turn analysis | Session overview faster |

---

## Best Practices

### Dataset Experiment
1. Use comprehensive judge model for per-turn accuracy
2. Collect larger datasets for statistical significance
3. Analyze turn-level failure patterns
4. Monitor individual metric trends

### Session Experiment
1. Use consistent session IDs for grouping
2. Ensure conversation continuity through turns
3. Focus on overall quality metrics
4. Compare session scores across personas/scenarios

### General
1. **Backup .env** - Never commit sensitive keys
2. **Version configs** - Track model and parameter changes
3. **Tag runs** - Use descriptive names for comparisons
4. **Document results** - Record insights from each experiment

---

## Troubleshooting

### Ollama Connection Issues
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Restart if needed
ollama serve
```

### Model Not Found
```bash
ollama pull model_name:latest
ollama list  # Verify installation
```

### Langfuse Auth Errors
```bash
# Verify credentials in .env
echo $LANGFUSE_PUBLIC_KEY
echo $LANGFUSE_SECRET_KEY
```

---

## File Structure

```
Langfuse_Research/
├── DataSet Experiment/          # Turn-level evaluation experiment
│   ├── agents/
│   ├── metrics/
│   ├── config/
│   ├── main.py                  # Main execution script
│   ├── main.ipynb               # Interactive notebook
│   └── requirements.txt
│
├── session_experiment/          # Session-level evaluation experiment
│   ├── agents/
│   ├── metrics/
│   ├── config/
│   ├── utils/
│   ├── session.ipynb            # Main execution notebook
│   ├── session2.ipynb           # Alternative notebook
│   └── requirements.txt
│
├── README.md                    # Project overview
├── WORKFLOW.md                  # This file
├── .gitignore                   # Git ignore rules
└── venv_py313/                  # Virtual environment
```

---

## Future Enhancements

- [ ] Cross-experiment comparison dashboard
- [ ] Automated per-turn vs. session quality correlation analysis
- [ ] Multi-model evaluation (judge model ensemble)
- [ ] Real-time monitoring dashboard
- [ ] Batch processing for large datasets
- [ ] Custom metric definitions

---

## References

- [Langfuse Documentation](https://langfuse.com/docs)
- [Ollama Models](https://ollama.ai/library)
- [LLM-as-a-Judge Pattern](https://arxiv.org/abs/2306.05685)

---

**Last Updated:** May 12, 2026  
**Status:** Active Development  
**Version:** 1.0
