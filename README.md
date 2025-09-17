## LangGraph Deep Research Agent

An example LangGraph project that builds a small, deterministic multi-node agent to clarify a user's research request and generate a high-quality research brief. It uses LangChain's `init_chat_model` with structured output and the LangGraph CLI for a Dev UI.

### Features

- **Clarify user request**: Decides whether to ask a concise clarifying question or proceed.
- **Write research brief**: Transforms the conversation into a detailed brief.
- **Deterministic structure**: Uses Pydantic schemas for structured outputs.
- **Dev UI via LangGraph CLI**: Run and chat with the agent locally.

### Requirements

- Python 3.13+
- An OpenAI API key (for `gpt-4o-mini` via `langchain-openai`)

### Quickstart

#### 1) Clone and setup environment

```bash
git clone https://github.com/your-org/langgraph_deep_agents.git
cd langgraph_deep_agents

# Using uv (recommended)
uv venv
source .venv/bin/activate
uv sync

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

#### 2) Configure environment variables

Copy `.env.example` to `.env` and set your keys.

```bash
cp .env.example .env
```

Required variables:

- `OPENAI_API_KEY`: Your OpenAI key used by `langchain-openai`.

Optional (for LangChain telemetry):

- `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, etc.

#### 3) Run the Dev UI (LangGraph CLI)

This project ships a `langgraph.json` that points to the graph in `src/graph.py`.

```bash
langgraph dev
```

This launches a local Dev UI in your browser. Select `deep_research_agent` and start chatting. The agent will either:

- Ask one clarifying question and stop, or
- Confirm it has enough info and proceed to generate a research brief.

### Programmatic usage

You can also run the compiled graph directly from Python.

```python
from langgraph.graph import MessagesState
from src.graph import graph

state: MessagesState = {"messages": [
    {"role": "user", "content": "Research top hiking backpacks for multi-day trips in 2025."}
]}

app = graph
result = app.invoke(state)
print(result.get("research_brief"))
```

Note: The graph compiled in `src/graph.py` expects a `MessagesState`-compatible input with `messages`.

### Project structure

```text
src/
  graph.py              # Builds and compiles the StateGraph
  state.py              # Agent state definitions (InputState, AgentState)
  schema.py             # Pydantic models for structured outputs
  utils.py              # Small helpers (date formatting)
  nodes/
    clarify_user_request.py  # Clarification router node
    write_research_brief.py  # Brief-generation node
langgraph.json          # CLI configuration (exposes deep_research_agent)
main.py                 # Placeholder entry point
notebook/evals.ipynb    # (Optional) space for experiments/evaluations
```

### How it works

- `src/graph.py` defines a `StateGraph` with two nodes:

  - `clarify_user_request`: Uses a structured schema `ClarifyUserRequest` to decide whether to ask a question (`need_clarification=True`) or proceed. If clarification is needed, it ends the graph and returns the question as an AI message.
  - `write_research_brief`: Uses `WriteResearchBrief` to produce a well-scoped, detailed brief from the message history, storing it on `AgentState.research_brief`.

- `src/state.py` defines:

  - `InputState(MessagesState)` – input schema compatible with LangGraph message flows.
  - `AgentState(MessagesState)` – extends state with a `research_brief: str` field.

- `src/schema.py` contains Pydantic models that constrain the LLM outputs to avoid hallucinated structure.

### Configuration

`langgraph.json`:

```json
{
  "graphs": {
    "deep_research_agent": "src/graph.py:graph"
  },
  "dependencies": ["."],
  "env": "./.env",
  "python_version": "3.13"
}
```

### Development

- Formatter/lint: follow your preferred toolchain. Code aims for clarity and explicitness.
- Python version pinned to 3.13 in `pyproject.toml` and `langgraph.json`.

### Troubleshooting

- "Model not found" or auth errors: ensure `OPENAI_API_KEY` is set and valid.
- Import errors for `langgraph` or `langchain-openai`: re-run `uv sync` or `pip install -e .`.
- Dev UI doesn’t show the graph: confirm `langgraph.json` points to `src/graph.py:graph` and that your venv is active.

### License

Add your preferred license here.
