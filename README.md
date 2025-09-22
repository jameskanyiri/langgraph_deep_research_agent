## LangGraph Deep Research Agent

A sophisticated multi-agent research system built with LangGraph that orchestrates multiple specialized agents to conduct comprehensive research on complex topics. The system features a supervisor agent that coordinates parallel research activities using specialized research agents equipped with web search capabilities.

### Features

- **Multi-Agent Architecture**: Supervisor agent coordinates multiple parallel research agents
- **Intelligent Research Coordination**: Supervisor decides research topics and manages parallel execution
- **Web Search Integration**: Research agents use Tavily API for real-time web search
- **Strategic Thinking**: Built-in reflection tools for quality decision-making during research
- **Research Compression**: Intelligent summarization of findings for supervisor consumption
- **Clarification System**: Initial clarification to refine research scope
- **Dev UI via LangGraph CLI**: Run and chat with the agent locally

### Requirements

- Python 3.13+
- An OpenAI API key (for `gpt-4o-mini` and `gpt-4.1` via `langchain-openai`)
- A Tavily API key (for web search capabilities)

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

- `OPENAI_API_KEY`: Your OpenAI key used by `langchain-openai`
- `TAVILY_API_KEY`: Your Tavily API key for web search capabilities

Optional (for LangChain telemetry):

- `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, etc.

#### 3) Run the Dev UI (LangGraph CLI)

This project ships a `langgraph.json` that points to the graph in `src/graph.py`.

```bash
langgraph dev
```

This launches a local Dev UI in your browser. Select `deep_research_agent` and start chatting. The system will:

1. **Clarify** your request if needed
2. **Generate** a detailed research brief from the conversation
3. **Coordinate** parallel research agents to investigate different aspects
4. **Compress** and synthesize findings into comprehensive results

You can also run individual components:

- `research_agent`: Individual research agent with search capabilities
- `supervisor_agent`: Supervisor that coordinates research activities

### Programmatic usage

You can also run the compiled graphs directly from Python.

```python
from src.graph import graph
from src.research_agent.agent import research_agent
from src.supervisor.supervisor import supervisor_agent

# Run the full research pipeline
state = {"messages": [
    {"role": "user", "content": "Research top hiking backpacks for multi-day trips in 2025."}
]}

result = graph.invoke(state)
print(result.get("notes"))  # Final research findings

# Run individual components
research_result = research_agent.invoke({
    "research_brief": "Compare hiking backpacks for 3+ day trips",
    "researcher_messages": []
})

supervisor_result = supervisor_agent.invoke({
    "research_brief": "Comprehensive analysis of hiking backpacks",
    "supervisor_messages": []
})
```

### Project structure

```text
src/
  graph.py              # Main research pipeline orchestration
  state.py              # Core state definitions (InputState, AgentState)
  schema.py             # Pydantic models for structured outputs
  utils.py              # Helper utilities (date formatting)

  nodes/
    clarify_user_request.py  # Initial clarification node
    write_research_brief.py  # Research brief generation node

  supervisor/           # Supervisor agent coordination
    supervisor.py       # Main supervisor agent implementation
    state.py           # Supervisor state management
    tools.py           # Supervisor tools (ConductResearch, ResearchComplete)
    prompt.py          # Supervisor system prompts
    utils.py           # Supervisor utilities

  research_agent/       # Individual research agents
    agent.py           # Research agent implementation
    state.py           # Research agent state management
    prompt.py          # Research agent prompts
    schema.py          # Research agent schemas
    tools/
      tavily/          # Web search integration
        tavily.py      # Tavily search tool
        utils.py       # Search result processing
        prompt.py      # Search-specific prompts
      think/           # Strategic thinking tools
        think.py       # Reflection and decision-making tool

langgraph.json          # CLI configuration (exposes all agents)
main.py                 # Entry point
notebook/              # Evaluation notebooks
  clarify_with_user_evals.ipynb
  research_phase_evals.ipynb
```

### How it works

The system operates through a sophisticated multi-agent pipeline:

#### 1. Initial Processing (`src/graph.py`)

- `clarify_user_request`: Analyzes the user's request and decides whether clarification is needed
- `write_research_brief`: Transforms the conversation into a detailed research brief using structured output

#### 2. Research Coordination (`src/supervisor/`)

- **Supervisor Agent**: Orchestrates the research process using `gpt-4.1`
- **Decision Making**: Analyzes research brief and decides what topics need investigation
- **Parallel Execution**: Launches multiple research agents simultaneously (up to 3 concurrent)
- **Tool Management**: Uses `ConductResearch` and `ResearchComplete` tools to coordinate activities

#### 3. Individual Research (`src/research_agent/`)

- **Research Agents**: Specialized agents using `gpt-4o-mini` for focused research
- **Web Search**: Integrated Tavily API for real-time web search capabilities
- **Strategic Thinking**: Built-in reflection tools for quality decision-making
- **Research Compression**: Intelligent summarization of findings for supervisor consumption

#### 4. State Management

- `InputState`: Compatible with LangGraph message flows
- `AgentState`: Extends state with research brief and supervisor messages
- `SupervisorState`: Manages supervisor coordination and research iterations
- `ResearcherState`: Handles individual research agent state

#### 5. Tool Integration

- **Tavily Search**: Web search with result deduplication and processing
- **Think Tool**: Strategic reflection for research quality
- **ConductResearch**: Delegates research tasks to specialized agents
- **ResearchComplete**: Signals research completion

### Configuration

`langgraph.json`:

```json
{
  "graphs": {
    "deep_research_agent": "src/graph.py:graph",
    "research_agent": "src/research_agent/agent.py:research_agent",
    "supervisor_agent": "src/supervisor/supervisor.py:supervisor_agent"
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

- "Model not found" or auth errors: ensure `OPENAI_API_KEY` and `TAVILY_API_KEY` are set and valid.
- Import errors for `langgraph`, `langchain-openai`, or `tavily-python`: re-run `uv sync` or `pip install -e .`.
- Dev UI doesn't show the graphs: confirm `langgraph.json` points to the correct graph definitions and that your venv is active.
- Research agents not finding results: check your Tavily API key and quota limits.
- Supervisor not launching research: verify the research brief is properly formatted and contains actionable research topics.

### License

Add your preferred license here.
