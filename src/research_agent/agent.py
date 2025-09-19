from src.research_agent.tools.tavily.tavily import tavily_search
from src.research_agent.tools.think.think import think_tool
from langchain.chat_models import init_chat_model
from src.research_agent.state import (
    ResearcherState,
    ResearcherInputState,
    ResearcherOutputState,
)
from src.research_agent.prompt import (
    RESEARCH_AGENT_PROMPT,
    COMPRESS_RESEARCH_SYSTEM_PROMPT,
    COMPRESS_RESEARCH_HUMAN_PROMPT,
)
from src.utils import get_today_str
from langchain_core.messages import ToolMessage, filter_messages
from typing import Literal
from langgraph.graph import StateGraph, END

# Get all the tools
tools = [tavily_search, think_tool]

tools_by_name = {tool.name: tool for tool in tools}

# Initialize Models
model = init_chat_model(model="openai:gpt-4o-mini", temperature=0)

model_with_tools = model.bind_tools(tools)

summarization_model = init_chat_model(model="openai:gpt-4o-mini", temperature=0)

compress_model = init_chat_model(model="gpt-4.1", temperature=0, max_tokens=32000)


# Agent Node
def agent(state: ResearcherState):
    """
    This node analyzes the current state and decide on the next action to take.

    The model will analyze the current conversation state and decide whether to:
    1. Call a search tool to gather more information
    2. Provide a final answer based on gathered information

    Return updated state with the model's response.
    """
    
    #Get the messages from the state
    messages = state["researcher_messages"]
    
    #If the messages are empty, add the research brief as the first message
    if len(messages) == 0:
        first_message = state['research_brief']
        messages = [{"role": "human", "content": first_message}]
    else:
        messages = messages

    system_instruction = RESEARCH_AGENT_PROMPT.format(date=get_today_str())

    response = model_with_tools.invoke([{"role": "system", "content": system_instruction}] + messages)

    return {"researcher_messages": [response]}


# Define tool node
def tool_node(state: ResearcherState):
    """
    This node will execute the tool calls based on the model's decision.

    Execute all tool calls from the previous LLM response.
    Return updated state with tool execution results.

    """

    tool_calls = state["researcher_messages"][-1].tool_calls

    tool_results = []

    # Execute all tool calls
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        tool_results.append(tool.invoke(tool_call["args"]))

    # Create a tool message outputs
    tool_response = [
        ToolMessage(
            content=result, name=tool_call["name"], tool_call_id=tool_call["id"]
        )
        for result, tool_call in zip(tool_results, tool_calls)
    ]

    return {"researcher_messages": tool_response}


# Define summarization node
def compress_research(state: ResearcherState):
    """
    Compress research finding into a concise summary.

    Takes all the research messages and tool responses and creates
    a compressed summary suitable for the supervisors decision making.

    """
    #Get the first message from the messages list
    first_message = state["researcher_messages"][0]
    research_topic = first_message.content

    system_instruction = COMPRESS_RESEARCH_SYSTEM_PROMPT.format(date=get_today_str())

    human_instruction = COMPRESS_RESEARCH_HUMAN_PROMPT.format(
        research_topic=research_topic
    )

    messages = (
        [{"role": "system", "content": system_instruction}]
        + state["researcher_messages"]
        + [{"role": "user", "content": human_instruction}]
    )

    response = compress_model.invoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content)
        for m in filter_messages(
            state["researcher_messages"], include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)],
    }


# Should continue node
def should_continue(
    state: ResearcherState,
) -> Literal["tools", "compress_research"]:
    """
    This node will decide whether to continue the research process or provide a final answer.

    Determines whether the agent should continue the research loop or provide a final answer based on whether the llm made tool calls.

    Returns:
        "tools: Continue to tool execution
        "compress_research: Stop and compress the research
    """

    messages = state["researcher_messages"]

    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"
    else:
        return "compress_research"


# Agent builder
research_agent_builder = StateGraph(
    ResearcherState,
    input_schema=ResearcherInputState,
    output_schema=ResearcherOutputState,
)

# Add nodes
research_agent_builder.add_node("agent", agent)
research_agent_builder.add_node("tools", tool_node)
research_agent_builder.add_node("compress_research", compress_research)

# Add edges
research_agent_builder.set_entry_point("agent")
research_agent_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "compress_research": "compress_research",
    },
)

research_agent_builder.add_edge("tools", "agent")

research_agent_builder.set_finish_point("compress_research")

research_agent = research_agent_builder.compile()
