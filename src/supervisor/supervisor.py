from langchain_core.messages import (
    BaseMessage,
    filter_messages,
    SystemMessage,
    ToolMessage,
    HumanMessage,
)
import asyncio
from typing import Literal
from src.supervisor.tools import ConductResearch, ResearchComplete
from langchain.chat_models import init_chat_model
from src.supervisor.state import SupervisorState
from langgraph.types import Command
from src.supervisor.prompt import SUPERVISOR_PROMPT
from src.utils import get_today_str
from langgraph.graph import StateGraph, END, START
from src.research_agent.tools.think.think import think_tool
from src.research_agent.agent import research_agent
from src.supervisor.utils import get_notes_from_tool_calls


# Configuration
supervisor_tools = [ConductResearch, ResearchComplete, think_tool]
supervisor_model = init_chat_model(model="openai:gpt-4.1")
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)

# Maximum number of tool call iterations for individual research agents
# This is to prevent infinite loops and controls research depth per topic
MAX_RESEARCH_ITERATIONS = 6

# maximum number of concurrent research agents the supervisor can launch
# This is passed to the supervisor prompt to control the number of concurrent research agents
MAX_CONCURRENT_RESEARCH_AGENTS = 3


async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """
    Coordinates research activities.

    Analyze the research brief and current progress to decide:
        - What research topic need investigation
        - Whether to conduct parallel research
        - When research is complete

    Args:
        state: Current supervisor state with messages and research progress

    Returns:
        Command to proceed to supervisor_tools node with the updates state
    """

    supervisor_messages = state.get("supervisor_messages", [])

    # System Instruction
    system_instruction = SUPERVISOR_PROMPT.format(
        date=get_today_str(),
        max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_AGENTS,
        max_researcher_iterations=MAX_RESEARCH_ITERATIONS,
    )

    messages = [SystemMessage(content=system_instruction)] + supervisor_messages

    # make the decision about the next research steps
    response = await supervisor_model_with_tools.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


async def supervisor_tools(
    state: SupervisorState,
) -> Command[Literal["supervisor", END]]:
    """
    Execute supervisor decisions
        Either conduct research or complete the research

    Handles:
        - Executing think_tool calls for strategic reflection.
        - Launching parallel research agents for different topics
        - Aggregating research findings from sub-agents
        - Determining when research is complete

    Args:
        state: Current supervisor state with messages and iteration count

    Returns:
        Command to continue supervision, end process or handle errors
    """

    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # Initialize variables for single return pattern
    tool_messages = []
    all_raw_notes = []
    next_node = "supervisor"
    should_stop = False

    # Check if the max iteration limit has been reached
    exceeded_iterations = research_iterations >= MAX_RESEARCH_ITERATIONS

    no_tool_calls = not most_recent_message.tool_calls

    research_complete = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or no_tool_calls or research_complete:
        should_stop = True
        next_node = END

    else:
        # Excute all the tool call before deciding the next step
        try:
            # Separate think tool calls from ConductResearch tool calls
            think_tool_calls = [
                tool_call
                for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "think_tool"
            ]

            conduct_research_calls = [
                tool_call
                for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "ConductResearch"
            ]

            # Handle think tool calls
            for tool_call in think_tool_calls:
                observation = think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                )

            # Handle ConductResearch tool calls
            if conduct_research_calls:
                # Launch parallel research agents
                coros = [
                    research_agent.ainvoke(
                        {
                            "researcher_messages": [
                                HumanMessage(
                                    content=tool_call["args"]["research_topic"]
                                )
                            ],
                            "research_brief": tool_call["args"]["research_topic"],
                        }
                    ) for tool_call in conduct_research_calls
                ]

                # Waite for all the research to complete
                tool_results = await asyncio.gather(*coros)

                # Format the research results as tool messages
                # Each sub agent returns compressed research finding in result['compressed_research']
                # We write the compressed research as the content of a ToolMessage, which allows
                # The supervisor to later retrieve these findings via get_notes_from_tool_calls
                research_tool_messages = [
                    ToolMessage(
                        content=result.get(
                            "compressed_research", "Error synthesizing research report"
                        ),
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                    for result, tool_call in zip(tool_results, conduct_research_calls)
                ]

                tool_messages.extend(research_tool_messages)

                # Aggregate the research findings from the research agents
                all_raw_notes = [
                    "\n\n".join(result.get("raw_notes", [])) for result in tool_results
                ]

        except Exception as e:
            print(f"Error executing tool calls: {e}")
            should_stop = True
            next_node = END

    # Single return point with appropriate state updates
    if should_stop:
        return Command(
            goto=next_node,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
            },
        )
    else:
        return Command(
            goto=next_node,
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": all_raw_notes,
            },
        )


supervisor_graph = StateGraph(SupervisorState)
supervisor_graph.add_node("supervisor", supervisor)
supervisor_graph.add_node("supervisor_tools", supervisor_tools)

supervisor_graph.add_edge(START, "supervisor")


supervisor_agent = supervisor_graph.compile()
