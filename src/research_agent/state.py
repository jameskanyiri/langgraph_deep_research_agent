from typing import Annotated, TypedDict, Sequence, List
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearcherInputState(TypedDict):
    """
    State for the research agent containing message history and research metadata
    """
    research_brief: str
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    


class ResearcherState(TypedDict):
    """
    State for the research agent containing message history and research metadata

    This state will track the researcher conversation, iteration count for limiting the number of tool calls,
    the research topic being investigated, compressed findings, and raw research notes for detailed analysis.
    """
    research_brief: str
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]


class ResearcherOutputState(TypedDict):
    """
    The Output state for the research agent containing final research results.

    This represent the final output of the research process with compressed research findings and all raw notes from the research process.
    """

    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
