from typing import TypedDict, Annotated, Sequence
import operator

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SupervisorState(TypedDict):
    #This will hold the messages exchanges with the supervisor for coordination and decision making
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    #This is the research brief from the scoping phase to guide the research direction
    research_brief: str
    # Counter tracking the number of research iterations performed
    research_iterations: int = 0
    #Processed and structured nodes ready for final report generation
    notes: Annotated[list[str], operator.add] = []
    #Raw notes collected from the sub agents
    raw_noted: Annotated[list[str], operator.add] = []