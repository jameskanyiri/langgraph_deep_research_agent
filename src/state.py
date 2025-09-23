#State definitio
from langgraph.graph.message import MessagesState
from langchain_core.messages import BaseMessage
from typing import Annotated, Sequence
from langgraph.graph.message import add_messages
import operator


class InputState(MessagesState):
    pass


class AgentState(MessagesState):
    
    #Research brief created from the conversation with user
    research_brief: str
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    #Processed and structured nodes ready for final report generation
    notes: Annotated[list[str], operator.add] = []
    #Raw notes collected from the sub agents
    raw_noted: Annotated[list[str], operator.add] = []
    # Final formatted research report
    final_report: str