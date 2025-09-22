#State definitio
from langgraph.graph.message import MessagesState
from langchain_core.messages import BaseMessage
from typing import Annotated, Sequence
from langgraph.graph.message import add_messages


class InputState(MessagesState):
    pass


class AgentState(MessagesState):
    
    #Research brief created from the conversation with user
    research_brief: str
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]