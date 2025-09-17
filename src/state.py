#State definitio
from langgraph.graph.message import MessagesState


class InputState(MessagesState):
    pass


class AgentState(MessagesState):
    
    #Research brief created from the conversation with user
    research_brief: str