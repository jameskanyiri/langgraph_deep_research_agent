from langgraph.graph import StateGraph, START, END
from src.state import AgentState, InputState
from src.nodes.clarify_user_request import clarify_user_request
from src.nodes.write_research_brief import write_research_brief

deep_research_builder = StateGraph(AgentState, input_schema=InputState)

deep_research_builder.add_node("clarify_user_request", clarify_user_request)
deep_research_builder.add_node("write_research_brief", write_research_brief)

deep_research_builder.add_edge(START, "clarify_user_request")

graph = deep_research_builder.compile()