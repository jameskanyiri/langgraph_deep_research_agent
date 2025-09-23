from langgraph.graph import StateGraph, START, END
from src.state import AgentState, InputState
from src.nodes.clarify_user_request import clarify_user_request
from src.nodes.write_research_brief import write_research_brief
from src.supervisor.supervisor import supervisor_agent
from src.generate_report.generate_report import generate_report

deep_research_builder = StateGraph(AgentState, input_schema=InputState)

deep_research_builder.add_node("clarify_user_request", clarify_user_request)
deep_research_builder.add_node("write_research_brief", write_research_brief)
deep_research_builder.add_node("research_phase", supervisor_agent)
deep_research_builder.add_node("generate_report", generate_report)

deep_research_builder.add_edge(START, "clarify_user_request")

deep_research_builder.add_edge("research_phase", "generate_report")

graph = deep_research_builder.compile()