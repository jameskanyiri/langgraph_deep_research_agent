from src.state import AgentState
from src.generate_report.prompt import FINAL_REPORT_GENERATION_PROMPT
from src.utils import get_today_str
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

writer_model = init_chat_model(model="openai:gpt-5-nano")


async def generate_report(state: AgentState):
    """This node is response for writing the final research report."""
    
    notes = state.get("notes", [])
    
    research_brief = state.get("research_brief", "")
    
    findings= "\n".join(notes)
    
    system_instruction = FINAL_REPORT_GENERATION_PROMPT.format(
        research_brief=research_brief,
        date=get_today_str(),
        findings=findings,
    )
    
    final_report = await writer_model.ainvoke([HumanMessage(content=system_instruction)])
    
    return {
        "final_report": final_report.content,
        "messages": ["Here is the final report: " + final_report.content],
    }
    
    
    
    
    