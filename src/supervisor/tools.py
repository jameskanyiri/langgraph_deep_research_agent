from langchain_core.tools import tool
from pydantic import BaseModel, Field

@tool
class ConductResearch(BaseModel):
    """
    Tool for delegating a research task to a specialized research agent.
    """
    research_topic: str = Field(description="TThe topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).")


@tool
class ResearchComplete(BaseModel):
    """
    Tool for indicating that the research process is complete.
    """
    pass
