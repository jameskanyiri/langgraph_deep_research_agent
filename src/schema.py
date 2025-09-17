# Structured output schema

from pydantic import BaseModel, Field


class ClarifyUserRequest(BaseModel):
    need_clarification: bool = Field(
        description="Whether the user needs clarification on the request. True if the user needs clarification, False otherwise."
    )
    question: str = Field(
        description="A question to ask the user to help clarify the report scope."
    )
    verification: str = Field(
        description="A verification message that research will start after the user has provided the necessary information"
    )


class WriteResearchBrief(BaseModel):
    research_brief: str = Field(
        description="A research brief that will be used to guide the research. It should be well detailed"
    )