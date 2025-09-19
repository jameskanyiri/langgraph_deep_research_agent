from pydantic import BaseModel,Field

class Summary(BaseModel):
    """Schema for webpage content summarization"""
    summary: str = Field(description="A concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the webpage content")