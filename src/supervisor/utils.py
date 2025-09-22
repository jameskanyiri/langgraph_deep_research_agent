from langchain_core.messages import BaseMessage, filter_messages


def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """
    Extract research notes from ToolMessage objects in supervisor message history
    
    This function retrieves the compressed research findings that sub-agents returns as ToolMessage content.
    When the supervisor delegates research to sub-agents via ConductResearch tool calls, each sub-agent returns its compressed findings as the content of a ToolMessage.
    This function extracts all such ToolMessage content to compile the final research notes.
    
    Args:
        messages: List of messages from the supervisor's conversation history
    
    Returns:
        List of research note string extracted from ToolMessage objects
    """
    
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]
    