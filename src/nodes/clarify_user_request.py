from src.state import AgentState
from langgraph.types import Command
from typing import Literal
from src.schema import ClarifyUserRequest
from langchain_core.messages import get_buffer_string,AIMessage
from langchain.chat_models import init_chat_model
from src.utils import get_today_str
from langgraph.graph import END

# Initialize the LLM and the structured LLM
llm = init_chat_model(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(ClarifyUserRequest)


# System instruction
AGENT_SYSTEM_INSTRUCTION = """
These are the messages that have been exchanged so far from the user asking for the report:                    
 <Messages>                                                                                                     
 {messages}                                                                                                     
 </Messages>                                                                                                    
                                                                                                                
Today's date is {date}.                                                                                        
                                                                                                                
 Assess whether you need to ask a clarifying question, or if the user has already provided enough information   
 for you to start research.                                                                                     
 IMPORTANT: If you can see in the messages history that you have already asked a clarifying question, you       
 almost always do not need to ask another one. Only ask another question if ABSOLUTELY NECESSARY.               
                                                                                                                
 If there are acronyms, abbreviations, or unknown terms, ask the user to clarify.                               
 If you need to ask a question, follow these guidelines:                                                        
 - Be concise while gathering all necessary information                                                         
 - Make sure to gather all the information needed to carry out the research task in a concise, well-structured  
 manner.                                                                                                        
 - Use bullet points or numbered lists if appropriate for clarity. Make sure that this uses markdown            
 formatting and will be rendered correctly if the string output is passed to a markdown renderer.               
 - Don't ask for unnecessary information, or information that the user has already provided. If you can see     
 that the user has already provided the information, do not ask for it again.                                   
                                                                                                                
 Respond in valid JSON format with these exact keys:                                                            
 "need_clarification": boolean,                                                                                 
 "question": "<question to ask the user to clarify the report scope>",                                          
 "verification": "<verification message that we will start research>"                                           
                                                                                                                
 If you need to ask a clarifying question, return:                                                              
 "need_clarification": true,                                                                                    
 "question": "<your clarifying question>",                                                                      
 "verification": ""                                                                                             
                                                                                                                
 If you do not need to ask a clarifying question, return:                                                       
 "need_clarification": false,                                                                                   
 "question": "",                                                                                                
 "verification": "<acknowledgement message that you will now start research based on the provided               
 information>"                                                                                                  
                                                                                                                
 For the verification message when no clarification is needed:                                                  
 - Acknowledge that you have sufficient information to proceed                                                  
 - Briefly summarize the key aspects of what you understand from their request                                  
 - Confirm that you will now begin the research process                                                         
 - Keep the message concise and professional    
"""

# Node function
def clarify_user_request(
    state: AgentState,
) -> Command[Literal[END, "write_research_brief"]]:
    """
    This node will be used to check if the user request contains enough information to start the research

    It used structured output to make deterministic decision and avoid hallucination

    If the user request contain enough information router to generate research brief
    If the user request does not contain enough information router to end with clarification question
    """
    
    list_of_messages = state['messages']
    
    system_intruction = AGENT_SYSTEM_INSTRUCTION.format(
        messages=get_buffer_string(list_of_messages),
        date=get_today_str(),
    )
    
    messages = [
        {
            "role": "system",
            "content": system_intruction,
        }
    ]
    
    response = structured_llm.invoke(messages)
    
    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )
    
    
    
