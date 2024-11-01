from langchain_anthropic import ChatAnthropic   
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from my_agent.utils.state import CustomState
from my_agent.utils.tools import RequestAssistance
from typing import Union

## Utilities
def _create_tool_response(response:str, ai_message:AIMessage):
    return ToolMessage(content=response, tool_call_id=ai_message.tool_calls[0]["id"])

## Nodes
# Chatbot
def chatbot(state: CustomState, llm: Union[ChatAnthropic, ChatOpenAI]):
    response = llm.invoke(state['messages'])
    ask_human = False
    if ( response.tool_calls and response.tool_calls[0]["name"] == RequestAssistance.__name__ ):
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}

# Human
def human_node(state: CustomState):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messages.append(_create_tool_response("No response from human", state["messages"][-1]))
    return {"messages": new_messages, "ask_human": False}
