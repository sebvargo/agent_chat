# Let's start from scratch for convinience
## Imports
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel
from typing import Annotated, TypedDict, Union
from functools import partial
import os

OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o')
ANTHROPIC_MODEL = os.environ.get('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20240620')

## Utilities
def _create_tool_response(response:str, ai_message:AIMessage):
    return ToolMessage(content=response, tool_call_id=ai_message.tool_calls[0]["id"])

def _stream_graph_updates(graph: StateGraph, user_input: str = None, config: dict = {"configurable":"1"}, stream_mode: str = "values"):
    """ Streams updates from a StateGraph based on user input and prints the assistant's responses. """
    if user_input:
        messages = {"messages":[("user", user_input)]}
    else:
        messages = None
    for event in graph.stream(messages, config=config, stream_mode=stream_mode):
        if "messages" in event:
            event["messages"][-1].pretty_print()

## State
# Define a Custom State
class CustomState(TypedDict):
    messages: Annotated[list[str], add_messages]
    ask_human: bool

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
    

## Edges
def select_next_node(state: CustomState):
    if state["ask_human"]:
        return "human"
    # Otherwise, we can route as before
    return tools_condition(state)

## Tools
# Web Search
web_search = TavilySearchResults(max_results=2) 
# Request Assistance
class RequestAssistance(BaseModel):
    """
    Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.
    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    
    Args:
        request (str): The user's request for assistance.
    """
    request: str

tools = [web_search, RequestAssistance]

## Select LLM
llm = ChatAnthropic(model=ANTHROPIC_MODEL)
print(f"Anthropic model: {ANTHROPIC_MODEL}")
# llm = ChatOpenAI(model=OPENAI_MODEL)
# print(f"OpenAI model: {OPENAI_MODEL}")    
llm_with_tools = llm.bind_tools(tools)


## Graph
# Define the StateGraph
builder = StateGraph(CustomState)

# Add nodes
builder.add_node("chatbot", partial(chatbot, llm=llm_with_tools))
builder.add_node("human", human_node)
builder.add_node("tools", ToolNode(tools=tools))

# Add edges
builder.add_edge(START, "chatbot")
builder.add_edge("tools", "chatbot")
builder.add_edge("human", "chatbot")
builder.add_conditional_edges(
    "chatbot", 
    select_next_node, 
    {
        "human": "human",
        "tools": "tools",
        END: END
    }
    )

# Add memory and compile graph
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["human"])

