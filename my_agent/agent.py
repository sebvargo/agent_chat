# Let's start from scratch for convinience
## Imports
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from my_agent.utils.edges import select_next_node
from my_agent.utils.nodes import chatbot, human_node
from my_agent.utils.state import CustomState
from my_agent.utils.tools import tools
from functools import partial
import os


# Load environment variables
load_dotenv()

OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o')
ANTHROPIC_MODEL = os.environ.get('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20240620')

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
# builder.add_edge("human", "chatbot")
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

