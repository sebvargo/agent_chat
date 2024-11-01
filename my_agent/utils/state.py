from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated


## State
# Define a Custom State
class CustomState(TypedDict):
    messages: Annotated[list[str], add_messages]
    ask_human: bool