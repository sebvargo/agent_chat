from langgraph.prebuilt import tools_condition
from my_agent.utils.state import CustomState

## Edges
def select_next_node(state: CustomState):
    if state["ask_human"]:
        return "human"
    # Otherwise, we can route as before
    return tools_condition(state)
