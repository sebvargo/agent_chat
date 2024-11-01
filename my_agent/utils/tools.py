from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel

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