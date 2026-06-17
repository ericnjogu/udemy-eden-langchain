import time

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI
import json

load_dotenv()

@tool
def search(query: str) -> str:
    """
    a tool to search the web for information
    Args:
        query: the query to search for
    Returns:
        the search results
    """
    print(f"Searching the web for: {query}")
    return "the weather in Bukavu is rainy"

llm: ChatOpenAI = ChatOpenAI()
tools: list[Tool] = [search]
agent = create_agent(llm, tools)

def main():
    print("starting the search...")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Bukavu?")]})
    print(json.dumps(
        {"messages": [m.model_dump() for m in result["messages"]]},
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
