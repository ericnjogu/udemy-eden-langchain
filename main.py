import time

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI
import json
from langchain_tavily import TavilySearch

load_dotenv()
llm: ChatOpenAI = ChatOpenAI(model='gpt-5')
tools: list[Tool] = [TavilySearch()]
agent = create_agent(llm, tools)

def main():
    print("starting the search...")
    result = agent.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "find 2 recipes using precooked cornmeal for vegan meals"
                )
            )
        ]
    })
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
