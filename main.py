import time

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """ schema of the source used by agent"""
    url:str = Field(description="the url of the source")
    reason:str = Field(description="reason for selecting the source")

class AgentResponse(BaseModel):
    """agent response with  answer and sources"""
    answer:str = Field(description="agent's response")
    sources: List[Source] = Field(default_factory=list, description="List of sources")

load_dotenv()
llm: ChatOpenAI = ChatOpenAI(model='gpt-5')
tools: list[Tool] = [TavilySearch()]
agent = create_agent(llm, tools, response_format=AgentResponse)

def main():
    print("starting the search...")
    result = agent.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "find scores for yesterday's football world cup matches that feature an african team"
                )
            )
        ]
    })
    print(result)


if __name__ == "__main__":
    main()
