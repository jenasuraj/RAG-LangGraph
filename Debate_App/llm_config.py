import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "LangGraph Debate Agent with Judge"
    },
    temperature=0.7,
    max_tokens=300
)

tavily_tool = TavilySearch(max_results=2)
tools = [tavily_tool]
llm_with_tools = llm.bind_tools(tools)
