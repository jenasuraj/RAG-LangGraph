import os
import requests
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "LangGraph MultiTool Agent"
    },
    temperature=0.7,
    max_tokens=300
)


tavily_tool = TavilySearch(max_results=2)
tools = [tavily_tool]
llm_with_tools = llm.bind_tools(tools)



class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str



def chatbot(state: State):
    result = llm_with_tools.invoke(state["messages"])
    print("from bind",result)
    return {"messages": [result]}



graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)


memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)





def stream_graph_updates(user_input: str, thread_id: str = "default_thread"):
    config = {"configurable": {"thread_id": thread_id}}
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config=config):
        for node, state in event.items():
            if "messages" in state:
                messages = state["messages"]
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, "content"):
                        print("Assistant:", last_message.content)



while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input, thread_id="default_thread")
    except Exception as e:
        print(f"Error: {e}")
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input, thread_id="default_thread")
        break