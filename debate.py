import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


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

class State(TypedDict):
    messages: Annotated[list, add_messages]
    agent1_data: dict 
    agent2_data: dict  



def agent1(state: State):
    print("✅ Before anything, the state is", state) 
    print("🧠 Agent 1 Started reasoning...")
    messages = state["messages"]
    prompt = messages + [
        SystemMessage(content="You are Agent 1 in a debate. Provide a concise, compelling argument supporting your position. Use facts if available.")
    ]
    result = llm_with_tools.invoke(prompt)
    print("[Agent1 Response]:", result) #Working well
    state["agent1_data"] = result 
    print("✅ Now the agent1_data dict has", state["agent1_data"])
    return {
        "messages": state["messages"] + [result], 
        "agent1_data": state["agent1_data"], }





def agent2(state: State):
    print("After Agent 1, the state becomes", state)
    print("🧠 Agent 2 Started reasoning...")
    messages = state["messages"]
    prompt = messages + [
        SystemMessage(content="You are Agent 2 in a debate.give a good detailed response")
    ]
    result = llm_with_tools.invoke(prompt)
    print("[Agent2 Response]:", result)
    state["agent2_data"] = result 
    print("✅ Now the agent2_data dict has", state["agent2_data"])
    return {
        "messages": state["messages"] + [result],  # Append, don't overwrite
        "agent2_data": state["agent2_data"], 
    }





def judge(state: State):
    print("After Agent 2, the state becomes", state)
    print("⚖️ Judge Started evaluating...")
    agent1_data = state["agent1_data"]
    agent2_data = state["agent2_data"]
    prompt = [
        SystemMessage(
            content=(
                "You are a neutral Judge in a debate. Evaluate the arguments from Agent 1 and Agent 2 based on clarity, logic, and persuasiveness. "
                "Decide which agent wins and provide a brief explanation. "
                "Format your response as: 'Winner: [Agent 1 or Agent 2]. Reason: [Your reasoning].'"
            )
        ),
        AIMessage(content=f"Agent 1 Argument: {agent1_data}"),
        AIMessage(content=f"Agent 2 Argument: {agent2_data}")
    ]
    result = llm.invoke(prompt)  # Use plain LLM, no tools needed
    print("[Judge Decision]:", result.content)
    return {
        "messages": state["messages"] + [result],  # Append decision to messages
    }


graph_builder = StateGraph(State)



def custom_tools_condition(state: State):
    print("✅in destination")
    from langgraph.prebuilt import tools_condition as real_tools_condition
    result = real_tools_condition(state)
    print("🔁 tools_condition result:", result)
    return result


# Add nodes
graph_builder.add_node("agent1", agent1)
graph_builder.add_node("agent2", agent2)
graph_builder.add_node("judge", judge)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent1")
graph_builder.add_conditional_edges(
    "agent1",
    custom_tools_condition,
    {"tools": "tools", "__end__": "agent2"}  
)
graph_builder.add_edge("tools", "agent1")
graph_builder.add_conditional_edges(
    "agent2",
    tools_condition,
    {"tools": "tools", "__end__": "judge"}  
)
graph_builder.add_edge("tools", "agent2")
graph_builder.add_edge("judge", END)
graph = graph_builder.compile()





def stream_graph_updates(user_input: str):
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "agent1_data": {},
        "agent2_data": {}
    }
    for event in graph.stream(initial_state):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    stream_graph_updates(user_input)