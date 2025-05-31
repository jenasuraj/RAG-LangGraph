# 📁 File: graph_builder.py
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from agents import agent1, agent2, judge
from llm_config import tools
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    agent1_data: dict
    agent2_data: dict

def custom_tools_condition(state: State):
    print("✅ in destination")
    from langgraph.prebuilt import tools_condition as real_tools_condition
    result = real_tools_condition(state)
    print("🔁 tools_condition result:", result)
    return result

graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("agent1", agent1)
graph_builder.add_node("agent2", agent2)
graph_builder.add_node("judge", judge)
graph_builder.add_node("tools", ToolNode(tools))

# Add edges
graph_builder.add_edge(START, "agent1")
graph_builder.add_conditional_edges("agent1", custom_tools_condition, {"tools": "tools", "__end__": "agent2"})
graph_builder.add_edge("tools", "agent1")
graph_builder.add_conditional_edges("agent2", tools_condition, {"tools": "tools", "__end__": "judge"})
graph_builder.add_edge("tools", "agent2")
graph_builder.add_edge("judge", END)

graph = graph_builder.compile()
