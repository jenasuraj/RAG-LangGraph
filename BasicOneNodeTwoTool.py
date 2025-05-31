import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv()

# Create LLMs
def create_llm():
    return ChatOpenAI(
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

agent1_llm = create_llm()
agent2_llm = create_llm()
judge_llm = create_llm()

# Tool
tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools_1 = agent1_llm.bind_tools(tools)
llm_with_tools_2 = agent2_llm.bind_tools(tools)

# State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Agent 1 node
def agent1(state: State):
    print("🧠 Agent 1 responding...")
    history = state["messages"]
    response = llm_with_tools_1.invoke(history)
    return {"messages": [response]}

# Agent 2 node
def agent2(state: State):
    print("🧠 Agent 2 responding...")
    history = state["messages"]
    response = llm_with_tools_2.invoke(history)
    return {"messages": [response]}

# Judge node: decides which agent's response is better
def judge(state: State):
    print("⚖️ Judge evaluating...")

    messages = state["messages"]
    human_message = next(msg for msg in messages if isinstance(msg, HumanMessage))
    agent_responses = [msg for msg in messages if isinstance(msg, AIMessage)]

    if len(agent_responses) < 2:
        raise ValueError("Expected at least two AI responses from agent1 and agent2")

    agent1_answer = agent_responses[0].content
    agent2_answer = agent_responses[1].content

    judge_prompt = PromptTemplate.from_template(
        """You are a judge in a debate between two AI agents.
        The user asked: "{question}"
        
        Agent 1 responded:
        {agent1_response}
        
        Agent 2 responded:
        {agent2_response}
        
        Based on helpfulness, accuracy, and completeness, decide which agent gave the better answer.
        Respond in this format ONLY:
        Winner: Agent 1 or Agent 2
        Reason: <your justification>
        """
    )

    prompt = judge_prompt.format(
        question=human_message.content,
        agent1_response=agent1_answer,
        agent2_response=agent2_answer
    )

    judge_response = judge_llm.invoke(prompt)
    print(judge_response.content)

    return {"messages": [AIMessage(content=judge_response.content)]}

# Build LangGraph
graph_builder = StateGraph(State)
graph_builder.add_node("agent1", agent1)
graph_builder.add_node("agent2", agent2)
graph_builder.add_node("judge", judge)
graph_builder.add_node("tools", ToolNode(tools=tools))

# Flow
graph_builder.add_edge(START, "agent1")
graph_builder.add_conditional_edges("agent1", tools_condition)
graph_builder.add_edge("agent1", "agent2")
graph_builder.add_conditional_edges("agent2", tools_condition)
graph_builder.add_edge("agent2", "judge")
graph_builder.add_edge("judge", END)

graph = graph_builder.compile()

# Run interaction loop
def stream_graph_updates(user_input: str):
    initial_message = {"messages": [HumanMessage(content=user_input)]}
    for event in graph.stream(initial_message):
        for value in event.values():
            last_msg = value["messages"][-1]
            print("Assistant:", last_msg.content)

# CLI
if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
