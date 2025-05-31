# 📁 File: agents.py
from langchain_core.messages import SystemMessage, AIMessage
from llm_config import llm_with_tools, llm

def agent1(state):
    print("✅ Before anything, the state is", state)
    print("🧠 Agent 1 Started reasoning...")
    messages = state["messages"]
    prompt = messages + [
        SystemMessage(content="You are Agent 1 in a debate. Provide a concise, compelling argument supporting your position. Use facts if available.")
    ]
    result = llm_with_tools.invoke(prompt)
    print("[Agent1 Response]:", result)
    state["agent1_data"] = result
    return {
        "messages": state["messages"] + [result],
        "agent1_data": state["agent1_data"],
    }

def agent2(state):
    print("After Agent 1, the state becomes", state)
    print("🧠 Agent 2 Started reasoning...")
    messages = state["messages"]
    prompt = messages + [
        SystemMessage(content="You are Agent 2 in a debate. Give a good detailed response.")
    ]
    result = llm_with_tools.invoke(prompt)
    print("[Agent2 Response]:", result)
    state["agent2_data"] = result
    return {
        "messages": state["messages"] + [result],
        "agent2_data": state["agent2_data"],
    }

def judge(state):
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
    result = llm.invoke(prompt)
    print("[Judge Decision]:", result.content)
    return {
        "messages": state["messages"] + [result],
    }
