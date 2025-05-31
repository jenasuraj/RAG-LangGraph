import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage


# Load environment variables
load_dotenv()

# LLM config
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


# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    data:str

data = State["messages"][0].content
print("the data is from user is",data)    



# 1️⃣ ChatBot Node (refines the user input)
chat_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Refine the user's input below to make it clearer and ready for processing.

User Input: {user}
Refined Input:
""")
chat_chain = chat_prompt | llm

def chatBot(state: State):
    user = state["messages"][-1].content
    response = chat_chain.invoke({"user": user})
    return {"messages": state["messages"] + [AIMessage(content=response.content)]}


def route(state: State):
    refined = state["messages"][-1].content.lower()
    if any(word in refined for word in ["solve", "math", "calculate"]):
        return "MathSolver"
    elif "grammar" in refined or "grammer" in refined:
        return "GrammerChecker"
    else:
        return "LLM"


# 3️⃣ MathSolver Node
math_prompt = PromptTemplate.from_template("""
You are a math solver. Solve the problem: {user}
""")
math_chain = math_prompt | llm

def MathSolver(state: State):
    user = state["messages"][-1].content
    response = math_chain.invoke({"user": user})
    return {"messages": state["messages"] + [AIMessage(content=response.content)]}

# 4️⃣ GrammerChecker Node
grammar_prompt = PromptTemplate.from_template("""
You are a grammar checker. Correct the following sentence: {user}
Return the corrected version only.
""")
grammar_chain = grammar_prompt | llm

def GrammerChecker(state: State):
    user = state["messages"][-1].content
    response = grammar_chain.invoke({"user": user})
    return {"messages": state["messages"] + [AIMessage(content=response.content)]}

# 5️⃣ Fallback LLM Node
def LLM(state: State):
    user = state["messages"][-1].content
    response = llm.invoke(user)
    return {"messages": state["messages"] +[AIMessage(content=response.content)]}


graphBuilder = StateGraph(State)
graphBuilder.add_node("chatBot", chatBot)
graphBuilder.add_node("MathSolver", MathSolver)
graphBuilder.add_node("GrammerChecker", GrammerChecker)
graphBuilder.add_node("LLM", LLM)

graphBuilder.set_entry_point("chatBot")
graphBuilder.add_conditional_edges("chatBot", route, {
    "MathSolver": "MathSolver",
    "GrammerChecker": "GrammerChecker",
    "LLM": "LLM"
})
graphBuilder.add_edge("MathSolver", END)
graphBuilder.add_edge("GrammerChecker", END)
graphBuilder.add_edge("LLM", END)
graph = graphBuilder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
  