import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END


load_dotenv()

# ---- LLMs ----
writer_llm = ChatOpenAI(
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

summarizer_llm = ChatOpenAI(
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



class State(TypedDict):
    user_input: str
    writer_output: str
    summary_output: str




# ---- Agent 1: Writer ----
writer_prompt = PromptTemplate.from_template("You have to write a content about: {question}")
writer_chain = writer_prompt | writer_llm
def writer(state: State) -> dict:
    question = state["user_input"]
    result = writer_chain.invoke({"question": question})
    return {"writer_output": result.content}




# ---- Agent 2: Summarizer ----
summarizer_prompt = PromptTemplate.from_template("Summarise this in 50 words: {answer}")
summarizer_chain = summarizer_prompt | summarizer_llm
def summarizer(state: State) -> dict:
    answer = state["writer_output"]
    result = summarizer_chain.invoke({"answer": answer})
    return {"summary_output": result.content}






graph_builder = StateGraph(State)
graph_builder.add_node("writer", writer)
graph_builder.add_node("summarizer", summarizer)
graph_builder.set_entry_point("writer")
graph_builder.add_edge("writer", "summarizer")
graph_builder.add_edge("summarizer", END)
graph = graph_builder.compile()



# ---- Execution ----
def run_graph(user_input: str):
    state = {"user_input": user_input}
    final_state = graph.invoke(state)
    print("\n📝Final Summary:\n", final_state["summary_output"])

# ---- REPL ----
if __name__ == "__main__":
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        run_graph(user_input)
