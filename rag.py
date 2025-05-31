import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_tavily import TavilySearch
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
    max_tokens=100
)

tavily_tool = TavilySearch(max_results=2)
tools = [tavily_tool]
llm_with_tools = llm.bind_tools(tools)

loader = PyPDFLoader(r"C:\Users\LENOVO\OneDrive\Desktop\nike.pdf")
documents = loader.load()
print("Loading pdf is done")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print("splitting into chunks is done")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=chunks)
print("data added to v-db")
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
     #print("\n\n the state in chatbot looks like",state["messages"])
     userQuery = state["messages"][-1].content
     #print("\n\n user query is",userQuery)
     responseTool = llm_with_tools.invoke(userQuery)
     #print("\n\n llm-bind-response -->",responseTool)
     return {
          "messages":state["messages"]+[responseTool]
            }



def rag(state:State):
     #print("\n\n in rag the state is ",state["messages"])
     user_query = state["messages"][0].content
     existing_response = state["messages"][-1].content
     vdb_response = retriever.invoke(user_query)
     response = '/n/n'.join([i.page_content for i in vdb_response])
     system_prompt = (
        "You are a helpful assistant. Use the following context to answer the question:\n\n"
        f"The rag response is :{response}\n\n while the existing response by tools/llm is: {existing_response}"
        "Now answer the user's question."
    )
     result = llm.invoke([
     {"role": "system", "content": system_prompt},
     {"role": "user", "content": user_query}
     ])
     #print("\n\n rag result is ",result)
     return{
          "messages":state["messages"]+[result]
     }
     


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("rag",rag)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot",tools_condition,{
     "tools":"tools","__end__":"rag"
})
graph_builder.add_edge("rag",END)
graph_builder.add_edge("tools","chatbot")



graph = graph_builder.compile()
def stream_graph_updates(user_input):
    initial_state = {
        "messages": [{"role": "user", "content": user_input}]
    }
    result = graph.invoke(initial_state)
    print(result["messages"][-1].content)  
while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
