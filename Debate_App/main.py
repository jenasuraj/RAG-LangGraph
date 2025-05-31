# 📁 File: main.py
from langchain_core.messages import HumanMessage
from graph_builder import graph 


def stream_graph_updates(user_input: str):
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "agent1_data": {},
        "agent2_data": {}
    }
    for event in graph.stream(initial_state):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)