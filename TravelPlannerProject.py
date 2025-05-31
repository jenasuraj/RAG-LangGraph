import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph

# Load API key
load_dotenv()

# ---- LLM Setup ----
planner_llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "LangGraph Travel Planner"
    },
    temperature=0.7,
    max_tokens=1000,
)

# ---- State Structure ----
class TravelState(TypedDict):
    destination: str
    total_days: int
    travel_type: str
    itinerary: str  





# ---- Prompt ----
travel_prompt = PromptTemplate.from_template("""
You are a travel planner. Create a detailed day-by-day itinerary for a {travel_type} trip to {destination} lasting {total_days} days.

Each day must include the following sections, with both the name and a short description for each:

- **Breakfast**
  - "Restaurant": Name of the recommended restaurant or café
  - "Description": A brief description about the place and why it's recommended

- **Morning Activity**
  - "Place": Name of the attraction or activity
  - "Description": A short background or reason to visit

- **Lunch**
  - "Restaurant": Name of the recommended restaurant
  - "Description": Short explanation of cuisine/type and reason to try it

- **Afternoon Activity**
  - "Place": Name of the location or experience
  - "Description": A brief explanation of what to do or see there

- **Dinner**
  - "Restaurant": Name of the dinner spot
  - "Description": Highlight what makes it a good choice for the evening

Return the itinerary as a JSON object structured like this:

{{
  "Day 1": {{
    "Breakfast": {{
      "Restaurant": "...",
      "Description": "..."
    }},
    "Morning Activity": {{
      "Place": "...",
      "Description": "..."
    }},
    "Lunch": {{
      "Restaurant": "...",
      "Description": "..."
    }},
    "Afternoon Activity": {{
      "Place": "...",
      "Description": "..."
    }},
    "Dinner": {{
      "Restaurant": "...",
      "Description": "..."
    }}
  }},
  ...
}}
""")




# ---- Chain ----
travel_chain = travel_prompt | planner_llm

# ---- Planner Node ----
def travel_planner_node(state: TravelState) -> dict:
    result = travel_chain.invoke({
        "destination": state["destination"],
        "total_days": state["total_days"],
        "travel_type": state["travel_type"]
    })
    return {"itinerary": result.content}

# ---- Graph ----
graph_builder = StateGraph(TravelState)
graph_builder.add_node("travel_planner", travel_planner_node)
graph_builder.set_entry_point("travel_planner")
graph = graph_builder.compile()





def run_graph(destination: str, total_days: int, travel_type: str):
    initial_state = {
        "destination": destination,
        "total_days": total_days,
        "travel_type": travel_type
    }
    final_state = graph.invoke(initial_state)
    print("\n📅 Final Itinerary:\n", final_state["itinerary"])




if __name__ == "__main__":
    print("🌍 Welcome to Travel Planner!\n")
    while True:
        destination = input("Destination (or 'q' to quit): ")
        if destination.lower() in ["q", "quit", "exit"]:
            break
        total_days = int(input("Total Days: "))
        travel_type = input("Travel Type (e.g. adventure, leisure, cultural): ")
        run_graph(destination, total_days, travel_type)
