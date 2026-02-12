#!/usr/bin/env python3
import sys, json, random
from config import get_openai_client, MODEL_NAME, Timer

# ── Mock tool implementations ───────────────────────────────────────────────
# Ai Generated Mock data
def search_flights(origin: str, destination: str, date: str, max_price: float = 999) -> list[dict]:
    flights = [
        {"airline": "Air New Zealand", "departure": "06:30", "arrival": "08:45", "price_nzd": 89, "flight": "NZ123"},
        {"airline": "Jetstar", "departure": "10:15", "arrival": "12:30", "price_nzd": 59, "flight": "JQ456"},
        {"airline": "Air New Zealand", "departure": "17:00", "arrival": "19:15", "price_nzd": 110, "flight": "NZ789"},
        {"airline": "Jetstar", "departure": "20:30", "arrival": "22:45", "price_nzd": 65, "flight": "JQ012"},
    ]
    return [f for f in flights if f["price_nzd"] <= max_price]
# Ai Generated Mock data
def get_weather(city: str, date: str) -> dict:
    return {"city": city, "date": date,
            "condition": random.choice(["Sunny", "Partly Cloudy", "Overcast", "Light Rain", "Clear"]),
            "high_c": random.randint(18, 26), "low_c": random.randint(10, 16), "rain_chance_pct": random.randint(0, 40)}
# Ai Generated Mock data
def search_attractions(city: str, category: str = "all") -> list[dict]:
    data = [
        {"name": "Sky Tower", "category": "landmark", "price_nzd": 32, "rating": 4.5, "duration_hrs": 1.5},
        {"name": "Auckland War Memorial Museum", "category": "museum", "price_nzd": 28, "rating": 4.7, "duration_hrs": 2.5},
        {"name": "Rangitoto Island Ferry & Hike", "category": "nature", "price_nzd": 40, "rating": 4.8, "duration_hrs": 5},
        {"name": "Viaduct Harbour Walk", "category": "free", "price_nzd": 0, "rating": 4.3, "duration_hrs": 1},
        {"name": "Auckland Art Gallery", "category": "museum", "price_nzd": 0, "rating": 4.4, "duration_hrs": 2},
        {"name": "Mission Bay Beach", "category": "free", "price_nzd": 0, "rating": 4.2, "duration_hrs": 2},
        {"name": "Kelly Tarlton's SEA LIFE", "category": "attraction", "price_nzd": 44, "rating": 4.1, "duration_hrs": 2},
        {"name": "Ponsonby Food Tour", "category": "food", "price_nzd": 55, "rating": 4.6, "duration_hrs": 3},
    ]
    return [a for a in data if category == "all" or a["category"] == category]
# Ai Generated Mock data
def search_accommodation(city: str, checkin: str, nights: int, max_price: float = 999) -> list[dict]:
    hotels = [
        {"name": "YHA Auckland City", "type": "hostel", "price_per_night_nzd": 45, "rating": 4.0},
        {"name": "CityLife Auckland", "type": "hotel", "price_per_night_nzd": 135, "rating": 4.3},
        {"name": "Haka Lodge Auckland", "type": "hostel", "price_per_night_nzd": 38, "rating": 4.2},
        {"name": "Ibis Budget Auckland", "type": "budget_hotel", "price_per_night_nzd": 95, "rating": 3.8},
    ]
    return [h for h in hotels if h["price_per_night_nzd"] * nights <= max_price]

# ── Tool registry (spec + dispatch) ────────────────────────────────────────

def _tool(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}

TOOLS_SPEC = [
    _tool("search_flights", "Search flights between cities. Returns prices in NZD.",
          {"origin": {"type": "string"}, "destination": {"type": "string"},
           "date": {"type": "string", "description": "YYYY-MM-DD"}, "max_price": {"type": "number"}},
          ["origin", "destination", "date"]),
    _tool("get_weather", "Get weather forecast for a city on a date.",
          {"city": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"}},
          ["city", "date"]),
    _tool("search_attractions", "Search tourist attractions. Categories: landmark, museum, nature, free, attraction, food.",
          {"city": {"type": "string"}, "category": {"type": "string"}}, ["city"]),
    _tool("search_accommodation", "Search accommodation in a city.",
          {"city": {"type": "string"}, "checkin": {"type": "string"}, "nights": {"type": "integer"},
           "max_price": {"type": "number"}}, ["city", "checkin", "nights"]),
]

TOOL_DISPATCH = {
    "search_flights": search_flights, "get_weather": get_weather,
    "search_attractions": search_attractions, "search_accommodation": search_accommodation,
}

# Agent loop

SYSTEM_PROMPT = """You are a travel planning agent. Use the provided tools to gather data, then output a JSON itinerary.
Steps: 1) Search flights (outbound+return) 2) Search accommodation 3) Check weather 4) Search attractions 5) Build itinerary.
Output JSON with: destination, total_budget_nzd, total_estimated_cost_nzd, days (array of {day, date, weather, activities}),
flights, accommodation, budget_breakdown (flights, accommodation, activities, food_estimate).
IMPORTANT: total cost MUST stay within budget."""

def run_agent(user_prompt: str) -> dict:
    client = get_openai_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    scratchpad = []

    for iteration in range(15):
        print(f"\n{'─'*60}\nAgent iteration {iteration + 1}")
        with Timer() as t:
            resp = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=TOOLS_SPEC, tool_choice="auto")
        msg = resp.choices[0].message
        print(f"  [latency: {t.elapsed_ms:.0f} ms]")

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                fn_name, fn_args = tc.function.name, json.loads(tc.function.arguments)
                log = f" {fn_name}({json.dumps(fn_args)})"
                print(log); scratchpad.append(log)

                result = TOOL_DISPATCH.get(fn_name, lambda **kw: {"error": "unknown tool"})(**fn_args)
                result_str = json.dumps(result, indent=2)
                print(f"     ->{len(result) if isinstance(result, list) else 1} result(s)")
                scratchpad.append(f"  Result: {result_str[:200]}...")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
        else:
            content = msg.content or ""
            print(f"\n{'═'*60}\nAgent finished.\n")
            try:
                itinerary = json.loads(content[content.find("{"):content.rfind("}")+1])
            except (json.JSONDecodeError, ValueError):
                itinerary = {"raw_response": content}
            print(json.dumps(itinerary, indent=2))
            print(f"\n{'─'*60}\nScratchpad:")
            for e in scratchpad: print(e)
            return itinerary

    return {"error": "Max iterations reached", "scratchpad": scratchpad}

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Plan a 2-day trip to Auckland for under NZ$500. Departing from Wellington on the 15th of next month."
    run_agent(prompt)
