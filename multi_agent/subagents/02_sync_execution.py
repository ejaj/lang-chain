"""
Synchronous subagent execution
======================================================

WHAT IS IT?
-----------
The main agent WAITS for the subagent to finish before doing
anything else. This is the default behaviour — no extra setup needed.

WHEN TO USE:
- The main agent needs the result to decide what to do next
- Tasks have order dependencies: fetch data → analyse → respond
- The task completes in seconds (not minutes)
- A subagent failure should block the whole response

WHY USE SYNC:
- Simplest to implement — just call and wait
- Easy to reason about: step 1 finishes, then step 2 starts

WATCH OUT:
- User sees nothing until ALL subagents complete
- If a subagent takes 30s, the conversation freezes for 30s
- Not suitable for long-running tasks (use 03_async for those)

FLOW:
    User asks → Main agent calls subagent → WAITS → Gets result → Replies
"""

from langchain.tools import tool
from langchain.agents import create_agent

# ------------------------------------------------------------------
# Example: weather assistant
# User asks → main agent calls weather subagent → waits → replies
# ------------------------------------------------------------------
@tool
def weather_api():
    print("Weather API")

weather_subagent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[weather_api],
    system_prompt="You fetch and summarise current weather data for cities."
)

@tool("get_weather", description="Get the current weather for a given city")
def call_weather_agent(city: str):
    # Main agent BLOCKS here until this function returns.
    # The subagent fetches the weather, summarises it, returns.
    result = weather_subagent.invoke({
        "messages": [{"role": "user", "content": f"Current weather in {city}"}]
    })
    return result["messages"][-1].content


main_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[call_weather_agent],
    system_prompt="You answer weather questions."
)

response = main_agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
})

print(response["messages"][-1].content)

# WHAT HAPPENS STEP BY STEP:
# 1. User:        "What's the weather in Tokyo?"
# 2. Main agent:  calls get_weather("Tokyo")
# 3.              >>> BLOCKS — waiting for subagent <<<
# 4. Subagent:    fetches weather API → "72°F, sunny"
# 5. Main agent:  receives result, writes reply
# 6. User sees:   "It's 72°F and sunny in Tokyo right now."
#
# Total time = however long the subagent takes
# User sees nothing during that time