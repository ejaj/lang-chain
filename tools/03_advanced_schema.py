# TYPE: Advanced Schema (Pydantic)
# DESCRIPTION: For tools with many inputs or complex validation,
# define a Pydantic model as the input schema using args_schema.
# This gives the AI precise field descriptions and valid values,
# reducing mistakes when it calls your tool.

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

# --- Define exact input shape with Pydantic ---
class WeatherInput(BaseModel):
    location: str = Field(
        description="City name, e.g. 'Paris' or 'New York'"
    )
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit — only 'celsius' or 'fahrenheit' allowed"
    )
    include_forecast: bool = Field(
        default=False,
        description="Set True to include a 5-day forecast"
    )

# --- Pass schema via args_schema ---
@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast for a location."""
    temp = 22 if units == "celsius" else 72
    result = f"Weather in {location}: {temp}° {units}"
    if include_forecast:
        result += "\nForecast: Sunny all week"
    return result

# --- AI now knows EXACTLY what to send ---
# Without schema → AI might send wrong types or misspell "celsius"
# With schema    → AI sees field descriptions and allowed values

# --- Example usage ---
from langchain.agents import create_agent

agent = create_agent("openai:gpt-4.1", tools=[get_weather])
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Tokyo in fahrenheit with forecast?"}]
})
print(result["messages"][-1].content)
# AI calls: get_weather(location="Tokyo", units="fahrenheit", include_forecast=True)