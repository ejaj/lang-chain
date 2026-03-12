# TYPE: Tool Return Values
# DESCRIPTION: Tools can return 3 different things depending on what you need.
# String  → plain text the AI reads and uses in its reply
# Dict    → structured data with named fields the AI can reason over
# Command → updates the agent's state (memory) AND optionally sends a message

from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

# ============================================================
# RETURN 1: String — simplest, plain text result
# Use when: result is naturally readable text
# ============================================================
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It is currently sunny and 22°C in {city}."
    # AI sees: "It is currently sunny and 22°C in Paris."
    # AI uses this text to form its reply

# ============================================================
# RETURN 2: Dict — structured data with named fields
# Use when: AI needs to reason about specific fields
# ============================================================
@tool
def get_weather_data(city: str) -> dict:
    """Get structured weather data for a city."""
    return {
        "city": city,
        "temperature_c": 22,
        "humidity_pct": 60,
        "conditions": "sunny",
    }
    # AI sees all fields and can say:
    # "Temperature is 22°C with 60% humidity"

# ============================================================
# RETURN 3: Command — updates state AND sends confirmation
# Use when: tool needs to save/change something in the conversation
# ============================================================
@tool
def set_language(language: str, runtime: ToolRuntime) -> Command:
    """Set the user's preferred response language."""
    return Command(
        update={
            "preferred_language": language,    # ← saves to state
            "messages": [
                ToolMessage(
                    content=f"Language set to {language}. I will now reply in {language}.",
                    tool_call_id=runtime.tool_call_id,  # links to this specific tool call
                )
            ],
        }
    )
    # State is now updated for all future steps in this conversation

# SIMPLE RULE:
# Just giving info?        → return str
# Many fields to reason?   → return dict
# Need to save something?  → return Command