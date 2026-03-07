# TYPE: ToolMessage
# DESCRIPTION: ToolMessage carries the RESULT of a tool back to the model.
# Flow: HumanMessage → AI requests tool → you run tool → ToolMessage → AI replies
# The tool_call_id must match the ID from the AI's tool call request.
# NOTE: When using create_agent(), this is all handled automatically.
#       ToolMessage is only needed when calling model standalone (no agent).

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool

model = init_chat_model("gpt-4.1", model_provider="openai")

@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny, 22°C in {location}"

model_with_tools = model.bind_tools([get_weather])

# --- Step 1: Ask question → AI requests a tool call ---
response = model_with_tools.invoke([
    HumanMessage("What's the weather in Paris?")
])

# AI doesn't answer yet — it requests a tool call
print(response.tool_calls)
# → [{"name": "get_weather", "args": {"location": "Paris"}, "id": "call_abc123"}]

# --- Step 2: YOU run the tool manually ---
tool_result = get_weather.invoke({"location": "Paris"})
# → "Sunny, 22°C in Paris"

# --- Step 3: Send result back as ToolMessage ---
tool_msg = ToolMessage(
    content=tool_result,
    tool_call_id=response.tool_calls[0]["id"]  # must match AI's call ID!
)

# --- Step 4: AI reads the result and gives final answer ---
final = model_with_tools.invoke([
    HumanMessage("What's the weather in Paris?"),
    response,   # AI's tool call request
    tool_msg,   # tool result
])
print(final.content)
# → "The weather in Paris is sunny and 22°C."

# AGAIN: create_agent() does ALL of steps 2-4 automatically for you!