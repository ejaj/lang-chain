# TYPE: Tool Error Handling
# When a tool crashes, catch the error and send a helpful
# message back to the model so it can try a different approach.

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))  # will crash on bad input

@wrap_tool_call
def handle_errors(request, handler):
    try:
        return handler(request)  # run tool normally

    except ZeroDivisionError:
        return ToolMessage(
            content="Cannot divide by zero. Please use a different number.",
            tool_call_id=request.tool_call["id"],
        )
    except Exception as e:
        return ToolMessage(
            content=f"Tool failed: {str(e)}. Please try a different approach.",
            tool_call_id=request.tool_call["id"],
        )

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[calculate],
    middleware=[handle_errors],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is 10 / 0?"}]
})
# Tool crashes → model gets friendly error → model responds gracefully