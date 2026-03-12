# TYPE: Tool Error Handling (ToolNode)
# DESCRIPTION: When a tool crashes, you can control what happens next.
# By default errors bubble up and break the agent.
# ToolNode lets you catch errors and send a friendly message back to the AI
# so it can recover gracefully instead of crashing.

from langchain.tools import tool
from langgraph.prebuilt import ToolNode

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b   # will crash if b=0

@tool
def search(query: str) -> str:
    """Search the web."""
    if not query:
        raise ValueError("Query cannot be empty")
    return f"Results for: {query}"

# ============================================================
# Option 1: Default — crashes re-raised, invocation errors caught
# ============================================================
tool_node = ToolNode([divide, search])

# ============================================================
# Option 2: Catch ALL errors → send error message back to AI
# AI can then try a different approach
# ============================================================
tool_node = ToolNode([divide, search], handle_tool_errors=True)
# AI receives: "Error: division by zero" and tries again

# ============================================================
# Option 3: Custom error message
# ============================================================
tool_node = ToolNode(
    [divide, search],
    handle_tool_errors="Something went wrong. Please check your input and try again."
)

# ============================================================
# Option 4: Custom error handler function
# ============================================================
def my_error_handler(e: Exception) -> str:
    if isinstance(e, ZeroDivisionError):
        return "Cannot divide by zero. Please use a non-zero denominator."
    if isinstance(e, ValueError):
        return f"Invalid input: {str(e)}"
    return f"Unexpected error: {str(e)}"

tool_node = ToolNode([divide, search], handle_tool_errors=my_error_handler)

# ============================================================
# Option 5: Only catch specific error types
# ============================================================
tool_node = ToolNode(
    [divide, search],
    handle_tool_errors=(ValueError, ZeroDivisionError)  # only catch these
)

# RULE:
# No error handling → agent crashes on any tool failure
# handle_tool_errors=True → AI gets the error and can try again  (recommended)
# Custom handler → full control over what message AI sees