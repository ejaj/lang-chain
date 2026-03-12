# TYPE: Custom Tool Name and Description
# DESCRIPTION: By default, tool name = function name and description = docstring.
# You can override both using @tool("name", description="...").
# Use custom names when the function name is unclear or too generic.
# Use custom descriptions when you need more precise guidance for the AI.

from langchain.tools import tool

# --- Default: name and description come from function ---
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

print(search.name)         # → "search"
print(search.description)  # → "Search for information."

# --- Custom name: override the function name ---
@tool("web_search")   # ← custom name
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

print(search.name)  # → "web_search"  (not "search")

# --- Custom description: override the docstring ---
@tool(
    "calculator",
    description="Performs arithmetic calculations. Use this for ANY math problems including addition, subtraction, multiplication, division."
)
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))

print(calc.name)         # → "calculator"
print(calc.description)  # → "Performs arithmetic calculations. Use this for ANY math..."

# WHY CUSTOM DESCRIPTION MATTERS:
# Bad:  "Does math."           → AI might not know when to use it
# Good: "Use for ANY math..."  → AI knows to always call this for numbers