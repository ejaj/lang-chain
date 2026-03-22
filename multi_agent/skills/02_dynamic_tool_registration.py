"""
Skills with dynamic tool registration
========================================================================

WHAT IS IT?
-----------
An extension of the basic skills pattern. When an agent loads a skill,
it doesn't just get a specialized prompt — it also gets NEW TOOLS
registered into its active toolset.

Example: loading the "database_admin" skill gives the agent both:
  - A specialized system prompt (SQL expertise, schema knowledge)
  - New tools: run_query, backup_db, list_tables

Before the skill loads → those tools don't exist for the agent.
After the skill loads → the agent can actually execute queries.

HOW IT WORKS:
State tracks which skills are loaded and which tools are active.
The load_skill tool returns a Command that updates both:
  1. The skill prompt (added to messages as context)
  2. The active_tools list in state

A middleware node reads active_tools from state and binds them
to the model before each LLM call.

WHEN TO USE:
- Skills need to unlock real capabilities, not just knowledge
- You want to prevent tool access until prerequisites are met
  (e.g. must load "db_admin" skill before getting run_query tool)
- Tools are heavyweight and shouldn't be in context unnecessarily

SCENARIO: A data platform assistant.
  No skill loaded → can only answer general questions
  "database_admin" skill loaded → gets run_query, list_tables, backup_db
  "visualization" skill loaded → gets render_chart, export_csv

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 02_dynamic_tool_registration.py
"""

from builtins import isinstance
from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command

# ------------------------------------------------------------------
# State — tracks loaded skills and currently active tools
# ------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    loaded_skills: list[str]     # which skills have been loaded
    active_tools: list[str]      # which tool names are currently available

# ------------------------------------------------------------------
# Domain tools — only available after the right skill is loaded
# ------------------------------------------------------------------

@tool
def run_query(sql: str) -> str:
    """Execute a SQL query against the database."""
    # In production: execute against real DB
    return f"Query executed. Results: [simulated data for: {sql[:60]}...]"

@tool
def list_tables() -> str:
    """List all tables in the database."""
    return "Tables: customers, orders, order_items, products, invoices"

@tool
def backup_db(label: str) -> str:
    """Create a database backup with a label."""
    return f"Backup '{label}' created successfully at 2025-03-22T10:00:00Z"

@tool
def render_chart(chart_type: str, data_description: str) -> str:
    """Render a chart from query results."""
    return f"Chart rendered: {chart_type} — {data_description}"

@tool
def export_csv(query: str, filename: str) -> str:
    """Export query results to a CSV file."""
    return f"Exported results to {filename}.csv (1,234 rows)"

# Registry: tool name → tool function
ALL_DOMAIN_TOOLS: dict[str, object] = {
    "run_query": run_query,
    "list_tables": list_tables,
    "backup_db": backup_db,
    "render_chart": render_chart,
    "export_csv": export_csv,
}

# ------------------------------------------------------------------
# Skill definitions — prompt + which tools they unlock
# ------------------------------------------------------------------

SKILL_REGISTRY = {
    "database_admin": {
        "prompt": """
DATABASE ADMIN SKILL LOADED.

You now have access to database tools: run_query, list_tables, backup_db.

Schema:
- customers(id, name, email, region, created_at)
- orders(id, customer_id, amount, status, created_at)
- products(id, name, category, price)

Guidelines:
- Always use list_tables first if unsure about schema
- Test queries with LIMIT 10 before running on full table
- Always create a backup before any destructive operation
""",
        "unlocks_tools": ["run_query", "list_tables", "backup_db"],
    },
    "visualization": {
        "prompt": """
VISUALIZATION SKILL LOADED.

You now have access to: render_chart, export_csv.

Chart types available: bar, line, pie, scatter, heatmap.
Always run the underlying query first, then visualize the results.
For exports, use descriptive filenames like 'top_customers_q1_2025'.
""",
        "unlocks_tools": ["render_chart", "export_csv"],
    },
}

# ------------------------------------------------------------------
# load_skill tool — updates state with prompt + new tools
# ------------------------------------------------------------------

@tool
def load_skill(
    skill_name: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Load a specialized skill and unlock its tools.

    Available skills:
    - database_admin  : Unlocks run_query, list_tables, backup_db
    - visualization   : Unlocks render_chart, export_csv
    """
    skill = SKILL_REGISTRY.get(skill_name)
    if not skill:
        available = list(SKILL_REGISTRY.keys())
        return f"Skill '{skill_name}' not found. Available: {available}"

    return Command(
        update={
            # Inject the skill prompt as context
            "messages": [
                ToolMessage(
                    content=skill["prompt"].strip(),
                    tool_call_id=tool_call_id,
                )
            ],
            # Register the new tools in state
            "active_tools": skill["unlocks_tools"],
            "loaded_skills": [skill_name],
        }
    )

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-20250514")

# ------------------------------------------------------------------
# Agent node — middleware reads active_tools from state each turn
# ------------------------------------------------------------------

def agent_node(state: AgentState) -> Command:
    """
    Middleware: reads active_tools from state, binds them to the model.
    This is how tools appear/disappear based on which skills are loaded.
    """
    # Always available: load_skill itself
    current_tools = [load_skill]

    # Add any tools unlocked by loaded skills
    for tool_name in state.get("active_tools", []):
        if tool_name in ALL_DOMAIN_TOOLS:
            current_tools.append(ALL_DOMAIN_TOOLS[tool_name])

    system = SystemMessage(content=(
        "You are a data platform assistant. "
        f"Loaded skills: {state.get('loaded_skills', ['none'])}. "
        f"Active tools: {[t.name for t in current_tools]}. "
        "Load a skill before attempting domain-specific tasks."
    ))

    bound_model = model.bind_tools(current_tools)
    response = bound_model.invoke([system] + state["messages"])

    # If model made tool calls, process them; otherwise end turn
    if response.tool_calls:
        return Command(update={"messages": [response]}, goto="tool_node")
    return Command(update={"messages": [response]}, goto=END)


# ------------------------------------------------------------------
# Tool execution node
# ------------------------------------------------------------------

from langchain_core.messages import ToolMessage as LCToolMessage

def tool_node(state: AgentState) -> Command:
    """Execute tool calls from the last AI message."""
    last_message = state["messages"][-1]

    all_tools = {
        "load_skill": load_skill,
        **ALL_DOMAIN_TOOLS,
    }

    tool_messages = []
    for tc in last_message.tool_calls:
        tool_fn = all_tools.get(tc["name"])
        if not tool_fn:
            tool_messages.append(LCToolMessage(
                content=f"Tool '{tc['name']}' not found.",
                tool_call_id=tc["id"]
            ))
            continue

        result = tool_fn.invoke({**tc["args"], "tool_call_id": tc["id"]})

        # If tool returned a Command (e.g. load_skill), execute it
        if isinstance(result, Command):
            return result

        tool_messages.append(LCToolMessage(
            content=str(result),
            tool_call_id=tc["id"]
        ))

    return Command(
        update={"messages": tool_messages},
        goto="agent_node"
    )

# ------------------------------------------------------------------
# Graph
# ------------------------------------------------------------------

builder = StateGraph(AgentState)
builder.add_node("agent_node", agent_node)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "agent_node")

graph = builder.compile()

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def chat(state: AgentState, user_message: str) -> tuple[str, AgentState]:
    state["messages"].append(HumanMessage(content=user_message))
    result = graph.invoke(state)
    reply = next(
        (m.content for m in reversed(result["messages"])
         if isinstance(m, AIMessage) and m.content),
        "(no reply)"
    )
    return reply, result

# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    state: AgentState = {
        "messages": [],
        "loaded_skills": [],
        "active_tools": [],
    }

    print("=== Dynamic Tool Registration Demo ===\n")

    # Turn 1: general question — no domain tools yet
    print("USER: What tables are in the database?")
    reply, state = chat(state, "What tables are in the database?")
    print(f"AGENT: {reply}")
    print(f"  [active tools: {state['active_tools']}]\n")

    # Turn 2: load database_admin skill → unlocks run_query, list_tables, backup_db
    print("USER: Load the database_admin skill so I can query the database.")
    reply, state = chat(state, "Load the database_admin skill so I can query the database.")
    print(f"AGENT: {reply}")
    print(f"  [active tools: {state['active_tools']}]\n")

    # Turn 3: now run_query is available
    print("USER: Show me the top 3 customers by order total.")
    reply, state = chat(state, "Show me the top 3 customers by order total.")
    print(f"AGENT: {reply}")
    print(f"  [active tools: {state['active_tools']}]\n")

    # Turn 4: load visualization skill → adds render_chart, export_csv
    print("USER: Also load the visualization skill.")
    reply, state = chat(state, "Also load the visualization skill.")
    print(f"AGENT: {reply}")
    print(f"  [active tools: {state['active_tools']}]\n")

    # Turn 5: now both tool sets available
    print("USER: Export those top customers to a CSV file.")
    reply, state = chat(state, "Export those top customers to a CSV file.")
    print(f"AGENT: {reply}")
    print(f"  [active tools: {state['active_tools']}]\n")