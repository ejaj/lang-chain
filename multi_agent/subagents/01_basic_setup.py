"""
01_basic_setup.py — Basic subagent setup
=========================================

WHAT IS IT?
-----------
The core mechanic of the subagents pattern:
1. Create a specialist agent
2. Wrap it as a tool
3. Give that tool to the main agent

The main agent sees the subagent as just another tool — like a
calculator or API call. It doesn't know or care that "research"
is itself an AI agent internally.

WHEN TO USE:
- This is the starting point for ALL subagents implementations.
  Every other file in this guide builds on this foundation.

WHY IT WORKS:
- The subagent's internal reasoning (tool calls, thinking steps)
  never reaches the main agent.
- Only result["messages"][-1].content (the final message) comes back.
- This keeps the main agent's context clean and focused.
"""

from langchain.tools import tool
from langchain.agents import create_agent


@tool 
def web_search():
    print("Web search tool")
@tool
def read_url():
    print("Read URL")
# ------------------------------------------------------------------
# STEP 1: Create the specialist subagent
# ------------------------------------------------------------------
# This agent only knows about research.
# It has its own tools (web_search, read_url) and its own prompt.

research_subagent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[web_search, read_url],       # specialist tools
    system_prompt="You are a research specialist. Find accurate, "
                  "up-to-date information and summarise your findings clearly."
)

# ------------------------------------------------------------------
# STEP 2: Wrap the subagent as a tool
# ------------------------------------------------------------------
# The @tool decorator + description is what the main agent reads
# to decide WHEN to call this subagent.
# → Clear, specific descriptions = correct routing
# → Vague descriptions = the main agent calls the wrong specialist

@tool("research", description="Research a topic on the web and return findings")
def call_research_agent(query: str):
    result = research_subagent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    # Only the final message comes back — not intermediate steps
    return result["messages"][-1].content


# ------------------------------------------------------------------
# STEP 3: Give the tool to the main agent
# ------------------------------------------------------------------

main_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[call_research_agent],        # subagent is just a tool
    system_prompt="You are a helpful assistant. Use the research "
                  "tool when you need to find information."
)

# ------------------------------------------------------------------
# STEP 4: Run it
# ------------------------------------------------------------------

response = main_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "What are the latest AI safety research papers from 2025?"
    }]
})

print(response["messages"][-1].content)

# WHAT HAPPENS INTERNALLY:
# 1. Main agent receives the user's question
# 2. Main agent decides to call the "research" tool
# 3. research_subagent runs, does web searches, compiles findings
# 4. Only the final summary comes back to the main agent
# 5. Main agent writes the reply to the user