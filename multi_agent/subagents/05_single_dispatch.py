"""
Single dispatch tool pattern
======================================================

WHAT IS IT?
-----------
One generic task(agent_name, description) tool that can invoke
ANY registered subagent by name. Agents are stored in a registry
dict. Adding a new agent = adding one line to the dict.

WHEN TO USE:
- Many agents (10+)
- Different teams build and own different agents independently
- You need to add new agents without touching the main agent's code
- You prefer convention over per-agent customisation

WHY USE THIS vs tool-per-agent (04):
- Adding an agent = 1 line in the registry, nothing else changes
- Each team can deploy their agent independently
- The main agent code never needs to change

TRADE-OFFS:
- Less control over individual agent input/output shapes
- All agents get the same generic input (a description string)
- Harder to do per-agent context engineering

KEY INSIGHT:
Sometimes subagents have the SAME capabilities as the main agent.
That's fine — the real reason to use a subagent is CONTEXT ISOLATION.
Complex multi-step work runs in its own clean window,
and only a short summary comes back to the main agent.
"""

from langchain.tools import tool
from langchain.agents import create_agent

@tool
def web_search():
    print("Web search")
@tool
def read_url():
    print("Read URL")
@tool
def legal_database():
    print("Load database")

# ------------------------------------------------------------------
# Agent registry — each team registers their agent here
# ------------------------------------------------------------------
# Team A builds research. Team B builds writer. Team C builds legal.
# They each add one line here. The main agent never changes.

AGENTS = {
    "research": create_agent(
        model="anthropic:claude-sonnet-4-20250514",
        tools=[web_search, read_url],
        system_prompt="You are a research specialist. Find facts, "
                      "summarise clearly, cite sources."
    ),
    "writer": create_agent(
        model="anthropic:claude-sonnet-4-20250514",
        tools=[],
        system_prompt="You are a writing specialist. Create clear, "
                      "engaging content tailored to the audience."
    ),
    "reviewer": create_agent(
        model="anthropic:claude-sonnet-4-20250514",
        tools=[],
        system_prompt="You review documents for accuracy, clarity, "
                      "and grammar. Return specific, actionable feedback."
    ),
    # New team adds their agent here — nothing else changes:
    "legal": create_agent(
        model="anthropic:claude-sonnet-4-20250514",
        tools=[legal_database],
        system_prompt="You handle legal questions and contract review."
    ),
}

# ------------------------------------------------------------------
# ONE generic tool for ALL agents
# ------------------------------------------------------------------

@tool
def task(agent_name: str, description: str) -> str:
    """Run a task using a named specialist agent.

    Available agents:
    - research: Fact-finding and web research
    - writer:   Content creation and editing
    - reviewer: Document and code review
    - legal:    Legal questions and contracts
    """
    if agent_name not in AGENTS:
        return f"Unknown agent '{agent_name}'. Available: {list(AGENTS.keys())}"

    agent = AGENTS[agent_name]
    result = agent.invoke({
        "messages": [{"role": "user", "content": description}]
    })
    return result["messages"][-1].content


# ------------------------------------------------------------------
# Main agent — one tool, learns about agents from the docstring
# ------------------------------------------------------------------

main_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[task],
    system_prompt="You coordinate specialist agents. Use the task tool "
                  "to delegate work to the right specialist."
)

# ------------------------------------------------------------------
# Example run — chains 3 agents together
# ------------------------------------------------------------------

response = main_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Write a 500-word blog post about quantum computing."
    }]
})

print(response["messages"][-1].content)

# WHAT HAPPENS INTERNALLY:
# 1. Main agent decides: needs research → writing → review
#
# 2. task("research", "find key facts about quantum computing,
#         its current state, and practical applications")
#    → returns research summary
#
# 3. task("writer", "write 500-word blog post about quantum computing.
#         Use these facts: [research results]")
#    → returns draft post
#
# 4. task("reviewer", "review this blog post for accuracy and clarity:
#         [draft]")
#    → returns feedback
#
# 5. Main agent combines everything → final polished post