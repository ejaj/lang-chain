"""
Tool per agent pattern
===============================================
 
WHAT IS IT?
-----------
Each subagent gets its own named tool. The main agent has a list of
N tools, one per specialist. Each tool can have a different input
shape, description, and output format.
 
WHEN TO USE:
- Small, fixed set of agents (2–6 specialists)
- One team owns and maintains all agents
- You need fine-grained control over what each agent receives
- Each agent has a meaningfully different input schema
 
WHY USE THIS:
- The main agent sees distinct, clearly-named tools
- You can customise each tool's description independently
- Easy to add per-agent logic (custom inputs, output formatting)
 
TRADE-OFFS vs single dispatch (05_single_dispatch.py):
- More setup code
- Adding a new agent = adding a new tool = touching main agent code
- But: more control over each individual agent
 
EXAMPLE SCENARIO:
A personal assistant that manages calendar, email, and CRM.
Each domain is its own specialist with different tools and prompts.
"""
 
from langchain.tools import tool
from langchain.agents import create_agent
 
# ------------------------------------------------------------------
# Three specialist subagents — each owns a different domain
# ------------------------------------------------------------------
 

@tool
def google_calendar_read(query: str) -> str:
    return "You have a meeting at 3pm"

@tool
def google_calendar_write(query: str) -> str:
    return "Wrtie it"

@tool
def gmail_read(query: str) -> str:
    return "Read Gamil"

@tool
def gmail_send(query: str) -> str:
    return "Send mail"

@tool
def salesforce_read(query: str) -> str:
    return "Read Salesforce Data"

@tool
def salesforce_write(query: str) -> str:
    return "Wrtie information"


calendar_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[google_calendar_read, google_calendar_write],
    system_prompt="You manage calendar events. You can create, update, "
                  "delete, and query events. Always confirm the timezone."
)
 
email_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[gmail_read, gmail_send],
    system_prompt="You read and send emails. Be concise and professional. "
                  "Always confirm before sending."
)
 
crm_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[salesforce_read, salesforce_write],
    system_prompt="You look up and update CRM records in Salesforce. "
                  "Return structured data when querying contacts or deals."
)
# ------------------------------------------------------------------
# Each gets its own tool — note different input schemas
# ------------------------------------------------------------------
 
@tool("calendar", description=(
    "Create, update, or query calendar events. "
    "Use for scheduling meetings, checking availability, or finding events."
))
def call_calendar(request: str):
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content
 
 
@tool("email", description=(
    "Read emails from inbox or send new emails. "
    "Use for checking messages, drafting replies, or sending communications."
))
def call_email(request: str):
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content
 
 
@tool("crm", description=(
    "Look up or update customer and deal records in the CRM. "
    "Use for finding contact details, deal status, or updating records."
))
def call_crm(customer_name: str, action: str):
    # Note: different input shape — two params instead of one
    result = crm_agent.invoke({
        "messages": [{"role": "user", "content": f"{action} for {customer_name}"}]
    })
    return result["messages"][-1].content
 
 
# ------------------------------------------------------------------
# Main agent sees three distinct, well-described tools
# ------------------------------------------------------------------
 
main_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[call_calendar, call_email, call_crm],
    system_prompt="You are a personal assistant with access to calendar, "
                  "email, and CRM. Coordinate across them to complete tasks."
)
 
# ------------------------------------------------------------------
# Example run — uses multiple specialists in sequence
# ------------------------------------------------------------------
 
response = main_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Schedule a meeting with Acme Corp tomorrow at 2pm "
                   "and send them a confirmation email."
    }]
})
 
print(response["messages"][-1].content)
 
# WHAT HAPPENS INTERNALLY:
# 1. Main agent reads request — needs CRM + calendar + email
# 2. Calls crm("Acme Corp", "get contact email") → "contact@acme.com"
# 3. Calls calendar("create meeting tomorrow 2pm with Acme Corp")
#    → "Meeting created: 2025-03-23 14:00"
# 4. Calls email("send meeting confirmation to contact@acme.com
#    for tomorrow 2pm meeting") → "Email sent"
# 5. Main agent replies: "Done — meeting created and confirmation sent."