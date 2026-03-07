from mcp.server.fastmcp import FastMCP
from typing import Literal
import json

mcp = FastMCP("my-server")


@mcp.tool()
async def say_hello(name: str) -> str:
    """Say hello to a person by name"""
    return f"Hello, {name}!"


@mcp.tool()
async def add_numbers(a: float, b: float) -> str:
    """Add two numbers together"""
    result = a + b
    return str(result)


@mcp.tool()
async def create_ticket(
    customer_id: str,
    issue: str,
    priority: Literal["low", "medium", "high", "urgent"]
) -> str:
    """Create a support ticket for a customer"""
    import time
    tid = f"T{int(time.time())}"
    return f"Ticket {tid} created for {customer_id} | priority: {priority}"

# static resource — same data every time
@mcp.resource("data://customers/all")
async def all_customers() -> str:
    """List of all customers in the system"""
    customers = [
        {"id": "C001", "name": "Arjun Sharma", "plan": "Pro"},
        {"id": "C002", "name": "Priya Patel",  "plan": "Basic"},
        {"id": "C003", "name": "Rahul Verma",  "plan": "Enterprise"},
    ]
    return json.dumps(customers)


# dynamic resource — changes based on customer_id in URL
@mcp.resource("data://customers/{customer_id}")
async def customer_detail(customer_id: str) -> str:
    """Get one specific customer by ID"""
    db = {
        "C001": {"id": "C001", "name": "Arjun",  "plan": "Pro",        "orders": 3},
        "C002": {"id": "C002", "name": "Priya",  "plan": "Basic",      "orders": 1},
        "C003": {"id": "C003", "name": "Rahul",  "plan": "Enterprise", "orders": 7},
    }
    result = db.get(customer_id, {"error": "customer not found"})
    return json.dumps(result)


# another resource — FAQ data
@mcp.resource("data://faq")
async def faq_list() -> str:
    """All FAQ questions and answers"""
    faqs = [
        {"q": "How to reset password?", "a": "Go to Settings → Reset Password"},
        {"q": "How to cancel?",         "a": "Go to Billing → Cancel"},
        {"q": "Payment failed?",        "a": "Update card in Billing settings"},
    ]
    return json.dumps(faqs)


# ════════════════════════════════════════
# PROMPTS — reusable AI instruction templates
# ════════════════════════════════════════

@mcp.prompt()
def support_agent_prompt(customer_name: str, issue: str) -> str:
    """System prompt for customer support sessions"""
    return f"""
    You are a helpful customer support agent.
    You are helping: {customer_name}
    Their issue is: {issue}

    Rules:
    - Always be polite and empathetic
    - Check customer orders before responding
    - Create a ticket for every unresolved issue
    - Send confirmation email after resolving
    """


@mcp.prompt()
def escalation_prompt(ticket_id: str, reason: str) -> str:
    """Prompt for escalating a ticket to senior team"""
    return f"Escalate ticket {ticket_id} to senior support. Reason: {reason}. Mark as urgent."


@mcp.prompt()
def refund_prompt(customer_name: str, amount: float) -> str:
    """Prompt for handling refund requests"""
    return f"Process refund of ${amount} for {customer_name}. Verify order first. Confirm via email."

if __name__ == "__main__":
    mcp.run()