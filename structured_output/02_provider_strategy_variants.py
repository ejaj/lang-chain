"""
TOPIC: Provider Strategy — Dataclass, TypedDict, JSON Schema

WHAT IT DOES:
    Shows all four schema types that ProviderStrategy supports beyond Pydantic.
    Useful when you don't want a Pydantic dependency or prefer plain dicts.

RETURN TYPES BY SCHEMA KIND:
    Pydantic BaseModel → validated Pydantic INSTANCE
    Dataclass          → plain dict  (not a dataclass instance)
    TypedDict          → plain dict
    JSON Schema dict   → plain dict

WHEN TO USE EACH:
    Dataclass  → you already use dataclasses in your codebase
    TypedDict  → you want type hints but don't need validation
    JSON Schema → you're defining the schema dynamically / from an API spec
"""

# ──────────────────────────────────────────────────────────────────────────
# A) DATACLASS
# ──────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from langchain.agents import create_agent


@dataclass
class BookSummary:
    title:  str
    author: str
    genre:  str
    pages:  int


agent_dc = create_agent(model="gpt-5", response_format=BookSummary)

result = agent_dc.invoke({
    "messages": [{"role": "user", "content":
        "Summarize: 'The Hobbit' by Tolkien, fantasy, ~310 pages"}]
})

print("── Dataclass result ──")
print(type(result["structured_response"]))   # dict (not BookSummary instance)
print(result["structured_response"])
# {'title': 'The Hobbit', 'author': 'J.R.R. Tolkien', 'genre': 'fantasy', 'pages': 310}
print()


# ──────────────────────────────────────────────────────────────────────────
# B) TYPEDDICT
# ──────────────────────────────────────────────────────────────────────────
from typing import TypedDict


class WeatherReport(TypedDict):
    city:        str
    temperature: float
    condition:   str
    humidity:    int


agent_td = create_agent(model="gpt-5", response_format=WeatherReport)

result = agent_td.invoke({
    "messages": [{"role": "user", "content":
        "Weather in Paris: 18°C, partly cloudy, 65% humidity"}]
})

print("── TypedDict result ──")
print(type(result["structured_response"]))   # dict
print(result["structured_response"])
# {'city': 'Paris', 'temperature': 18.0, 'condition': 'partly cloudy', 'humidity': 65}
print()


# ──────────────────────────────────────────────────────────────────────────
# C) JSON SCHEMA (raw dict)
# Useful when you generate schemas dynamically or receive them from an API.
# ──────────────────────────────────────────────────────────────────────────
json_schema = {
    "title": "InvoiceData",
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string",  "description": "The invoice ID"},
        "total_amount":   {"type": "number",  "description": "Total in USD"},
        "due_date":       {"type": "string",  "description": "Due date YYYY-MM-DD"},
        "is_paid":        {"type": "boolean", "description": "Has the invoice been paid?"},
    },
    "required": ["invoice_number", "total_amount", "due_date", "is_paid"],
}

agent_js = create_agent(model="gpt-5", response_format=json_schema)

result = agent_js.invoke({
    "messages": [{"role": "user", "content":
        "Invoice INV-2024-042, $1,250.00, due 2024-03-31, not yet paid"}]
})

print("── JSON Schema result ──")
print(type(result["structured_response"]))   # dict
print(result["structured_response"])
# {'invoice_number': 'INV-2024-042', 'total_amount': 1250.0,
#  'due_date': '2024-03-31', 'is_paid': False}