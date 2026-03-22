"""
Skills with reference awareness
============================================================

WHAT IS IT?
-----------
Skills don't have to contain ALL the knowledge inline. Instead,
a skill prompt can REFERENCE external files and tell the agent:
  - what files exist
  - what each file contains
  - when to read each one

The agent reads files into memory only when needed for the current task.
This is progressive disclosure applied to file assets, not just prompts.

ANALOGY:
A new employee gets an onboarding doc that says:
  "For expense reports → see /wiki/finance/expense-policy.md"
  "For code style     → see /wiki/engineering/style-guide.md"
The employee doesn't read every linked doc upfront.
They only look up the relevant one when they need it.

WHY REFERENCE AWARENESS:
- Some knowledge assets are large (schemas, style guides, templates)
- Loading all of them upfront wastes tokens
- The skill prompt acts as an index: "these files exist, here's what's in them"
- The agent reads a file only when the task requires it

WHEN TO USE:
- Skills have associated large reference documents
- You want the agent to know ABOUT files without reading ALL of them
- File content changes frequently (read fresh each time vs. baked into prompt)
- Different files are relevant for different sub-tasks

SCENARIO: A coding assistant skill that references:
  - A database schema file (read when writing queries)
  - A style guide file (read when reviewing code)
  - An API spec file (read when building integrations)

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 04_reference_awareness.py
"""

import os
import tempfile
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# ------------------------------------------------------------------
# Simulate reference files on disk
# (In production: real files in a repo, S3, or knowledge base)
# ------------------------------------------------------------------

MOCK_FILES: dict[str, str] = {
    "/docs/schema.sql": """
-- Database schema (last updated 2025-03-01)

CREATE TABLE customers (
    id          UUID PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    email       VARCHAR(255) UNIQUE NOT NULL,
    region      VARCHAR(50),           -- 'NA', 'EU', 'APAC'
    tier        VARCHAR(20),           -- 'free', 'pro', 'enterprise'
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE orders (
    id          UUID PRIMARY KEY,
    customer_id UUID REFERENCES customers(id),
    amount      NUMERIC(10,2) NOT NULL,
    status      VARCHAR(20),           -- 'pending', 'paid', 'refunded', 'failed'
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE order_items (
    id          UUID PRIMARY KEY,
    order_id    UUID REFERENCES orders(id),
    product_id  UUID REFERENCES products(id),
    quantity    INT NOT NULL,
    unit_price  NUMERIC(10,2) NOT NULL
);

CREATE TABLE products (
    id          UUID PRIMARY KEY,
    name        VARCHAR(255),
    category    VARCHAR(100),
    price       NUMERIC(10,2),
    stock_qty   INT DEFAULT 0
);

-- Indexes
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
""",

    "/docs/style_guide.md": """
# Python Style Guide

## Naming
- Variables/functions: snake_case
- Classes: PascalCase
- Constants: UPPER_SNAKE_CASE
- Private methods: _leading_underscore

## Functions
- Max 20 lines. If longer, extract to helper functions.
- One responsibility per function.
- Always type-hint parameters and return values.
- Docstring required for all public functions.

## Error handling
- Never use bare `except:` — always specify exception type.
- Use custom exceptions for domain errors (class PaymentError(Exception): pass)
- Log exceptions with context, not just the message.

## Imports
- Standard library first, then third-party, then local.
- No wildcard imports (from module import *).
- Prefer absolute imports over relative.

## Tests
- One test file per module: test_<module_name>.py
- Test function naming: test_<what>_<condition>_<expected_result>
- Use pytest fixtures, not setUp/tearDown.
""",

    "/docs/api_spec.yaml": """
openapi: 3.0.0
info:
  title: Orders API
  version: 1.0.0

paths:
  /v1/orders:
    get:
      summary: List orders
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [pending, paid, refunded, failed]
        - name: customer_id
          in: query
          schema:
            type: string
            format: uuid
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: per_page
          in: query
          schema:
            type: integer
            default: 20
            maximum: 100
      responses:
        '200':
          description: List of orders
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Order'
                  total: { type: integer }
                  page:  { type: integer }

    post:
      summary: Create order
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateOrderRequest'
      responses:
        '201': { description: Order created }
        '422': { description: Validation error }

components:
  schemas:
    Order:
      type: object
      properties:
        id:          { type: string, format: uuid }
        customer_id: { type: string, format: uuid }
        amount:      { type: number }
        status:      { type: string }
        created_at:  { type: string, format: date-time }

    CreateOrderRequest:
      type: object
      required: [customer_id, items]
      properties:
        customer_id: { type: string, format: uuid }
        items:
          type: array
          items:
            type: object
            properties:
              product_id: { type: string, format: uuid }
              quantity:   { type: integer, minimum: 1 }
""",
}

# ------------------------------------------------------------------
# Skill: references files but doesn't load them upfront
# ------------------------------------------------------------------

CODING_ASSISTANT_SKILL = """
CODING ASSISTANT SKILL LOADED.

You are a backend engineering assistant for our e-commerce platform.

REFERENCE FILES AVAILABLE (read them with read_file when relevant):

/docs/schema.sql
  → Full database schema with all tables, columns, types, and indexes.
  → Read this when: writing SQL queries, designing new tables,
    or answering questions about data structure.

/docs/style_guide.md
  → Python coding standards and conventions.
  → Read this when: reviewing code, writing new Python functions,
    or answering questions about code style.

/docs/api_spec.yaml
  → OpenAPI specification for the Orders API.
  → Read this when: building API integrations, writing client code,
    or answering questions about endpoints and request/response shapes.

Do NOT read all files upfront. Read only the file(s) relevant to the
current task.
"""

# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------

@tool
def load_skill(skill_name: str) -> str:
    """Load a specialized skill.

    Available skills:
    - coding_assistant : Backend engineering for our e-commerce platform.
      Provides access to schema, style guide, and API spec files.
    """
    if skill_name == "coding_assistant":
        return CODING_ASSISTANT_SKILL.strip()
    return f"Skill '{skill_name}' not found. Available: ['coding_assistant']"


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a reference file.

    Use this to load detailed reference material when needed.
    Only read files that are relevant to the current task.
    """
    content = MOCK_FILES.get(file_path)
    if content is None:
        available = list(MOCK_FILES.keys())
        return f"File '{file_path}' not found. Available files: {available}"
    return content.strip()

# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-20250514")

agent = create_react_agent(
    model=model,
    tools=[load_skill, read_file],
    prompt=(
        "You are a helpful engineering assistant. "
        "Load the coding_assistant skill before attempting engineering tasks. "
        "After loading the skill, use read_file to load specific reference "
        "documents only when the task requires them. "
        "Do not read all files — only the ones relevant to the current question."
    )
)

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def chat(history: list, user_message: str) -> tuple[str, list]:
    history.append(HumanMessage(content=user_message))
    result = agent.invoke({"messages": history})
    reply = result["messages"][-1].content
    history = result["messages"]
    return reply, history

# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    history = []

    print("=== Reference Awareness Demo ===\n")

    # Turn 1: SQL question → agent loads skill, reads schema.sql
    print("USER: Write a query to get total revenue per customer tier for last month.")
    reply, history = chat(history, "Write a query to get total revenue per customer tier for last month.")
    print(f"AGENT: {reply}\n")
    print("-" * 60)

    # Turn 2: code review → agent reads style_guide.md
    print("USER: Can you review this function for style issues?")
    code = """
def GetOrdersByCustomer(customerId, db):
    try:
        result = db.query('SELECT * FROM orders WHERE customer_id = %s', customerId)
        return result
    except:
        return None
"""
    reply, history = chat(history, f"Can you review this function for style issues?\n{code}")
    print(f"AGENT: {reply}\n")
    print("-" * 60)

    # Turn 3: API question → agent reads api_spec.yaml
    print("USER: How do I call the orders API to fetch all pending orders for a specific customer?")
    reply, history = chat(history, "How do I call the orders API to fetch all pending orders for a specific customer?")
    print(f"AGENT: {reply}\n")

    # Show which files were read
    print("\n=== Files read during session ===")
    for m in history:
        if hasattr(m, "name") and m.name == "read_file":
            print(f"  → {m.content[:60]}...")