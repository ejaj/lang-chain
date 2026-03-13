"""
TOPIC: PIIMiddleware

WHAT IT DOES:
    Detects Personally Identifiable Information (PII) in conversation
    messages and applies one of several strategies: redact, mask, hash,
    or block. Protects sensitive data before it reaches the model or logs.

WHY THIS MATTERS:
    Healthcare, finance, and customer service apps handle sensitive data.
    Logging or sending PII to external APIs can violate GDPR, HIPAA, etc.
    PIIMiddleware sanitizes data automatically without changing your code.

STRATEGIES:
    "redact"  → replaces PII with its type label  e.g. "john@x.com" → "[EMAIL]"
    "mask"    → partially hides it               e.g. "4111..." → "****...1234"
    "hash"    → replaces with a consistent hash  e.g. SHA256 of the value
    "block"   → raises an error if PII is found  (stops the agent entirely)

BUILT-IN DETECTORS (just pass the name as a string):
    "email", "credit_card", "ssn", "phone", "ip_address", etc.

CUSTOM DETECTORS (three methods):
    1. Regex string:        r"sk-[a-zA-Z0-9]{32}"
    2. Compiled regex:      re.compile(r"\+?\d{10}")
    3. Callable function:   def detect_ssn(content: str) -> list[dict]: ...

CONFIGURATION:
    PIIMiddleware(
        "email",                    # built-in type name
        strategy="redact",          # what to do when found
        apply_to_input=True,        # sanitize user input (before model sees it)
    )

WHEN TO USE:
    Healthcare chatbots (HIPAA)
    Financial applications (PCI-DSS)
    GDPR-compliant customer service
    Any app where PII must not appear in logs
"""

import re
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware


# ---------------------------------------------------------------------------
# 1. Tool
# ---------------------------------------------------------------------------
def process_request(info: str) -> str:
    """Process a customer request."""
    return f"Request processed: {info}"


# ---------------------------------------------------------------------------
# 2. Agent with multiple PII rules
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-4.1",
    tools=[process_request],
    middleware=[
        # Built-in: redact email addresses in user input
        PIIMiddleware("email",
                      strategy="redact",
                      apply_to_input=True),

        # Built-in: mask credit card numbers (show last 4 digits only)
        PIIMiddleware("credit_card",
                      strategy="mask",
                      apply_to_input=True),

        # Custom: regex string — detect API keys like "sk-abc123..."
        PIIMiddleware("api_key",
                      detector=r"sk-[a-zA-Z0-9]{20,}",
                      strategy="block",
                      apply_to_input=True),

        # Custom: compiled regex — detect phone numbers
        PIIMiddleware("phone_number",
                      detector=re.compile(r"\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{4}"),
                      strategy="mask",
                      apply_to_input=True),
    ],
)


# ---------------------------------------------------------------------------
# 3. Test: input contains email + credit card
# ---------------------------------------------------------------------------
print("=" * 60)
print("PIIMiddleware — redact email, mask credit card")
print("=" * 60)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "Hi, I'm john.doe@gmail.com and my card is 4111 1111 1111 1234. "
            "Please process my refund."
        ),
    }]
})
print(f"Response: {result['messages'][-1].content}")
# The model NEVER sees the real email or card number.
# It sees: "[EMAIL] and my card is ****...1234"
print()


# ---------------------------------------------------------------------------
# 4. Custom detector function — SSN with validation logic
# ---------------------------------------------------------------------------
def detect_ssn(content: str) -> list[dict]:
    """
    Custom SSN detector with domain-logic validation.
    Returns list of {text, start, end} dicts.
    """
    matches = []
    pattern = r"\d{3}-\d{2}-\d{4}"
    for match in re.finditer(pattern, content):
        ssn = match.group(0)
        first_three = int(ssn[:3])
        # Validate: 000, 666, and 900-999 are not valid SSN prefixes
        if first_three not in [0, 666] and not (900 <= first_three <= 999):
            matches.append({
                "text":  ssn,
                "start": match.start(),
                "end":   match.end(),
            })
    return matches


agent_ssn = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        PIIMiddleware("ssn",
                      detector=detect_ssn,     # custom callable
                      strategy="hash",          # replace with consistent hash
                      apply_to_input=True),
    ],
)

print("Custom SSN detector agent created.")
print("SSNs in user input will be hashed before reaching the model.")
print()

# ---------------------------------------------------------------------------
# 5. Summary of all strategies
# ---------------------------------------------------------------------------
print("=" * 60)
print("Strategy examples for 'john@example.com'")
print("=" * 60)
print('  redact  → "[EMAIL]"')
print('  mask    → "j***@example.com"')
print('  hash    → "a1b2c3d4..."  (SHA256, consistent per value)')
print('  block   → raises PIIDetectedError immediately')