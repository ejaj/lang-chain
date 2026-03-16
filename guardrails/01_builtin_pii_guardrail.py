"""
TOPIC: Built-in PII Guardrail (PIIMiddleware)

WHAT IS A GUARDRAIL:
    A safety check that runs at a specific point in the agent loop to
    validate, filter, or block content. Guardrails prevent unsafe outputs,
    privacy violations, and compliance failures.

TWO TYPES OF GUARDRAILS:
    Deterministic → regex, keyword lists, rule checks (fast, cheap, predictable)
    Model-based   → LLM evaluates content semantically (catches subtle issues)

    PIIMiddleware is DETERMINISTIC — uses regex + Luhn validation.

WHAT PIIMiddleware DOES:
    Detects Personally Identifiable Information in messages and applies
    one of four strategies: redact, mask, hash, or block.

WHERE IT APPLIES:
    apply_to_input=True          → scan user messages BEFORE model sees them
    apply_to_output=True         → scan AI messages AFTER model responds
    apply_to_tool_results=True   → scan tool return values

FOUR STRATEGIES:
    "redact"  → replace with [REDACTED_EMAIL], [REDACTED_CREDIT_CARD], etc.
    "mask"    → show partial: ****-****-****-1234
    "hash"    → deterministic SHA256 hash (consistent, anonymized)
    "block"   → raise an exception immediately (hard stop)

BUILT-IN DETECTORS:
    "email", "credit_card", "ip", "mac_address", "url"
    + custom: pass a regex string, compiled regex, or a callable function

WHEN TO USE:
    Healthcare (HIPAA) — never log or send PII to the model
    Finance (PCI-DSS) — mask card numbers in all logs
    GDPR compliance — redact emails before AI processes them
    API key protection — block requests containing secrets
"""

import re
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def customer_service_tool(query: str) -> str:
    """Handle a customer service request."""
    return f"Your request has been processed: {query}"


def email_tool(to: str, subject: str, body: str) -> str:
    """Send an email to a customer."""
    return f"Email queued to: {to}"


# ---------------------------------------------------------------------------
# Agent with full PII protection stack
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-4.1",
    tools=[customer_service_tool, email_tool],
    middleware=[
        # ── INPUT SANITIZATION (before model sees user messages) ──────────

        # 1. Redact email addresses in user input
        #    "Please contact john.doe@gmail.com" → "Please contact [REDACTED_EMAIL]"
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),

        # 2. Mask credit card numbers (show only last 4 digits)
        #    "Card: 5105-1051-0510-5100" → "Card: ****-****-****-5100"
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),

        # 3. Hash IP addresses (anonymize but stay consistent)
        #    "My IP is 192.168.1.100" → "My IP is a8f5f167..."
        PIIMiddleware(
            "ip",
            strategy="hash",
            apply_to_input=True,
        ),

        # 4. CUSTOM: Block OpenAI-style API keys — hard stop if found
        #    "sk-abc123..." → raises PIIDetectedError
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{20,}",     # regex string as detector
            strategy="block",
            apply_to_input=True,
        ),

        # ── OUTPUT SANITIZATION (after model responds) ────────────────────

        # 5. Also redact any emails the model might include in its response
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=False,
            apply_to_output=True,    # scan AI messages
        ),
    ],
)


# ---------------------------------------------------------------------------
# Test 1: User input contains email + credit card
# The model NEVER sees the real PII — it only sees the sanitized version
# ---------------------------------------------------------------------------
print("=" * 60)
print("Test 1: User input with email + credit card")
print("=" * 60)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "Hi, I'm john.doe@example.com and my card is "
            "5105-1051-0510-5100. Please process my refund."
        ),
    }]
})
print(f"Response: {result['messages'][-1].content[:300]}")
print()
# Model saw: "Hi, I'm [REDACTED_EMAIL] and my card is ****-****-****-5100. ..."


# ---------------------------------------------------------------------------
# Test 2: Custom detector — block API key in input
# ---------------------------------------------------------------------------
print("=" * 60)
print("Test 2: API key detection (block strategy)")
print("=" * 60)

try:
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "My API key is sk-abcdefghijklmnopqrstuvwxyz1234 — is it valid?",
        }]
    })
except Exception as e:
    print(f"Blocked: {type(e).__name__} — {e}")
print()


# ---------------------------------------------------------------------------
# Custom detector function example — SSN with Luhn-style validation
# ---------------------------------------------------------------------------
def detect_ssn(content: str) -> list[dict]:
    """
    Detect US Social Security Numbers with format and range validation.
    Returns list of {text, start, end} for each match.
    """
    matches = []
    pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    for match in re.finditer(pattern, content):
        ssn = match.group(0)
        first = int(ssn[:3])
        # Invalid SSN prefixes: 000, 666, 900–999
        if first != 0 and first != 666 and not (900 <= first <= 999):
            matches.append({
                "text":  ssn,
                "start": match.start(),
                "end":   match.end(),
            })
    return matches


agent_ssn = create_agent(
    model="gpt-4.1",
    tools=[customer_service_tool],
    middleware=[
        PIIMiddleware(
            "ssn",
            detector=detect_ssn,     # callable detector with validation logic
            strategy="redact",
            apply_to_input=True,
        ),
    ],
)

print("=" * 60)
print("Test 3: Custom SSN detector")
print("=" * 60)

result = agent_ssn.invoke({
    "messages": [{
        "role": "user",
        "content": "My SSN is 078-05-1120. Please verify my identity.",
    }]
})
print(f"Response: {result['messages'][-1].content[:200]}")
# Model saw: "My SSN is [REDACTED_SSN]. Please verify my identity."

# ---------------------------------------------------------------------------
# Strategy summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Strategy comparison for 'john@example.com':")
print("=" * 60)
print('  redact → "[REDACTED_EMAIL]"')
print('  mask   → "j***@example.com"   (partial reveal)')
print('  hash   → "a1b2c3d4e5f6..."   (SHA256, consistent)')
print('  block  → PIIDetectedError raised immediately')