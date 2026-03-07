# TYPE: Token Usage in Messages
# DESCRIPTION: Every AIMessage contains usage_metadata showing how many tokens were used.
# Tokens = the unit models charge money for. ~1 token ≈ 1 word.
# input_tokens  = tokens in your message (what you sent)
# output_tokens = tokens in AI reply (what it generated)
# total_tokens  = input + output = what you're billed for

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

model = init_chat_model("gpt-4.1", model_provider="openai")

# --- Check usage after a call ---
response = model.invoke("Hello!")
print(response.usage_metadata)
# {
#   "input_tokens": 8,     ← your message used 8 tokens
#   "output_tokens": 10,   ← AI reply used 10 tokens
#   "total_tokens": 18,    ← total = 18 tokens billed
#   "input_token_details":  {"cache_read": 0},
#   "output_token_details": {"reasoning": 0}
# }

# --- Longer message = more input tokens ---
response_short = model.invoke("Hi")
response_long  = model.invoke("Hi " * 100)  # 100 words

print(response_short.usage_metadata["input_tokens"])  # small number
print(response_long.usage_metadata["input_tokens"])   # much larger

# --- Why this matters for cost ---
# Short conversation (10 tokens total)  → very cheap
# Long conversation (10,000 tokens)     → more expensive
# System prompt counts too → keep it concise

# --- Reasoning tokens (Claude / o-series models) ---
# Some models use "reasoning tokens" internally to think
# These are billed but NOT shown in the reply text
# output_token_details: {"reasoning": 256} → 256 tokens used for thinking
response = model.invoke([
    SystemMessage("Think step by step."),
    HumanMessage("What is 17 * 23?")
])
print(response.usage_metadata.get("output_token_details", {}))
# → {"reasoning": 50}  ← model spent 50 tokens thinking before answering