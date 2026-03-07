# TYPE: Token Usage Tracking
# DESCRIPTION: Tokens are the "units" models charge you for.
# Roughly: 1 token ≈ 1 word (actually ~0.75 words).
# Track usage to monitor costs across multiple model calls.

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

model_gpt    = init_chat_model("gpt-4.1-mini",             model_provider="openai")
model_claude = init_chat_model("claude-haiku-4-5-20251001", model_provider="anthropic")

# --- Track usage with a callback handler ---
callback = UsageMetadataCallbackHandler()

# Run both models — callback tracks both automatically
result_1 = model_gpt.invoke(   "Hello!", config={"callbacks": [callback]})
result_2 = model_claude.invoke("Hello!", config={"callbacks": [callback]})

# --- See how many tokens each model used ---
print(callback.usage_metadata)
# {
#   "gpt-4.1-mini": {
#       "input_tokens": 8,    ← tokens YOU sent
#       "output_tokens": 10,  ← tokens AI replied with
#       "total_tokens": 18    ← input + output = what you're billed for
#   },
#   "claude-haiku-4-5-20251001": {
#       "input_tokens": 8,
#       "output_tokens": 21,
#       "total_tokens": 29
#   }
# }

# --- Also available on each response directly ---
print(result_1.usage_metadata)
# → {"input_tokens": 8, "output_tokens": 10, "total_tokens": 18}

# SIMPLE RULE:
# input_tokens  = length of your message
# output_tokens = length of AI's reply
# total_tokens  = what you pay for → keep this low to save money