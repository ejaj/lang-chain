# TYPE: Model Profile
# DESCRIPTION: Every model has a profile — a dictionary of its capabilities.
# Use .profile to check what a model can do before using it.
# Useful for writing code that works with ANY model, not just one specific model.

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4.1", model_provider="openai")

# --- Check what this model can do ---
print(model.profile)
# {
#   "max_input_tokens": 128000,   ← how much text it can read at once
#   "image_inputs": True,         ← can it read images?
#   "tool_calling": True,         ← can it call tools?
#   "reasoning_output": False,    ← does it show its thinking steps?
#   ...
# }

# --- Practical use: only use image feature if model supports it ---
if model.profile.get("image_inputs"):
    print("This model can read images")
else:
    print("This model cannot read images")

# --- Practical use: check context window size ---
max_tokens = model.profile.get("max_input_tokens", 0)
if max_tokens > 100_000:
    print(f"Large context model — can handle big documents ({max_tokens} tokens)")
else:
    print(f"Small context model — keep inputs short ({max_tokens} tokens)")

# WHY USEFUL: your code can automatically adapt to any model
# instead of hardcoding "if model == gpt-4.1" checks everywhere