# TYPE: Streaming Messages (AIMessageChunk)
# DESCRIPTION: When streaming, you get many small AIMessageChunk pieces
# instead of one big AIMessage at the end.
# Each chunk has a tiny piece of the reply.
# Add them together (+) to build the full message.

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4.1", model_provider="openai")

# --- Basic: print each word as it arrives ---
print("Answer: ", end="", flush=True)
for chunk in model.stream("Tell me a joke"):
    print(chunk.text, end="", flush=True)  # prints word by word
print()  # newline at end

# --- Collect all chunks into one full message ---
chunks = []
full_message = None

for chunk in model.stream("What is Python?"):
    chunks.append(chunk)
    full_message = chunk if full_message is None else full_message + chunk

# full_message is now a complete AIMessage — same as invoke() output
print(full_message.content)         # full text
print(full_message.usage_metadata)  # token counts

# --- With reasoning model (shows thinking steps) ---
reasoning_model = init_chat_model("claude-sonnet-4-6", model_provider="anthropic")

for chunk in reasoning_model.stream("Why is the sky blue?"):
    for block in chunk.content_blocks:
        if block["type"] == "reasoning":
            print(f"{block['reasoning']}")   # thinking step
        elif block["type"] == "text":
            print(f"{block['text']}")         # final answer

# KEY DIFFERENCE:
# invoke() → 1 AIMessage  (wait for full reply)
# stream()  → many AIMessageChunks (see reply build up live)
# Both give same final content — stream just shows it progressively