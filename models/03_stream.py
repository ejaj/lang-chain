# TYPE: Stream
# DESCRIPTION: stream() shows the reply word by word as it's being generated.
# Use this for better user experience — user sees output immediately
# instead of waiting for the full response.

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4.1", model_provider="openai")

# --- Basic streaming: print each word as it arrives ---
print("Answer: ", end="")
for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)
# Output builds up live:
# → "Parrots|have|colorful|feathers|because|..."

print()  # new line after streaming ends

# --- Collect full message while streaming ---
full = None
for chunk in model.stream("What color is the sky?"):
    full = chunk if full is None else full + chunk
    print(full.text)  # grows with each chunk

# After loop, full is a complete AIMessage — same as invoke() output
print(full.content_blocks)
# → [{"type": "text", "text": "The sky is typically blue..."}]

# KEY DIFFERENCE:
# invoke() → waits, returns 1 AIMessage at the end
# stream() → returns many small AIMessageChunks as they arrive