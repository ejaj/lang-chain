# TYPE: Batch
# DESCRIPTION: batch() sends multiple questions at the same time (in parallel).
# Much faster than calling invoke() one by one.
# Use when you have many independent questions to answer.

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4.1", model_provider="openai")

questions = [
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?",
]

# --- Basic batch: all 3 sent at once, wait for all to finish ---
responses = model.batch(questions)
for response in responses:
    print(response.content)
    print("---")

# --- Batch with max concurrency: limit to 2 parallel calls at a time ---
# Useful when provider has rate limits
responses = model.batch(
    questions,
    config={"max_concurrency": 2}  # only 2 requests at a time
)

# --- batch_as_completed: print each answer as soon as it's ready ---
# (answers may arrive out of order — faster overall)
for response in model.batch_as_completed(questions):
    print(response.content)
    print("---")

# KEY DIFFERENCE vs invoke():
# invoke()              → 1 question  → 1 answer
# batch()               → many questions → all answers (wait for slowest)
# batch_as_completed()  → many questions → answers arrive as ready (fastest)