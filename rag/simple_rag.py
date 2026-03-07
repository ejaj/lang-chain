from typing import List, Tuple
import numpy as np
import re 
from langchain.text_splitter import RecursiveCharacterTextSplitter

def cosine_similarity(a:List[float], b:List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    # Convert to numpy arrays for easier computation
    vec_a = np.array(a)
    vec_b = np.array(b)

    # Cosine similarity = dot product / (magnitude_a * magnitude_b)
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)

    # Avoid division by zero
    if magnitude_a == 0 or magnitude_b ==0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)

def sentence_chunking(text, max_sentences=5):
    """
    Split text into chunks by sentences
    
    Args:
        text: Input text
        max_sentences: Maximum sentences per chunk
    """
    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= max_sentences:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    # Add remaining sentences
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def semantic_chunking(text, chunk_size=500, overlap=50):
    """
    Split text intelligently by semantic boundaries
    
    Tries to split at:
    1. Paragraph breaks (\n\n)
    2. Sentence breaks (.)
    3. Words
    4. Characters (last resort)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks

def simple_rag(
    query: str,
    documents: List[str],
    top_k: int=3
) -> Tuple[List[str], str]:
    """
    Retrieve top-k relevant documents and generate prompt.
    
    Returns:
        Tuple of (retrieved_docs, final_prompt)
    """
    # 1. Embed query and documents (simulate with random but deterministic vectors)
    # Use hash of text for reproducibility
    np.random.seed(hash(query) % (2**32))
    query_embedding = np.random.randn(384).tolist() 

    doc_embeddings = []
    for doc in documents:
        np.random.seed(hash(doc) % (2**32))
        doc_embeddings.append(np.random.randn(384).tolist())
    
    # 2. Calculate similarity scores for each document
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim_score = cosine_similarity(query_embedding, doc_emb)
        similarities.append((i, sim_score))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [idx for idx, _ in similarities[:top_k]]
    retrieved_docs = [documents[idx] for idx in top_k_indices]

    context = "\n\n".join(
        [f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)]
    )
    final_prompt = f"""Given the following context, please answer the question.

Context: {context}
Question: {query}
Answer:"""
    return (retrieved_docs, final_prompt)

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start+chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks



if __name__ == "__main__":
    # Sample documents
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning models require large amounts of training data.",
        "RAG systems combine retrieval with generation for better answers.",
        "Natural language processing helps computers understand human language.",
        "Vector databases store embeddings for efficient similarity search.",
    ]
    
    query = "What is RAG and how does it work?"
    
    retrieved, prompt = simple_rag(query, documents, top_k=2)
    
    print("Retrieved Documents:")
    for i, doc in enumerate(retrieved, 1):
        print(f"{i}. {doc}")
    
    print("\n" + "="*60)
    print("\nFinal Prompt:")
    print(prompt)
    # Example
    text = """
    Our company was founded in 2020. We specialize in AI solutions.
    Our refund policy is customer-friendly. Full refunds within 30 days.
    Contact support@company.com for assistance. We respond within 24 hours.
    """

    chunks = sentence_chunking(text, max_sentences=2)

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}\n")
