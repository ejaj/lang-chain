import os
from typing import List, Dict
import numpy as np 
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from anthropic import Anthropic

class ProductionRAG:

    def __init__(self):
        """Initialize Production RAG System"""
        # Embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Vector database
        print("Initializing vector database...")
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name="production_docs",
            metadata={"hnsw:space": "cosine"}
        )

        # LLM for generation
        self.llm = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,        # Characters per chunk
            chunk_overlap=50,      # Overlap between chunks
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        print("RAG System initialized!\n")

    def chunk_document(self, document: str, metadata: Dict = None) -> List[Dict]:
        """
        Split document into chunks with metadata
        
        Args:
            document: Text to chunk
            metadata: Optional metadata (source, date, etc.)
        
        Returns:
            List of chunks with metadata
        """
        chunks = self.text_splitter.split_text(document)
        chunks_objects = []
        for i, chunk in enumerate(chunks):
           chunk_obj = {
                "text": chunk,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
           chunks_objects.append(chunk_obj)

        return chunks_objects
    
    def index_documents(self, documents: List[Dict]):
        """
        Index documents (OFFLINE)
        
        Args:
            documents: List of {text, metadata} dicts
        """
        print("OFFLINE: Indexing documents...")
        
        all_chunks = []
        all_metadata = []
        all_ids = []
        
        doc_counter = 0
        
        for doc in documents:
            # Chunk each document
            chunks = self.chunk_document(
                doc["text"],
                metadata=doc.get("metadata", {})
            )
            
            # Process each chunk
            for chunk in chunks:
                all_chunks.append(chunk["text"])
                all_metadata.append(chunk["metadata"])
                all_ids.append(f"doc_{doc_counter}")
                doc_counter += 1
        
        # Generate embeddings for all chunks
        print(f"  Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(all_chunks)
        
        # Store in vector database
        print(f"  Storing in vector database...")
        self.collection.add(
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            metadatas=all_metadata,
            ids=all_ids
        )
        
        print(f"Indexed {len(documents)} documents → {len(all_chunks)} chunks\n")
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant chunks (ONLINE)
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
        
        Returns:
            List of relevant chunks with metadata
        """    
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Search vector database
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        # Format results
        retrieved_chunks = []
        for i in range(len(results['documents'][0])):
            retrieved_chunks.append({
                "text":results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distance' in results else None
            })
        return retrieved_chunks
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict:
        """
        Generate answer using LLM (ONLINE)
        
        Args:
            query: User question
            context_chunks: Retrieved chunks
        
        Returns:
            Answer with sources
        """
        # Build context from chunks
        context = "\n\n".join([
            f"[Source {i+1}]: {chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Build prompt
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context provided below.

If the answer is not in the context, say "I don't have that information in my documents."

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate answer with Claude
        response = self.llm.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer_text = response.content[0].text
        
        # Return answer with sources
        return {
            "answer": answer_text,
            "sources": [
                {
                    "text": chunk["text"][:100] + "...",  # Preview
                    "metadata": chunk["metadata"]
                }
                for chunk in context_chunks
            ]
        }
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Complete RAG pipeline (ONLINE)
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
        
        Returns:
            Answer with sources
        """
        print(f" ONLINE: Processing query: '{question}'")
        
        # Step 1: Retrieve relevant chunks
        print(f" Retrieving top {top_k} relevant chunks...")
        chunks = self.retrieve(question, top_k=top_k)
        
        # Step 2: Generate answer
        print(f" Generating answer...")
        result = self.generate_answer(question, chunks)
        
        print(f" Answer generated!\n")
        
        return result


# ============================================
# USAGE EXAMPLE
# ============================================

# Initialize RAG system
rag = ProductionRAG()

# ============================================
# OFFLINE: Index documents
# ============================================

documents = [
    {
        "text": """
REFUND POLICY

Our company offers a customer-friendly refund policy. 
You can request a full refund within 30 days of purchase for any reason.

To initiate a refund:
1. Email support@company.com with your order number
2. Include reason for refund (optional)
3. We'll process your request within 24 hours

Refunds are credited to your original payment method within 5-7 business days.
For orders over $500, refunds may take up to 10 business days.

Exceptions: Digital products are non-refundable after download.
        """,
        "metadata": {
            "source": "refund_policy.pdf",
            "category": "policy",
            "last_updated": "2024-01-15"
        }
    },
    {
        "text": """
SHIPPING INFORMATION

We offer multiple shipping options to suit your needs:

Standard Shipping (FREE on orders $50+):
- Delivery: 5-7 business days
- Tracking provided
- Available nationwide

Express Shipping ($15):
- Delivery: 2-3 business days
- Priority handling
- Signature required

Overnight Shipping ($35):
- Delivery: Next business day
- Must order before 2 PM EST
- Available in major cities only

International shipping is available to select countries.
Contact support@company.com for international rates.
        """,
        "metadata": {
            "source": "shipping_info.pdf",
            "category": "shipping",
            "last_updated": "2024-02-01"
        }
    },
    {
        "text": """
CUSTOMER SUPPORT

Our support team is here to help!

Contact Methods:
- Email: support@company.com (Response within 24 hours)
- Phone: 1-800-555-0123 (Mon-Fri, 9 AM - 5 PM EST)
- Live Chat: Available on our website during business hours

For urgent issues, please call our phone line.
For general questions, email is preferred.

Premium customers get priority support with response within 4 hours.
        """,
        "metadata": {
            "source": "support_info.pdf",
            "category": "support",
            "last_updated": "2024-01-20"
        }
    }
]

# Index all documents
rag.index_documents(documents)

# ============================================
# ONLINE: Query the system
# ============================================

# Question 1
print("="*60)
print("QUESTION 1")
print("="*60)
result1 = rag.query("How do I get a refund?", top_k=3)
print(f"Answer: {result1['answer']}\n")
print("Sources:")
for i, source in enumerate(result1['sources'], 1):
    print(f"  {i}. {source['text']}")
    print(f"     From: {source['metadata']['source']}\n")

# Question 2
print("="*60)
print("QUESTION 2")
print("="*60)
result2 = rag.query("What shipping options are available?", top_k=3)
print(f"Answer: {result2['answer']}\n")
print("Sources:")
for i, source in enumerate(result2['sources'], 1):
    print(f"  {i}. {source['text']}")
    print(f"     From: {source['metadata']['source']}\n")

# Question 3
print("="*60)
print("QUESTION 3")
print("="*60)
result3 = rag.query("How can I contact support?", top_k=3)
print(f"Answer: {result3['answer']}\n")
print("Sources:")
for i, source in enumerate(result3['sources'], 1):
    print(f"  {i}. {source['text']}")
    print(f"     From: {source['metadata']['source']}\n")