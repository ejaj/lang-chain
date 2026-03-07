from production_rag import ProductionRAG
class ProductionRAGWithMetrics(ProductionRAG):
    def __init__(self):
        super().__init__()

        self.metrics = {
            "queries": 0,
            "avg_retrieval_time": 0,
            "avg_generation_time": 0
        }
    def query(self, question, top_k=3):
        import time
        # Track retrieval time
        start = time.time()
        chunks = self.retrieve(question, top_k)
        retrieval_time = time.time() - start
        
        # Track generation time
        start = time.time()
        result = self.generate_answer(question, chunks)
        generation_time = time.time() - start
        
       # Update metrics
        self.metrics["queries"] += 1
        self.metrics["avg_retrieval_time"] = (
            (self.metrics["avg_retrieval_time"] * (self.metrics["queries"] - 1) +
             retrieval_time) / self.metrics["queries"]
        )
        self.metrics["avg_generation_time"] = (
            (self.metrics["avg_generation_time"] * (self.metrics["queries"] - 1) +
             generation_time) / self.metrics["queries"]
        )
        
        # Add timing to result
        result["metrics"] = {
            "retrieval_time": f"{retrieval_time:.3f}s",
            "generation_time": f"{generation_time:.3f}s",
            "total_time": f"{retrieval_time + generation_time:.3f}s"
        }
        return result
    def get_metrics(self):
        return self.metrics

