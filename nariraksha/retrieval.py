import json
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

class SafetyKnowledgeRetriever:
    """
    Retrieval module for NariRaksha.
    Retrieves relevant laws, BNS sections, and POSH guidelines based on a user's scenario.
    Uses TF-IDF and Cosine Similarity as a baseline.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.knowledge_base = []
        self.vectors = None

    def load_knowledge_base(self, processed_data_paths: List[str]):
        logger.info(f"Loading knowledge base from {processed_data_paths}")
        for path in processed_data_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        # Assuming each item has 'content' and 'source'
                        if 'content' in item:
                            self.knowledge_base.append(item)
            except FileNotFoundError:
                logger.warning(f"File not found: {path}. Skipping.")
        
        if self.knowledge_base:
            corpus = [item['content'] for item in self.knowledge_base]
            self.vectors = self.vectorizer.fit_transform(corpus)
            logger.info(f"Indexed {len(self.knowledge_base)} knowledge items.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.knowledge_base or self.vectors is None:
            logger.warning("Knowledge base is empty. Returning empty results.")
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors).flatten()
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1: # Threshold
                results.append({
                    "score": float(similarities[idx]),
                    "source": self.knowledge_base[idx].get('source', 'Unknown'),
                    "content": self.knowledge_base[idx]['content']
                })
                
        return results

if __name__ == "__main__":
    retriever = SafetyKnowledgeRetriever()
    # retriever.load_knowledge_base(["data/processed/bns_processed.json"])
    # res = retriever.retrieve("I am facing continuous unwanted messages online.")
    # print(res)
