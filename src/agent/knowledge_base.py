"""
Knowledge Base Module

File Responsibility:
    Provides vector search functionality for energy-related queries
    using semantic similarity or keyword-based fallback.

Inputs:
    - User queries about energy topics

Outputs:
    - Relevant knowledge documents with relevance scores
    - Context strings for LLM prompts

Assumptions:
    - sentence-transformers is optional (keyword fallback available)
    - Documents are loaded from knowledge_documents module

Failure Modes:
    - Missing sentence-transformers uses keyword fallback
    - Empty queries return no results
"""

import numpy as np
from typing import List, Dict

from knowledge_documents import (
    ENERGY_KNOWLEDGE_BASE,
    get_all_documents,
    get_documents_by_category,
    get_all_categories
)


class KnowledgeBase:
    """
    RAG Knowledge Base for energy-related queries.
    
    Purpose: Provide semantic search over energy knowledge documents.
    """
    
    def __init__(self):
        self.documents = ENERGY_KNOWLEDGE_BASE
        self.embeddings = None
        self.model = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self._compute_embeddings()
        except ImportError:
            print("sentence-transformers not installed. Using keyword search.")
            self.model = None
    
    def _compute_embeddings(self):
        """Compute embeddings for all documents."""
        if self.model is None:
            return
        texts = [f"{doc['title']} {doc['content']}" for doc in self.documents]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search knowledge base for relevant documents.
        
        Purpose: Find most relevant documents for a query.
        
        Inputs:
            query: User's question
            top_k: Number of results to return
            
        Outputs:
            List of documents with relevance scores
        """
        if self.model is not None and self.embeddings is not None:
            return self._semantic_search(query, top_k)
        return self._keyword_search(query, top_k)
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Search using semantic similarity."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['score'] = float(similarities[idx])
            results.append(doc)
        return results
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback keyword-based search."""
        query_words = set(query.lower().split())
        
        scores = []
        for doc in self.documents:
            text = f"{doc['title']} {doc['content']}".lower()
            score = sum(1 for word in query_words if word in text)
            scores.append(score)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx].copy()
                doc['score'] = scores[idx]
                results.append(doc)
        return results
    
    def get_context_for_query(self, query: str, max_tokens: int = 1500) -> str:
        """Get formatted context string for LLM prompt."""
        results = self.search(query, top_k=3)
        
        context_parts = []
        total_length = 0
        
        for doc in results:
            content = f"**{doc['title']}**\n{doc['content']}\n"
            if total_length + len(content) < max_tokens * 4:
                context_parts.append(content)
                total_length += len(content)
        
        return "\n---\n".join(context_parts)
    
    def get_all_categories(self) -> List[str]:
        """Get all unique categories."""
        return get_all_categories()
    
    def get_documents_by_category(self, category: str) -> List[Dict]:
        """Get documents in a specific category."""
        return get_documents_by_category(category)


# Global knowledge base instance
_knowledge_base = None


def get_knowledge_base() -> KnowledgeBase:
    """Get or create the global knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base


def query_knowledge_base(question: str) -> Dict:
    """
    Query knowledge base for relevant information.
    
    Purpose: Simple interface for dashboard and agent.
    
    Inputs:
        question: User's question about energy
        
    Outputs:
        Dictionary with context and source documents
    """
    kb = get_knowledge_base()
    results = kb.search(question, top_k=3)
    context = kb.get_context_for_query(question)
    
    return {
        'context': context,
        'sources': [
            {'title': r['title'], 'category': r['category'], 'score': r.get('score', 0)} 
            for r in results
        ]
    }


if __name__ == "__main__":
    kb = get_knowledge_base()
    
    print("=" * 50)
    print("KNOWLEDGE BASE TEST")
    print("=" * 50)
    
    test_queries = [
        "How can I save money on AC bills?",
        "What is the PM Surya Ghar scheme?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = query_knowledge_base(query)
        print(f"Sources: {[s['title'] for s in result['sources']]}")
    
    print("\n✓ Knowledge base working correctly!")
