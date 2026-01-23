"""
Smart AI Energy Consumption Predictor
Knowledge Base Module for RAG System

Contains energy-related knowledge documents and vector search functionality.
"""

import os
import numpy as np
from typing import List, Dict, Tuple
import json

# Energy Knowledge Base - Documents for RAG
ENERGY_KNOWLEDGE_BASE = [
    # Energy Saving Tips
    {
        "id": "tip_1",
        "category": "energy_saving",
        "title": "AC Temperature Optimization",
        "content": """Setting your air conditioner to 24-26°C instead of lower temperatures can save up to 6% energy for each degree raised. Using ceiling fans along with AC helps distribute cool air more efficiently, allowing you to set a higher temperature while maintaining comfort. Regular cleaning of AC filters every month improves efficiency by 5-15%."""
    },
    {
        "id": "tip_2", 
        "category": "energy_saving",
        "title": "LED Lighting Benefits",
        "content": """LED bulbs consume 80% less electricity than incandescent bulbs and 50% less than CFLs. They last 25 times longer than incandescent bulbs. Replacing all home lighting with LEDs can save ₹2,000-5,000 annually on electricity bills. Motion sensors in low-traffic areas further reduce lighting costs."""
    },
    {
        "id": "tip_3",
        "category": "energy_saving", 
        "title": "Off-Peak Usage Strategy",
        "content": """Electricity rates are often lower during off-peak hours (typically 11 PM to 6 AM). Running heavy appliances like washing machines, dishwashers, and water heaters during these hours can reduce bills by 15-25%. Many utilities offer time-of-use tariffs that reward off-peak consumption."""
    },
    {
        "id": "tip_4",
        "category": "energy_saving",
        "title": "Standby Power Reduction",
        "content": """Phantom loads or standby power from electronics can account for 5-10% of home energy use. Using power strips with switches, unplugging chargers when not in use, and choosing appliances with low standby consumption can save ₹1,000-2,000 annually. Smart power strips automatically cut power to devices in standby mode."""
    },
    {
        "id": "tip_5",
        "category": "energy_saving",
        "title": "Refrigerator Efficiency",
        "content": """Refrigerators run 24/7 and account for 15-20% of home electricity use. Keep the thermostat at 3-4°C for the fridge and -18°C for the freezer. Don't place hot food directly inside. Ensure door seals are tight. Keep the refrigerator away from heat sources and leave space behind for ventilation. A 5-star rated refrigerator uses 30-40% less energy than lower-rated models."""
    },
    
    # Solar Energy
    {
        "id": "solar_1",
        "category": "solar",
        "title": "Rooftop Solar Benefits",
        "content": """A 3kW rooftop solar system can generate 12-15 units of electricity per day, saving ₹3,000-4,500 monthly on electricity bills. Solar panels have a lifespan of 25+ years with minimal maintenance. Net metering allows you to sell excess power back to the grid. Solar panels also provide roof insulation, reducing cooling costs."""
    },
    {
        "id": "solar_2",
        "category": "solar",
        "title": "PM Surya Ghar Yojana",
        "content": """Under the PM Surya Ghar Muft Bijli Yojana, households can get subsidies up to ₹78,000 for installing rooftop solar systems. The scheme covers 1-3 kW systems with Central Financial Assistance (CFA) of ₹30,000/kW for systems up to 2 kW and ₹18,000/kW for 2-3 kW capacity. Apply through the National Portal for Rooftop Solar. Many states offer additional incentives."""
    },
    {
        "id": "solar_3",
        "category": "solar",
        "title": "Solar System Sizing",
        "content": """To size a solar system, calculate your average daily consumption in kWh and divide by 4-5 (average peak sun hours in India). A typical household using 10 units/day needs a 2-3 kW system. Consider roof space (10 sq.m per kW), orientation (south-facing is optimal), and shading. Battery storage adds 30-50% to system cost but provides backup during outages."""
    },
    
    # Carbon Footprint
    {
        "id": "carbon_1",
        "category": "carbon",
        "title": "India Electricity Emission Factor",
        "content": """India's electricity grid has an emission factor of approximately 0.82 kg CO2 per kWh (as per Central Electricity Authority). This means every unit of electricity consumed generates 820 grams of CO2. The emission factor varies by source: coal (0.91 kg/kWh), natural gas (0.45 kg/kWh), solar (0.05 kg/kWh lifecycle), and wind (0.01 kg/kWh). Reducing consumption directly reduces your carbon footprint."""
    },
    {
        "id": "carbon_2",
        "category": "carbon",
        "title": "Carbon Offset with Trees",
        "content": """A mature tree absorbs approximately 22 kg of CO2 per year. An average Indian household consuming 200 units/month generates about 164 kg CO2 monthly or 2 tonnes annually. This would require planting 90 trees to offset. Reducing consumption is more effective than offsetting - cutting 50 units/month saves the equivalent of 22 trees annually."""
    },
    
    # Appliances
    {
        "id": "appliance_1",
        "category": "appliances",
        "title": "Energy Star Ratings",
        "content": """BEE (Bureau of Energy Efficiency) star ratings help identify efficient appliances. A 5-star AC uses 20-30% less energy than a 3-star model. For a 1.5-ton AC running 8 hours daily, this saves ₹3,000-5,000 annually. Always check the annual energy consumption (in kWh) on the label - lower is better. 5-star appliances cost more upfront but save money over their lifetime."""
    },
    {
        "id": "appliance_2",
        "category": "appliances",
        "title": "Inverter Technology",
        "content": """Inverter ACs and refrigerators adjust compressor speed based on load, unlike conventional fixed-speed units that cycle on/off. This provides 30-50% energy savings, faster cooling, quieter operation, and longer lifespan. The higher upfront cost is typically recovered within 2-3 years through energy savings. BLDC (Brushless DC) fans use 60% less electricity than regular fans."""
    },
    
    # Tariffs
    {
        "id": "tariff_1",
        "category": "tariffs",
        "title": "Electricity Tariff Slabs in India",
        "content": """Most Indian states have telescopic/slab-based tariffs where the rate increases with consumption. Typical domestic slabs: 0-100 units (₹3-4/unit), 101-200 units (₹4.5-5.5/unit), 201-300 units (₹6-7/unit), above 300 units (₹7.5-8.5/unit). Some states have Time-of-Day tariffs with lower rates during off-peak hours. Fixed charges and taxes add 20-30% to the energy charges."""
    },
    
    # Smart Home
    {
        "id": "smart_1",
        "category": "smart_home",
        "title": "Smart Home Energy Management",
        "content": """Smart plugs and energy monitors help track consumption of individual appliances. Smart thermostats learn your preferences and optimize heating/cooling schedules. Smart lighting with motion sensors and schedules reduces wastage. Home energy management systems (HEMS) can reduce overall consumption by 10-20% through automation and insights. Voice assistants can control devices remotely."""
    }
]


class KnowledgeBase:
    """RAG Knowledge Base for energy-related queries."""
    
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
            print("sentence-transformers not installed. Using keyword search fallback.")
            self.model = None
    
    def _compute_embeddings(self):
        """Compute embeddings for all documents."""
        if self.model is None:
            return
        
        texts = [f"{doc['title']} {doc['content']}" for doc in self.documents]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search the knowledge base for relevant documents.
        
        Args:
            query: User's question
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with scores
        """
        if self.model is not None and self.embeddings is not None:
            return self._semantic_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Search using semantic similarity."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['score'] = float(similarities[idx])
            results.append(doc)
        
        return results
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback keyword-based search."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scores = []
        for doc in self.documents:
            text = f"{doc['title']} {doc['content']}".lower()
            # Count matching words
            score = sum(1 for word in query_words if word in text)
            scores.append(score)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx].copy()
                doc['score'] = scores[idx]
                results.append(doc)
        
        return results
    
    def get_context_for_query(self, query: str, max_tokens: int = 1500) -> str:
        """
        Get relevant context from knowledge base for a query.
        
        Args:
            query: User's question
            max_tokens: Maximum context length (approximate)
            
        Returns:
            Concatenated relevant context
        """
        results = self.search(query, top_k=3)
        
        context_parts = []
        total_length = 0
        
        for doc in results:
            content = f"**{doc['title']}**\n{doc['content']}\n"
            if total_length + len(content) < max_tokens * 4:  # Approximate token count
                context_parts.append(content)
                total_length += len(content)
        
        return "\n---\n".join(context_parts)
    
    def get_all_categories(self) -> List[str]:
        """Get all unique categories in the knowledge base."""
        return list(set(doc['category'] for doc in self.documents))
    
    def get_documents_by_category(self, category: str) -> List[Dict]:
        """Get all documents in a specific category."""
        return [doc for doc in self.documents if doc['category'] == category]


# Initialize global knowledge base
_knowledge_base = None

def get_knowledge_base() -> KnowledgeBase:
    """Get or create the global knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base


def query_knowledge_base(question: str) -> Dict:
    """
    Query the knowledge base and return relevant information.
    
    Args:
        question: User's question about energy
        
    Returns:
        Dictionary with context and sources
    """
    kb = get_knowledge_base()
    results = kb.search(question, top_k=3)
    context = kb.get_context_for_query(question)
    
    return {
        'context': context,
        'sources': [{'title': r['title'], 'category': r['category'], 'score': r.get('score', 0)} for r in results]
    }


if __name__ == "__main__":
    # Test the knowledge base
    kb = get_knowledge_base()
    
    print("=" * 60)
    print("KNOWLEDGE BASE TEST")
    print("=" * 60)
    
    test_queries = [
        "How can I save money on AC bills?",
        "What is the PM Surya Ghar scheme?",
        "How much CO2 does electricity produce?",
        "What is a 5-star rating appliance?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        result = query_knowledge_base(query)
        print(f"Sources: {[s['title'] for s in result['sources']]}")
        print(f"Context preview: {result['context'][:200]}...")
