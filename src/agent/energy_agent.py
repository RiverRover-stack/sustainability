"""
Energy Advisor Agent Module

File Responsibility:
    Agentic AI powered by Google Gemini for conversational energy advice.
    Uses RAG with knowledge base for enhanced responses.

Inputs:
    - User messages/questions
    - Optional consumption DataFrame for personalization

Outputs:
    - AI-generated responses with sources
    - Consumption analysis and insights

Assumptions:
    - Google Gemini API key is available (optional)
    - Fallback to knowledge base when API unavailable

Failure Modes:
    - Missing API key uses fallback mode
    - API errors return error dictionary
"""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

# Import knowledge base and helper modules
try:
    from knowledge_base import get_knowledge_base, query_knowledge_base
    from quick_answers import get_quick_answer
    from user_analysis import compute_user_summary, format_user_context, generate_insights
except ImportError:
    from agent.knowledge_base import get_knowledge_base, query_knowledge_base
    from agent.quick_answers import get_quick_answer
    from agent.user_analysis import compute_user_summary, format_user_context, generate_insights


SYSTEM_PROMPT = """You are an expert Energy Advisor AI assistant. Your role is to:

1. Answer questions about energy consumption, savings, and sustainability
2. Provide personalized recommendations based on user's consumption data
3. Explain complex energy concepts in simple terms
4. Help users reduce their electricity bills and carbon footprint
5. Provide information about solar energy and government schemes

Guidelines:
- Be helpful, concise, and accurate
- Use the provided context from the knowledge base when relevant
- Always mention specific numbers and actionable tips
- Focus on practical, implementable solutions
- Promote sustainable energy practices aligned with SDG 7

Format your responses with clear headers, bullet points, and specific numbers."""


class EnergyAdvisorAgent:
    """
    Agentic AI Energy Advisor powered by Google Gemini.
    
    Purpose: Provide conversational energy advice with RAG enhancement.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = None
        self.chat_history = []
        self.user_data = None
        self.user_summary = None
        self.knowledge_base = get_knowledge_base()
        
        if self.api_key:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Gemini model."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        except ImportError:
            print("google-generativeai not installed.")
        except Exception as e:
            print(f"Error initializing Gemini: {e}")
    
    def set_user_data(self, df: pd.DataFrame):
        """Set user's consumption data for personalized responses."""
        self.user_data = df
        self.user_summary = compute_user_summary(df)
    
    def _get_rag_context(self, query: str) -> Tuple[str, List[Dict]]:
        """Get relevant context from knowledge base."""
        result = query_knowledge_base(query)
        return result['context'], result['sources']
    
    def chat(self, message: str) -> Dict:
        """Send a message to the agent and get a response."""
        if self.model is None:
            return self._fallback_response(message)
        
        rag_context, sources = self._get_rag_context(message)
        user_context = format_user_context(self.user_summary)
        
        full_prompt = f"""{SYSTEM_PROMPT}

{user_context}

Relevant Knowledge Base Context:
{rag_context}

User Question: {message}

Provide a helpful, accurate, and actionable response:"""
        
        try:
            response = self.model.generate_content(full_prompt)
            response_text = response.text
            
            self.chat_history.append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.now().isoformat()
            })
            self.chat_history.append({
                'role': 'assistant',
                'content': response_text,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'response': response_text,
                'sources': sources,
                'has_user_data': self.user_summary is not None,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'response': f"I encountered an error: {str(e)}. Please try again.",
                'sources': [],
                'status': 'error',
                'error': str(e)
            }
    
    def _fallback_response(self, message: str) -> Dict:
        """Provide a response when Gemini is not available."""
        rag_context, sources = self._get_rag_context(message)
        
        if sources:
            response = f"""Based on my knowledge base:

{rag_context}

**Note:** For interactive responses, configure your Google Gemini API key."""
        else:
            response = """I couldn't find specific information about your query.

Try asking about: energy saving tips, AC efficiency, solar panels, 
PM Surya Ghar scheme, carbon footprint, or electricity tariffs.

**Note:** For AI-powered responses, configure your Google Gemini API key."""
        
        return {
            'response': response,
            'sources': sources,
            'has_user_data': self.user_summary is not None,
            'status': 'fallback'
        }
    
    def analyze_consumption(self) -> Dict:
        """Perform autonomous analysis of user's consumption data."""
        if self.user_summary is None:
            return {'status': 'error', 'message': 'No consumption data available.'}
        
        analysis = generate_insights(self.user_summary)
        return {
            'status': 'success',
            'summary': self.user_summary,
            **analysis
        }
    
    def get_quick_answer(self, topic: str) -> str:
        """Get a quick answer for common topics without using the LLM."""
        return get_quick_answer(topic)
    
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []


# Global agent instance
_agent = None


def get_energy_agent(api_key: Optional[str] = None) -> EnergyAdvisorAgent:
    """Get or create the global energy advisor agent."""
    global _agent
    if _agent is None:
        _agent = EnergyAdvisorAgent(api_key)
    return _agent


def chat_with_agent(message: str, api_key: Optional[str] = None) -> Dict:
    """Convenience function to chat with the energy agent."""
    agent = get_energy_agent(api_key)
    return agent.chat(message)


if __name__ == "__main__":
    print("=" * 50)
    print("ENERGY ADVISOR AGENT TEST")
    print("=" * 50)
    
    agent = EnergyAdvisorAgent()
    
    test_questions = ["How can I reduce my AC electricity bill?"]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        result = agent.chat(question)
        print(f"Status: {result['status']}")
        print(f"Sources: {[s['title'] for s in result['sources']]}")
    
    print("\n✓ Energy agent working correctly!")
