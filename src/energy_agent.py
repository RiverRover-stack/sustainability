"""
Smart AI Energy Consumption Predictor
Energy Advisor Agent Module

Agentic AI powered by Google Gemini for conversational energy advice.
"""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

# Import knowledge base
try:
    from knowledge_base import get_knowledge_base, query_knowledge_base
except ImportError:
    from src.knowledge_base import get_knowledge_base, query_knowledge_base


class EnergyAdvisorAgent:
    """
    Agentic AI Energy Advisor powered by Google Gemini.
    
    Features:
    - RAG-enhanced responses using knowledge base
    - Context-aware analysis of user's consumption data
    - Personalized recommendations
    - Conversational interface
    """
    
    SYSTEM_PROMPT = """You are an expert Energy Advisor AI assistant for the Smart AI Energy Consumption Predictor application. Your role is to:

1. Answer questions about energy consumption, savings, and sustainability
2. Provide personalized recommendations based on user's consumption data
3. Explain complex energy concepts in simple terms
4. Help users reduce their electricity bills and carbon footprint
5. Provide information about solar energy and government schemes

Guidelines:
- Be helpful, concise, and accurate
- Use the provided context from the knowledge base when relevant
- If user's consumption data is provided, use it for personalized advice
- Always mention specific numbers and actionable tips
- Focus on practical, implementable solutions
- If you don't know something, say so honestly
- Promote sustainable energy practices aligned with SDG 7

Format your responses with:
- Clear headers when appropriate
- Bullet points for lists
- Specific numbers and percentages
- Actionable recommendations
"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Energy Advisor Agent.
        
        Args:
            api_key: Google Gemini API key (or set GOOGLE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = None
        self.chat_history = []
        self.user_data = None
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
            print("google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            print(f"Error initializing Gemini: {e}")
    
    def set_user_data(self, df: pd.DataFrame):
        """
        Set user's consumption data for personalized responses.
        
        Args:
            df: DataFrame with consumption data
        """
        self.user_data = df
        self._compute_user_summary()
    
    def _compute_user_summary(self):
        """Compute summary statistics from user data."""
        if self.user_data is None:
            self.user_summary = None
            return
        
        df = self.user_data
        
        # Calculate key metrics
        total_kwh = df['consumption_kwh'].sum()
        daily_avg = df.groupby(df['timestamp'].dt.date)['consumption_kwh'].sum().mean()
        
        # Monthly breakdown
        monthly = df.groupby(df['timestamp'].dt.month)['consumption_kwh'].sum()
        highest_month = monthly.idxmax()
        lowest_month = monthly.idxmin()
        
        # Peak hours analysis
        hourly = df.groupby('hour')['consumption_kwh'].mean()
        peak_hour = hourly.idxmax()
        
        # Weekend vs weekday
        weekend_avg = df[df['is_weekend'] == 1]['consumption_kwh'].mean()
        weekday_avg = df[df['is_weekend'] == 0]['consumption_kwh'].mean()
        
        # Carbon footprint
        carbon_kg = total_kwh * 0.82
        
        self.user_summary = {
            'total_kwh': round(total_kwh, 2),
            'daily_avg_kwh': round(daily_avg, 2),
            'monthly_avg_kwh': round(total_kwh / 12, 2),
            'highest_month': highest_month,
            'lowest_month': lowest_month,
            'peak_hour': peak_hour,
            'weekend_avg': round(weekend_avg, 3),
            'weekday_avg': round(weekday_avg, 3),
            'carbon_kg': round(carbon_kg, 2),
            'data_days': len(df['timestamp'].dt.date.unique())
        }
    
    def _get_user_context(self) -> str:
        """Get user data context for the prompt."""
        if self.user_summary is None:
            return "No user consumption data available."
        
        s = self.user_summary
        return f"""
User's Energy Consumption Summary:
- Total consumption: {s['total_kwh']} kWh over {s['data_days']} days
- Daily average: {s['daily_avg_kwh']} kWh
- Monthly average: {s['monthly_avg_kwh']} kWh
- Highest consumption month: Month {s['highest_month']}
- Peak usage hour: {s['peak_hour']}:00
- Weekend average: {s['weekend_avg']} kWh/hr (Weekday: {s['weekday_avg']} kWh/hr)
- Annual carbon footprint: {s['carbon_kg']} kg CO₂
"""
    
    def _get_rag_context(self, query: str) -> Tuple[str, List[Dict]]:
        """Get relevant context from knowledge base."""
        result = query_knowledge_base(query)
        return result['context'], result['sources']
    
    def chat(self, message: str) -> Dict:
        """
        Send a message to the agent and get a response.
        
        Args:
            message: User's message/question
            
        Returns:
            Dictionary with response and metadata
        """
        if self.model is None:
            return self._fallback_response(message)
        
        # Get RAG context
        rag_context, sources = self._get_rag_context(message)
        
        # Build the full prompt
        user_context = self._get_user_context()
        
        full_prompt = f"""{self.SYSTEM_PROMPT}

{user_context}

Relevant Knowledge Base Context:
{rag_context}

User Question: {message}

Provide a helpful, accurate, and actionable response:"""
        
        try:
            # Generate response
            response = self.model.generate_content(full_prompt)
            response_text = response.text
            
            # Store in history
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
        # Get RAG context
        rag_context, sources = self._get_rag_context(message)
        
        if sources:
            response = f"""Based on my knowledge base, here's what I found:

{rag_context}

**Note:** For more interactive and personalized responses, please configure your Google Gemini API key.
"""
        else:
            response = """I couldn't find specific information about your query in my knowledge base.

Please try asking about:
- Energy saving tips
- AC and appliance efficiency
- Solar panel installation
- Government schemes like PM Surya Ghar
- Carbon footprint reduction
- Electricity tariffs

**Note:** For AI-powered conversational responses, please configure your Google Gemini API key.
"""
        
        return {
            'response': response,
            'sources': sources,
            'has_user_data': self.user_summary is not None,
            'status': 'fallback'
        }
    
    def analyze_consumption(self) -> Dict:
        """
        Perform autonomous analysis of user's consumption data.
        
        Returns:
            Dictionary with analysis results and recommendations
        """
        if self.user_summary is None:
            return {
                'status': 'error',
                'message': 'No consumption data available for analysis.'
            }
        
        analysis = {
            'status': 'success',
            'summary': self.user_summary,
            'insights': [],
            'recommendations': []
        }
        
        s = self.user_summary
        
        # Generate insights
        if s['weekend_avg'] > s['weekday_avg'] * 1.3:
            analysis['insights'].append({
                'type': 'pattern',
                'finding': 'Weekend consumption is significantly higher than weekdays',
                'detail': f"Weekend: {s['weekend_avg']:.3f} kWh/hr vs Weekday: {s['weekday_avg']:.3f} kWh/hr"
            })
            analysis['recommendations'].append("Consider energy-conscious activities on weekends")
        
        if s['peak_hour'] in [18, 19, 20, 21]:
            analysis['insights'].append({
                'type': 'peak',
                'finding': f"Peak consumption occurs in the evening at {s['peak_hour']}:00",
                'detail': "Evening peak hours often have higher tariff rates"
            })
            analysis['recommendations'].append("Shift some activities to off-peak hours (post 11 PM)")
        
        if s['daily_avg_kwh'] > 15:
            analysis['insights'].append({
                'type': 'high_usage',
                'finding': 'Daily consumption is above typical household average',
                'detail': f"Your daily average: {s['daily_avg_kwh']} kWh (typical: 10-15 kWh)"
            })
            analysis['recommendations'].append("Consider energy audit to identify high-consumption appliances")
        
        if s['carbon_kg'] > 2000:
            analysis['insights'].append({
                'type': 'carbon',
                'finding': 'Annual carbon footprint exceeds 2 tonnes CO₂',
                'detail': f"Your footprint: {s['carbon_kg']} kg CO₂"
            })
            analysis['recommendations'].append("Consider solar installation to reduce carbon footprint")
        
        # Add solar recommendation for high consumers
        if s['daily_avg_kwh'] > 10:
            recommended_solar = max(2, int(s['daily_avg_kwh'] / 4))
            analysis['recommendations'].append(
                f"A {recommended_solar}kW solar system could offset most of your consumption"
            )
        
        return analysis
    
    def get_quick_answer(self, topic: str) -> str:
        """
        Get a quick answer for common topics without using the LLM.
        
        Args:
            topic: Topic keyword
            
        Returns:
            Quick answer string
        """
        quick_answers = {
            'ac': "Set AC to 24-26°C to save up to 6% per degree. Use ceiling fans to distribute cool air. Clean filters monthly for 5-15% efficiency gain.",
            'led': "LED bulbs use 80% less electricity than incandescent and last 25x longer. Switching all lights to LED can save ₹2,000-5,000/year.",
            'solar': "A 3kW solar system generates 12-15 units/day, saving ₹3,000-4,500/month. PM Surya Ghar offers up to ₹78,000 subsidy.",
            'carbon': "India's grid emission factor is 0.82 kg CO₂/kWh. Reducing 100 units/month saves 82 kg CO₂, equivalent to 4 trees.",
            'off-peak': "Running appliances during off-peak hours (11 PM - 6 AM) can reduce bills by 15-25% with time-of-use tariffs.",
            'standby': "Phantom loads from standby devices account for 5-10% of energy use. Use power strips with switches to eliminate standby waste.",
            '5-star': "5-star appliances use 20-30% less energy than 3-star. Higher upfront cost is recovered in 2-3 years through savings.",
            'inverter': "Inverter ACs/refrigerators save 30-50% energy by adjusting compressor speed. BLDC fans use 60% less than regular fans."
        }
        
        topic_lower = topic.lower()
        for key, answer in quick_answers.items():
            if key in topic_lower:
                return answer
        
        return "I don't have a quick answer for that topic. Please ask a detailed question."
    
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
    """
    Convenience function to chat with the energy agent.
    
    Args:
        message: User's message
        api_key: Optional API key
        
    Returns:
        Response dictionary
    """
    agent = get_energy_agent(api_key)
    return agent.chat(message)


if __name__ == "__main__":
    # Test the agent
    print("=" * 60)
    print("ENERGY ADVISOR AGENT TEST")
    print("=" * 60)
    
    agent = EnergyAdvisorAgent()
    
    # Test without API key (fallback mode)
    test_questions = [
        "How can I reduce my AC electricity bill?",
        "What is the PM Surya Ghar scheme?",
        "How much CO2 does my electricity use produce?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        print("-" * 40)
        result = agent.chat(question)
        print(f"Status: {result['status']}")
        print(f"Sources: {[s['title'] for s in result['sources']]}")
        print(f"Response preview: {result['response'][:300]}...")
