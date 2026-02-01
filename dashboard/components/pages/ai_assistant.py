"""
AI Assistant Page Module

File Responsibility:
    Handles the AI Assistant page with chat, analysis, and knowledge base.

Inputs:
    - Filtered DataFrame
    - Session state

Outputs:
    - Rendered Streamlit page

Assumptions:
    - Energy agent module available
    - Knowledge base initialized

Failure Modes:
    - Missing API key uses fallback mode
"""

import streamlit as st
import sys
import os

# Add src directory to path
_src_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

try:
    from agent.energy_agent import EnergyAdvisorAgent, get_energy_agent
    from agent.knowledge_base import get_knowledge_base, query_knowledge_base
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Agent import error - {e}")
    AGENT_AVAILABLE = False
    
    # Mock fallbacks
    class EnergyAdvisorAgent:
        def set_user_data(self, df): pass
        def chat(self, msg): return {'response': 'AI Agent not available. Please check imports.', 'sources': [], 'status': 'fallback'}
        def analyze_consumption(self): return {'status': 'error', 'message': 'Agent not available'}
        def get_quick_answer(self, topic): return f'Information about {topic} is not available.'
    
    def get_energy_agent(api_key=None): return EnergyAdvisorAgent()
    def get_knowledge_base(): return type('KB', (), {'documents': []})()
    def query_knowledge_base(q): return {'sources': []}


def render_ai_assistant_page(filtered_df):
    """Render the AI Assistant page."""
    st.subheader("🤖 AI Energy Advisor")
    
    st.write("""
    Chat with our AI-powered energy advisor! Ask questions about:
    - Energy saving tips
    - Solar panel installation
    - Government schemes (PM Surya Ghar)
    - Carbon footprint reduction
    - Electricity tariffs
    """)
    
    # Get API key from session state
    saved_api_key = st.session_state.get('gemini_api_key', None)
    
    # Initialize chat history
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Auto Analysis", "📚 Knowledge Base"])
    
    with tab1:
        _render_chat_tab(filtered_df, saved_api_key)
    
    with tab2:
        _render_analysis_tab(filtered_df, saved_api_key)
    
    with tab3:
        _render_knowledge_tab()


def _render_chat_tab(filtered_df, api_key):
    """Render the chat interface."""
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about energy..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent = get_energy_agent(api_key)
                agent.set_user_data(filtered_df)
                result = agent.chat(prompt)
                
                response = result['response']
                st.markdown(response)
                
                if result['sources']:
                    with st.expander("📚 Sources"):
                        for source in result['sources']:
                            st.write(f"• {source['title']} ({source['category']})")
                
                if result['status'] == 'fallback':
                    st.info("💡 For AI-powered responses, configure your Gemini API key.")
        
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_messages = []
        st.rerun()


def _render_analysis_tab(filtered_df, api_key):
    """Render the auto analysis tab."""
    st.write("**🔍 Autonomous Data Analysis**")
    st.write("Let the AI analyze your consumption data and provide insights.")
    
    if st.button("🚀 Run AI Analysis", type="primary"):
        with st.spinner("Analyzing your consumption patterns..."):
            agent = get_energy_agent(api_key)
            agent.set_user_data(filtered_df)
            analysis = agent.analyze_consumption()
            
            if analysis['status'] == 'success':
                st.write("**📊 Consumption Summary:**")
                summary = analysis['summary']
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Daily Avg", f"{summary['daily_avg_kwh']} kWh")
                col2.metric("Monthly Avg", f"{summary['monthly_avg_kwh']} kWh")
                col3.metric("Peak Hour", f"{summary['peak_hour']}:00")
                col4.metric("Carbon/Year", f"{summary['carbon_kg']} kg")
                
                st.divider()
                
                if analysis['insights']:
                    st.write("**💡 Key Insights:**")
                    for insight in analysis['insights']:
                        st.info(f"**{insight['finding']}**\n\n{insight['detail']}")
                
                if analysis['recommendations']:
                    st.write("**✅ AI Recommendations:**")
                    for rec in analysis['recommendations']:
                        st.success(f"• {rec}")
            else:
                st.error(analysis.get('message', 'Analysis failed'))


def _render_knowledge_tab():
    """Render the knowledge base tab."""
    st.write("**📚 Energy Knowledge Base**")
    st.write("Search our knowledge base for energy-related information.")
    
    # Quick topic buttons
    st.write("**Quick Topics:**")
    col1, col2, col3, col4 = st.columns(4)
    
    topics = ['AC', 'LED', 'Solar', 'Carbon', 'Off-Peak', 'Standby', '5-Star', 'Inverter']
    for i, topic in enumerate(topics):
        cols = [col1, col2, col3, col4]
        if cols[i % 4].button(topic, key=f"topic_{topic}"):
            agent = get_energy_agent()
            answer = agent.get_quick_answer(topic)
            st.info(f"**{topic}:** {answer}")
    
    st.divider()
    
    # Search
    search_query = st.text_input("🔍 Search knowledge base:", 
                                  placeholder="e.g., how to save AC electricity")
    
    if search_query:
        result = query_knowledge_base(search_query)
        st.write("**Search Results:**")
        for source in result['sources']:
            score = source.get('score', 0)
            with st.expander(f"📄 {source['title']} (Score: {score:.2f})"):
                kb = get_knowledge_base()
                for doc in kb.documents:
                    if doc['title'] == source['title']:
                        st.write(doc['content'])
                        break
