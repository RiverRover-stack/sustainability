"""
Dashboard Display Helpers Module

File Responsibility:
    Provides helper functions for displaying UI elements.

Inputs:
    - Recommendations list
    - Various UI data

Outputs:
    - Rendered Streamlit components

Assumptions:
    - Running within Streamlit context

Failure Modes:
    - Invalid data silently skipped
"""

import streamlit as st


def get_custom_css() -> str:
    """Return custom CSS for dark mode dashboard."""
    return """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #aaa;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .stMetric {
        background: rgba(30, 136, 229, 0.15) !important;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(30, 136, 229, 0.3);
    }
    [data-testid="stMetricLabel"] {
        color: #aaa !important;
    }
    [data-testid="stMetricValue"] {
        color: #fff !important;
    }
    .recommendation-card {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .priority-high { border-left-color: #e53935; }
    .priority-medium { border-left-color: #fb8c00; }
    .priority-low { border-left-color: #43a047; }
</style>
"""


def display_recommendations(recommendations: list):
    """
    Display recommendations in expandable cards.
    
    Purpose: Render recommendation cards with actions.
    
    Inputs:
        recommendations: List of recommendation dicts
        
    Side effects: Renders Streamlit components
    """
    priority_colors = {
        'high': '🔴',
        'medium': '🟡',
        'low': '🟢'
    }
    
    for rec in recommendations:
        priority_icon = priority_colors.get(rec['priority'], '⚪')
        
        with st.expander(f"{priority_icon} {rec['title']} ({rec['category']})"):
            st.write(rec['description'])
            st.write(f"**Potential Savings:** {rec['potential_savings']}")
            
            if 'action_items' in rec:
                st.write("**Action Items:**")
                for item in rec['action_items']:
                    st.write(f"• {item}")
            
            if 'solar_estimate' in rec:
                st.write("**Solar Estimate:**")
                solar = rec['solar_estimate']
                cols = st.columns(3)
                cols[0].metric("Recommended Capacity", solar['recommended_capacity'])
                cols[1].metric("Estimated Cost", solar['estimated_cost'])
                cols[2].metric("Payback Period", solar['payback_period'])


def display_footer():
    """Display dashboard footer."""
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🌱 <strong>Smart AI Energy Predictor</strong> | Supporting SDG 7 - Affordable and Clean Energy</p>
        <p>Built with ❤️ for a sustainable future</p>
    </div>
    """, unsafe_allow_html=True)
