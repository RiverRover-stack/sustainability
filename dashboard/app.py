"""
Smart AI Energy Predictor Dashboard

File Responsibility:
    Main Streamlit application entry point.
    Orchestrates page navigation and renders dashboard pages.

Inputs:
    - Energy consumption data (CSV or generated)
    - Optional trained ML models

Outputs:
    - Interactive web dashboard

Assumptions:
    - Streamlit is installed
    - Data and models in standard locations

Failure Modes:
    - Missing data triggers generation
    - Missing models shows warning
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'components'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import components
from charts import (
    create_consumption_chart,
    create_forecast_chart,
    create_carbon_chart,
    create_peak_analysis_chart
)
from data_loader import load_energy_data, load_models
from display_helpers import display_recommendations, display_footer, get_custom_css
from pages.ai_assistant import render_ai_assistant_page
from pages.data_upload import render_data_upload_page

# Import src modules - use package imports
try:
    from data.tariff_calculator import calculate_bill, get_tariff_slabs
    from data.preprocessing import prepare_features, prepare_data_for_ml
    from recommender.engine import EnergyRecommender
    from carbon.carbon_calculator import CarbonCalculator, calculate_carbon_footprint
except ImportError as e:
    print(f"Warning: Import error - {e}")
    # Minimal fallback
    calculate_bill = lambda x: x * 5
    get_tariff_slabs = lambda: []
    EnergyRecommender = None
    CarbonCalculator = None
    calculate_carbon_footprint = lambda x: x * 0.82

# Page config
st.set_page_config(
    page_title="Smart Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


def render_overview_page(filtered_df):
    """Render the Overview page."""
    col1, col2, col3, col4 = st.columns(4)
    
    total_kwh = filtered_df['consumption_kwh'].sum()
    avg_daily = filtered_df.groupby(filtered_df['timestamp'].dt.date)['consumption_kwh'].sum().mean()
    max_hourly = filtered_df['consumption_kwh'].max()
    avg_temp = filtered_df['temperature'].mean()
    
    col1.metric("Total Consumption", f"{total_kwh:,.0f} kWh")
    col2.metric("Daily Average", f"{avg_daily:.1f} kWh")
    col3.metric("Peak Hourly", f"{max_hourly:.2f} kWh")
    col4.metric("Avg Temperature", f"{avg_temp:.1f}°C")
    
    st.divider()
    
    chart_period = st.radio("View Period", ['Daily', 'Hourly', 'Monthly'], horizontal=True)
    st.plotly_chart(create_consumption_chart(filtered_df, chart_period.lower()), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_peak_analysis_chart(filtered_df), use_container_width=True)
    
    with col2:
        daily_pattern = filtered_df.groupby('day_of_week')['consumption_kwh'].mean().reset_index()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_pattern['day_name'] = daily_pattern['day_of_week'].map(lambda x: day_names[x])
        
        fig = px.bar(daily_pattern, x='day_name', y='consumption_kwh',
                     title='Average Consumption by Day of Week')
        fig.update_traces(marker_color='#1E88E5')
        fig.update_layout(template='plotly_dark', height=350,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)


def render_forecasting_page(filtered_df, models):
    """Render the Forecasting page."""
    st.subheader("🔮 Energy Consumption Forecasting")
    
    if models:
        st.success(f"✅ {len(models)} models loaded: {', '.join(models.keys())}")
        
        comparison_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_comparison.csv')
        if os.path.exists(comparison_path):
            results = pd.read_csv(comparison_path)
            st.write("**Model Performance:**")
            
            cols = st.columns(3)
            for i, row in results.iterrows():
                if i < 3:
                    cols[i].metric(row['model'], f"{row['MAPE']:.2f}% MAPE")
            
            best = results.loc[results['MAPE'].idxmin()]
            if best['MAPE'] < 10:
                st.success(f"✅ Best: **{best['model']}** ({best['MAPE']:.2f}% error, below 10% target!)")
        
        st.divider()
        
        df_processed = prepare_features(filtered_df)
        _, _, _, _, feature_names, scaler = prepare_data_for_ml(df_processed)
        
        fig = create_forecast_chart(filtered_df, models, scaler)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        importance_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_importance.csv')
        if os.path.exists(importance_path):
            st.subheader("🔍 Feature Importance")
            importance = pd.read_csv(importance_path).head(15)
            fig = px.bar(importance, x='importance', y='feature', orientation='h',
                         title='Top 15 Features')
            fig.update_traces(marker_color='#1E88E5')
            fig.update_layout(template='plotly_dark', height=500,
                              yaxis={'categoryorder': 'total ascending'},
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No models found. Run training first.")
        st.code("python src/data_generator.py\npython src/train_models.py")


def render_bill_estimator_page(filtered_df):
    """Render the Bill Estimator page."""
    st.subheader("💰 Electricity Bill Estimator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_historical = st.checkbox("Use historical data", value=True)
        if use_historical:
            monthly = filtered_df.groupby(filtered_df['timestamp'].dt.to_period('M'))['consumption_kwh'].sum()
            monthly_kwh = monthly.iloc[-1] if len(monthly) > 0 else 300
        else:
            monthly_kwh = st.number_input("Monthly (kWh)", min_value=0, max_value=5000, value=300)
        st.metric("Monthly Consumption", f"{monthly_kwh:.0f} kWh")
    
    with col2:
        bill = calculate_bill(monthly_kwh)
        breakdown_df = pd.DataFrame(bill['breakdown'])
        fig = px.bar(breakdown_df, x='slab', y='cost', title='Cost by Tariff Slab')
        fig.update_traces(marker_color='#1E88E5')
        fig.update_layout(template='plotly_dark', height=300,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Units", f"{bill['total_units']:.0f} kWh")
    col2.metric("Total Bill", f"₹{bill['total_bill']:,.2f}")
    col3.metric("Avg Rate", f"₹{bill['total_bill']/max(bill['total_units'],1):.2f}/kWh")


def render_carbon_page(filtered_df):
    """Render the Carbon Footprint page."""
    st.subheader("🌍 Carbon Footprint Tracker")
    
    carbon_data = calculate_carbon_footprint(filtered_df)
    summary = carbon_data['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monthly CO₂", f"{summary['co2_kg']:.1f} kg")
    col2.metric("Emission Factor", f"{summary['emission_factor']} kg/kWh")
    
    status_icons = {'excellent': '🌟', 'good': '✅', 'average': '⚠️', 'high': '🔴'}
    status = summary['benchmark_comparison']['status']
    col3.metric("Status", f"{status_icons.get(status, '')} {status.title()}")
    
    vs_benchmark = summary['benchmark_comparison']['vs_benchmark_percent']
    col4.metric("vs Average", f"{vs_benchmark:+.1f}%", delta_color="inverse" if vs_benchmark > 0 else "normal")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    col1.plotly_chart(create_carbon_chart(filtered_df), use_container_width=True)
    
    with col2:
        st.write("**🌳 Environmental Equivalents:**")
        equiv = summary['equivalents']
        st.info(f"🚗 Equivalent to driving {equiv['car_km']:.0f} km")
        st.info(f"✈️ Equivalent to {equiv['flight_km']:.0f} km of air travel")
        st.info(f"🌲 {equiv['trees_needed']:.0f} trees needed to offset")


def render_recommendations_page(filtered_df):
    """Render the Recommendations page."""
    st.subheader("💡 Personalized Energy Optimization Tips")
    
    recommender = EnergyRecommender()
    recommendations = recommender.generate_recommendations(filtered_df)
    
    if recommendations:
        priority_filter = st.multiselect("Filter by Priority", ['high', 'medium', 'low'],
                                          default=['high', 'medium', 'low'])
        filtered_recs = [r for r in recommendations if r['priority'] in priority_filter]
        st.write(f"**{len(filtered_recs)} recommendations found:**")
        display_recommendations(filtered_recs)
    else:
        st.info("No recommendations - your usage is optimal!")
    
    st.divider()
    st.subheader("⚡ Quick Tips")
    tips = recommender.get_quick_tips()
    cols = st.columns(2)
    for i, tip in enumerate(tips):
        cols[i % 2].write(tip)


def main():
    """Main dashboard function."""
    # Header
    st.markdown('<h1 class="main-header">⚡ Smart AI Energy Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered prediction | SDG 7 - Clean Energy</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.subheader("📊 Data Source")
    data_source = st.sidebar.radio("Source:", ["Synthetic Data", "Uploaded Data"])
    
    # Load data
    with st.spinner('Loading data...'):
        if data_source == "Uploaded Data" and 'uploaded_data' in st.session_state:
            df = st.session_state['uploaded_data']
            st.sidebar.success(f"✅ Using uploaded ({len(df)} records)")
        else:
            df = load_energy_data()
            if data_source == "Uploaded Data":
                st.sidebar.warning("⚠️ No data uploaded")
        models = load_models()
    
    # Navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Go to", [
        "📈 Overview", "🔮 Forecasting", "💰 Bill Estimator",
        "🌍 Carbon Footprint", "💡 Recommendations",
        "🤖 AI Assistant", "📤 Data Upload"
    ])
    
    # Date filter
    st.sidebar.subheader("📅 Date Range")
    min_date, max_date = df['timestamp'].min().date(), df['timestamp'].max().date()
    start_date = st.sidebar.date_input("Start", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End", max_date, min_value=min_date, max_value=max_date)
    
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df[mask].copy()
    
    if len(filtered_df) == 0:
        st.error("No data for selected range.")
        return
    
    # API Key settings
    with st.sidebar.expander("⚙️ AI Settings"):
        api_key = st.text_input("Gemini API Key", type="password")
        if api_key:
            st.session_state['gemini_api_key'] = api_key
            st.success("✅ API Key set!")
    
    # Render page
    if page == "📈 Overview":
        render_overview_page(filtered_df)
    elif page == "🔮 Forecasting":
        render_forecasting_page(filtered_df, models)
    elif page == "💰 Bill Estimator":
        render_bill_estimator_page(filtered_df)
    elif page == "🌍 Carbon Footprint":
        render_carbon_page(filtered_df)
    elif page == "💡 Recommendations":
        render_recommendations_page(filtered_df)
    elif page == "🤖 AI Assistant":
        render_ai_assistant_page(filtered_df)
    elif page == "📤 Data Upload":
        render_data_upload_page()
    
    display_footer()


if __name__ == "__main__":
    main()
