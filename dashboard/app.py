import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import pickle
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generator import generate_data, calculate_bill, get_tariff_slabs
from preprocessing import load_data, prepare_features, prepare_data_for_ml
from recommender import EnergyRecommender, get_recommendations_for_dashboard
from carbon_calculator import CarbonCalculator, calculate_carbon_footprint
from energy_agent import EnergyAdvisorAgent, get_energy_agent
from knowledge_base import get_knowledge_base, query_knowledge_base

# Page config
st.set_page_config(
    page_title="Smart Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark mode compatible
st.markdown("""
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
    .priority-high {
        border-left-color: #e53935;
    }
    .priority-medium {
        border-left-color: #fb8c00;
    }
    .priority-low {
        border-left-color: #43a047;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_energy_data():
    """Load or generate energy data."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'energy_data.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
    else:
        # Generate data if not exists
        st.info("Generating synthetic energy data...")
        df = generate_data(save_path=data_path)
    
    return df


@st.cache_resource
def load_models():
    """Load trained models."""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    models = {}
    
    model_files = {
        'Linear Regression': 'linear_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }
    
    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    
    return models


def create_consumption_chart(df, period='daily'):
    """Create consumption visualization."""
    if period == 'hourly':
        chart_df = df.tail(168)  # Last 7 days hourly
        fig = px.line(chart_df, x='timestamp', y='consumption_kwh',
                     title='Hourly Consumption (Last 7 Days)',
                     labels={'consumption_kwh': 'Consumption (kWh)', 'timestamp': 'Time'})
    elif period == 'daily':
        chart_df = df.groupby(df['timestamp'].dt.date).agg({
            'consumption_kwh': 'sum',
            'temperature': 'mean'
        }).reset_index()
        chart_df.columns = ['date', 'consumption_kwh', 'avg_temperature']
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=chart_df['date'], y=chart_df['consumption_kwh'], 
                   name='Consumption', marker_color='#1E88E5'),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=chart_df['date'], y=chart_df['avg_temperature'],
                      name='Temperature', line=dict(color='#FF5722', width=2)),
            secondary_y=True
        )
        fig.update_layout(title='Daily Consumption vs Temperature')
        fig.update_yaxes(title_text="Consumption (kWh)", secondary_y=False)
        fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    else:  # monthly
        chart_df = df.groupby(df['timestamp'].dt.to_period('M')).agg({
            'consumption_kwh': 'sum'
        }).reset_index()
        chart_df['timestamp'] = chart_df['timestamp'].astype(str)
        
        fig = px.bar(chart_df, x='timestamp', y='consumption_kwh',
                    title='Monthly Consumption',
                    labels={'consumption_kwh': 'Consumption (kWh)', 'timestamp': 'Month'})
        fig.update_traces(marker_color='#1E88E5')
    
    fig.update_layout(
        template='plotly_dark',
        hovermode='x unified',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig


def create_forecast_chart(df, models, feature_names, scaler):
    """Create forecast visualization."""
    if not models:
        return None
    
    # Prepare recent data for prediction
    df_processed = prepare_features(df)
    recent = df_processed.tail(48).copy()  # Last 2 days
    
    exclude_cols = ['timestamp', 'consumption_kwh']
    feature_cols = [col for col in recent.columns if col not in exclude_cols]
    X = recent[feature_cols].values
    
    if scaler:
        X = scaler.transform(X)
    
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=recent['timestamp'],
        y=recent['consumption_kwh'],
        mode='lines',
        name='Actual',
        line=dict(color='#1E88E5', width=2)
    ))
    
    # Model predictions
    colors = {'Linear Regression': '#43a047', 'Random Forest': '#fb8c00', 'XGBoost': '#e53935'}
    
    for name, model in models.items():
        if name in colors:
            try:
                pred = model.predict(X)
                fig.add_trace(go.Scatter(
                    x=recent['timestamp'],
                    y=pred,
                    mode='lines',
                    name=f'{name} Prediction',
                    line=dict(color=colors[name], width=2, dash='dot')
                ))
            except Exception as e:
                st.warning(f"Could not generate prediction for {name}: {e}")
    
    fig.update_layout(
        title='Model Predictions vs Actual',
        xaxis_title='Time',
        yaxis_title='Consumption (kWh)',
        template='plotly_dark',
        hovermode='x unified',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig


def create_carbon_chart(df):
    """Create carbon footprint visualization."""
    calc = CarbonCalculator()
    monthly = calc.calculate_monthly_emissions(df)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly['month_name'],
        y=monthly['total_co2_kg'],
        marker_color='#43a047',
        name='CO₂ Emissions'
    ))
    
    # Add benchmark line
    benchmark = 150  # kg CO2 per month
    fig.add_hline(y=benchmark, line_dash="dash", line_color="red",
                  annotation_text="Indian Household Average")
    
    fig.update_layout(
        title='Monthly Carbon Emissions',
        xaxis_title='Month',
        yaxis_title='CO₂ Emissions (kg)',
        template='plotly_dark',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig


def create_peak_analysis_chart(df):
    """Create peak hour analysis chart."""
    hourly = df.groupby('hour')['consumption_kwh'].mean().reset_index()
    
    # Define peak hours
    peak_hours = list(range(6, 10)) + list(range(18, 22))
    hourly['is_peak'] = hourly['hour'].isin(peak_hours)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hourly['hour'],
        y=hourly['consumption_kwh'],
        marker_color=hourly['is_peak'].map({True: '#e53935', False: '#1E88E5'}),
        name='Consumption'
    ))
    
    fig.update_layout(
        title='Average Consumption by Hour (Red = Peak Hours)',
        xaxis_title='Hour of Day',
        yaxis_title='Avg Consumption (kWh)',
        template='plotly_dark',
        height=350,
        xaxis=dict(tickmode='linear', dtick=2),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig


def display_recommendations(recommendations):
    """Display recommendations in cards."""
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


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">⚡ Smart AI Energy Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered energy consumption prediction and optimization | SDG 7 - Affordable and Clean Energy</p>', unsafe_allow_html=True)
    
    # Data source selection in sidebar
    st.sidebar.subheader("📊 Data Source")
    data_source = st.sidebar.radio(
        "Select data source:",
        ["Synthetic Data", "Uploaded Data"],
        help="Choose between demo data or your own uploaded data"
    )
    
    # Load data based on selection
    with st.spinner('Loading energy data...'):
        if data_source == "Uploaded Data" and 'uploaded_data' in st.session_state:
            df = st.session_state['uploaded_data']
            st.sidebar.success(f"✅ Using uploaded data ({len(df)} records)")
        else:
            df = load_energy_data()
            if data_source == "Uploaded Data":
                st.sidebar.warning("⚠️ No uploaded data. Upload data in the Data Upload page.")
        models = load_models()
    
    # Sidebar
    st.sidebar.image("https://sdgs.un.org/sites/default/files/goals/E_SDG_Icons-07.jpg", width=150)
    st.sidebar.title("📊 Dashboard Controls")
    
    page = st.sidebar.radio("Navigate", [
        "📈 Overview",
        "🔮 Forecasting",
        "💰 Bill Estimator",
        "🌍 Carbon Footprint",
        "💡 Recommendations",
        "🤖 AI Assistant",
        "📤 Data Upload"
    ])
    
    # Date range filter
    st.sidebar.subheader("📅 Date Range")
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    # Filter data
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df[mask].copy()
    
    if len(filtered_df) == 0:
        st.error("No data available for selected date range.")
        return
    
    # Page content
    if page == "📈 Overview":
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_kwh = filtered_df['consumption_kwh'].sum()
        avg_daily = filtered_df.groupby(filtered_df['timestamp'].dt.date)['consumption_kwh'].sum().mean()
        max_hourly = filtered_df['consumption_kwh'].max()
        avg_temp = filtered_df['temperature'].mean()
        
        col1.metric("Total Consumption", f"{total_kwh:,.0f} kWh", 
                   help="Total energy consumed in selected period")
        col2.metric("Daily Average", f"{avg_daily:.1f} kWh",
                   help="Average daily consumption")
        col3.metric("Peak Hourly", f"{max_hourly:.2f} kWh",
                   help="Maximum hourly consumption")
        col4.metric("Avg Temperature", f"{avg_temp:.1f}°C",
                   help="Average temperature in period")
        
        st.divider()
        
        # Consumption charts
        chart_period = st.radio("View Period", ['Daily', 'Hourly', 'Monthly'], horizontal=True)
        st.plotly_chart(create_consumption_chart(filtered_df, chart_period.lower()), 
                       use_container_width=True)
        
        # Peak analysis
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_peak_analysis_chart(filtered_df), use_container_width=True)
        
        with col2:
            # Day of week pattern
            daily_pattern = filtered_df.groupby('day_of_week')['consumption_kwh'].mean().reset_index()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            daily_pattern['day_name'] = daily_pattern['day_of_week'].map(lambda x: day_names[x])
            
            fig = px.bar(daily_pattern, x='day_name', y='consumption_kwh',
                        title='Average Consumption by Day of Week',
                        labels={'consumption_kwh': 'Avg kWh', 'day_name': 'Day'})
            fig.update_traces(marker_color='#1E88E5')
            fig.update_layout(template='plotly_dark', height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔮 Forecasting":
        st.subheader("🔮 Energy Consumption Forecasting")
        
        if models:
            st.success(f"✅ {len(models)} models loaded: {', '.join(models.keys())}")
            
            # Load model comparison
            comparison_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_comparison.csv')
            if os.path.exists(comparison_path):
                results = pd.read_csv(comparison_path)
                
                st.write("**Model Performance Comparison:**")
                
                col1, col2, col3 = st.columns(3)
                for i, row in results.iterrows():
                    cols = [col1, col2, col3]
                    if i < 3:
                        with cols[i]:
                            st.metric(
                                row['model'],
                                f"{row['MAPE']:.2f}% MAPE",
                                help=f"MAE: {row['MAE']:.4f}, RMSE: {row['RMSE']:.4f}"
                            )
                
                # Best model highlight
                best = results.loc[results['MAPE'].idxmin()]
                if best['MAPE'] < 10:
                    st.success(f"✅ Best Model: **{best['model']}** with {best['MAPE']:.2f}% error (below 10% target!)")
                else:
                    st.info(f"📊 Best Model: **{best['model']}** with {best['MAPE']:.2f}% error")
            
            st.divider()
            
            # Prediction chart
            df_processed = prepare_features(filtered_df)
            X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data_for_ml(df_processed)
            
            fig = create_forecast_chart(filtered_df, models, feature_names, scaler)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            importance_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_importance.csv')
            if os.path.exists(importance_path):
                st.subheader("🔍 Feature Importance (SHAP)")
                importance = pd.read_csv(importance_path).head(15)
                
                fig = px.bar(importance, x='importance', y='feature', orientation='h',
                            title='Top 15 Important Features',
                            labels={'importance': 'SHAP Importance', 'feature': 'Feature'})
                fig.update_traces(marker_color='#1E88E5')
                fig.update_layout(template='plotly_dark', height=500, yaxis={'categoryorder': 'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No trained models found. Please run the training script first:")
            st.code("cd d:\\sustainability\npython src/data_generator.py\npython src/train_models.py")
    
    elif page == "💰 Bill Estimator":
        st.subheader("💰 Electricity Bill Estimator")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Enter your consumption or use historical data:**")
            
            use_historical = st.checkbox("Use historical data", value=True)
            
            if use_historical:
                # Monthly consumption from data
                monthly = filtered_df.groupby(filtered_df['timestamp'].dt.to_period('M'))['consumption_kwh'].sum()
                if len(monthly) > 0:
                    monthly_kwh = monthly.iloc[-1]
                else:
                    monthly_kwh = 300
            else:
                monthly_kwh = st.number_input("Monthly Consumption (kWh)", 
                                             min_value=0, max_value=5000, value=300)
            
            st.metric("Monthly Consumption", f"{monthly_kwh:.0f} kWh")
        
        with col2:
            # Calculate bill
            bill = calculate_bill(monthly_kwh)
            
            st.write("**Bill Breakdown:**")
            
            # Create bill breakdown chart
            breakdown_df = pd.DataFrame(bill['breakdown'])
            
            fig = px.bar(breakdown_df, x='slab', y='cost',
                        title='Cost by Tariff Slab',
                        labels={'cost': 'Cost (₹)', 'slab': 'Slab'})
            fig.update_traces(marker_color='#1E88E5')
            fig.update_layout(template='plotly_dark', height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        
        # Bill summary
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Units", f"{bill['total_units']:.0f} kWh")
        col2.metric("Total Bill", f"₹{bill['total_bill']:,.2f}")
        col3.metric("Avg Rate", f"₹{bill['total_bill']/max(bill['total_units'],1):.2f}/kWh")
        
        # Tariff structure
        with st.expander("📋 View Tariff Structure"):
            tariffs = get_tariff_slabs()
            tariff_df = pd.DataFrame(tariffs)
            tariff_df['max_units'] = tariff_df['max_units'].apply(lambda x: '∞' if x == float('inf') else str(int(x)))
            st.dataframe(tariff_df, hide_index=True)
    
    elif page == "🌍 Carbon Footprint":
        st.subheader("🌍 Carbon Footprint Tracker")
        
        calc = CarbonCalculator()
        carbon_data = calculate_carbon_footprint(filtered_df)
        summary = carbon_data['summary']
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Monthly CO₂", f"{summary['co2_kg']:.1f} kg")
        col2.metric("Emission Factor", f"{summary['emission_factor']} kg/kWh")
        
        status_colors = {'excellent': '🌟', 'good': '✅', 'average': '⚠️', 'high': '🔴'}
        status = summary['benchmark_comparison']['status']
        col3.metric("Status", f"{status_colors.get(status, '')} {status.title()}")
        
        vs_benchmark = summary['benchmark_comparison']['vs_benchmark_percent']
        col4.metric("vs Average", f"{vs_benchmark:+.1f}%",
                   delta_color="inverse" if vs_benchmark > 0 else "normal")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_carbon_chart(filtered_df), use_container_width=True)
        
        with col2:
            st.write("**🌳 Environmental Equivalents:**")
            equiv = summary['equivalents']
            st.info(f"🚗 {equiv['car_description']}")
            st.info(f"✈️ {equiv['flight_description']}")
            st.info(f"🌲 {equiv['trees_description']}")
        
        # Reduction scenarios
        st.subheader("📉 Reduction Impact Scenarios")
        
        cols = st.columns(3)
        for i, scenario in enumerate(carbon_data['reduction_scenarios']):
            with cols[i]:
                st.metric(
                    f"{scenario['reduction_percent']}% Reduction",
                    f"{scenario['co2_saved_kg']:.1f} kg CO₂ saved/month",
                    help=f"Trees equivalent: {scenario['trees_equivalent']}"
                )
    
    elif page == "💡 Recommendations":
        st.subheader("💡 Personalized Energy Optimization Tips")
        
        recommender = EnergyRecommender()
        recommendations = recommender.generate_recommendations(filtered_df)
        
        if recommendations:
            # Priority filter
            priority_filter = st.multiselect(
                "Filter by Priority",
                ['high', 'medium', 'low'],
                default=['high', 'medium', 'low']
            )
            
            filtered_recs = [r for r in recommendations if r['priority'] in priority_filter]
            
            st.write(f"**{len(filtered_recs)} recommendations found:**")
            display_recommendations(filtered_recs)
        else:
            st.info("No specific recommendations at this time. Your energy usage is optimal!")
        
        st.divider()
        
        # Quick tips
        st.subheader("⚡ Quick Energy Saving Tips")
        tips = recommender.get_quick_tips()
        
        cols = st.columns(2)
        for i, tip in enumerate(tips):
            cols[i % 2].write(tip)
        
        # Savings calculator
        st.divider()
        st.subheader("💵 Savings Potential Calculator")
        
        monthly_kwh = filtered_df.groupby(filtered_df['timestamp'].dt.to_period('M'))['consumption_kwh'].sum().mean()
        scenarios = recommender.calculate_savings_potential(monthly_kwh)
        
        for key, scenario in scenarios.items():
            with st.expander(f"📊 {scenario['action']}"):
                cols = st.columns(4)
                cols[0].metric("Monthly Savings", f"₹{scenario['monthly_savings']:,.0f}")
                cols[1].metric("Annual Savings", f"₹{scenario['annual_savings']:,.0f}")
                cols[2].metric("Investment", f"₹{scenario['implementation_cost']:,}")
                cols[3].metric("Payback", f"{scenario['payback_months']:.0f} months" if scenario['payback_months'] > 0 else "Immediate")
    
    elif page == "🤖 AI Assistant":
        st.subheader("🤖 AI Energy Advisor")
        
        st.write("""
        Chat with our AI-powered energy advisor! Ask questions about:
        - Energy saving tips
        - Solar panel installation
        - Government schemes (PM Surya Ghar)
        - Carbon footprint reduction
        - Electricity tariffs
        """)
        
        # API Key configuration
        with st.sidebar.expander("⚙️ AI Settings"):
            api_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                help="Get your free API key from https://makersuite.google.com/app/apikey"
            )
            if api_key:
                st.session_state['gemini_api_key'] = api_key
                st.success("✅ API Key configured!")
        
        # Initialize chat history in session state
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        
        # Get API key from session state
        saved_api_key = st.session_state.get('gemini_api_key', None)
        
        # Create tabs for different AI features
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Auto Analysis", "📚 Knowledge Base"])
        
        with tab1:
            # Display chat history
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me anything about energy..."):
                # Add user message to history
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        agent = get_energy_agent(saved_api_key)
                        agent.set_user_data(filtered_df)
                        result = agent.chat(prompt)
                        
                        response = result['response']
                        st.markdown(response)
                        
                        # Show sources if available
                        if result['sources']:
                            with st.expander("📚 Sources"):
                                for source in result['sources']:
                                    st.write(f"• {source['title']} ({source['category']})")
                        
                        # Show status
                        if result['status'] == 'fallback':
                            st.info("💡 For AI-powered responses, configure your Gemini API key in settings.")
                
                # Add assistant message to history
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
            
            # Clear chat button
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_messages = []
                st.rerun()
        
        with tab2:
            st.write("**🔍 Autonomous Data Analysis**")
            st.write("Let the AI analyze your consumption data and provide insights.")
            
            if st.button("🚀 Run AI Analysis", type="primary"):
                with st.spinner("Analyzing your consumption patterns..."):
                    agent = get_energy_agent(saved_api_key)
                    agent.set_user_data(filtered_df)
                    analysis = agent.analyze_consumption()
                    
                    if analysis['status'] == 'success':
                        # Display summary
                        st.write("**📊 Consumption Summary:**")
                        summary = analysis['summary']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Daily Avg", f"{summary['daily_avg_kwh']} kWh")
                        col2.metric("Monthly Avg", f"{summary['monthly_avg_kwh']} kWh")
                        col3.metric("Peak Hour", f"{summary['peak_hour']}:00")
                        col4.metric("Carbon/Year", f"{summary['carbon_kg']} kg")
                        
                        st.divider()
                        
                        # Display insights
                        if analysis['insights']:
                            st.write("**💡 Key Insights:**")
                            for insight in analysis['insights']:
                                st.info(f"**{insight['finding']}**\n\n{insight['detail']}")
                        
                        # Display recommendations
                        if analysis['recommendations']:
                            st.write("**✅ AI Recommendations:**")
                            for rec in analysis['recommendations']:
                                st.success(f"• {rec}")
                    else:
                        st.error(analysis.get('message', 'Analysis failed'))
        
        with tab3:
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
            
            # Search knowledge base
            search_query = st.text_input("🔍 Search knowledge base:", placeholder="e.g., how to save AC electricity")
            
            if search_query:
                result = query_knowledge_base(search_query)
                
                st.write("**Search Results:**")
                for source in result['sources']:
                    with st.expander(f"📄 {source['title']} (Score: {source.get('score', 'N/A'):.2f})"):
                        kb = get_knowledge_base()
                        for doc in kb.documents:
                            if doc['title'] == source['title']:
                                st.write(doc['content'])
                                break
    
    elif page == "📤 Data Upload":
        st.subheader("📤 Upload Your Own Data")
        
        st.write("""
        Upload your own energy consumption data to get personalized predictions and recommendations.
        You can use data from smart meters, electricity bills, or manual tracking.
        """)
        
        tab1, tab2, tab3 = st.tabs(["📁 CSV Upload", "✍️ Manual Entry", "📋 Data Format"])
        
        with tab1:
            st.write("**Upload a CSV file with your energy data:**")
            
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file with timestamp and consumption columns"
            )
            
            if uploaded_file is not None:
                try:
                    # Try to read the uploaded file
                    uploaded_df = pd.read_csv(uploaded_file)
                    
                    st.write("**Preview of uploaded data:**")
                    st.dataframe(uploaded_df.head(10))
                    
                    st.write(f"**Rows:** {len(uploaded_df)} | **Columns:** {list(uploaded_df.columns)}")
                    
                    # Column mapping
                    st.write("---")
                    st.write("**Map your columns:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        timestamp_col = st.selectbox(
                            "Timestamp column",
                            options=uploaded_df.columns.tolist(),
                            index=0 if 'timestamp' not in uploaded_df.columns else uploaded_df.columns.tolist().index('timestamp')
                        )
                    
                    with col2:
                        consumption_col = st.selectbox(
                            "Consumption (kWh) column",
                            options=uploaded_df.columns.tolist(),
                            index=1 if len(uploaded_df.columns) > 1 else 0
                        )
                    
                    # Optional columns
                    st.write("**Optional columns (leave blank if not available):**")
                    col1, col2, col3 = st.columns(3)
                    
                    all_cols = ['None'] + uploaded_df.columns.tolist()
                    with col1:
                        temp_col = st.selectbox("Temperature column", options=all_cols)
                    with col2:
                        humidity_col = st.selectbox("Humidity column", options=all_cols)
                    with col3:
                        occupancy_col = st.selectbox("Occupancy column", options=all_cols)
                    
                    if st.button("✅ Process and Use This Data", type="primary"):
                        # Process the uploaded data
                        processed_df = pd.DataFrame()
                        processed_df['timestamp'] = pd.to_datetime(uploaded_df[timestamp_col])
                        processed_df['consumption_kwh'] = pd.to_numeric(uploaded_df[consumption_col], errors='coerce')
                        
                        # Add optional columns or generate defaults
                        if temp_col != 'None':
                            processed_df['temperature'] = pd.to_numeric(uploaded_df[temp_col], errors='coerce')
                        else:
                            processed_df['temperature'] = 28  # Default temperature
                        
                        if humidity_col != 'None':
                            processed_df['humidity'] = pd.to_numeric(uploaded_df[humidity_col], errors='coerce')
                        else:
                            processed_df['humidity'] = 60  # Default humidity
                        
                        if occupancy_col != 'None':
                            processed_df['occupancy'] = pd.to_numeric(uploaded_df[occupancy_col], errors='coerce')
                        else:
                            processed_df['occupancy'] = 0.7  # Default occupancy
                        
                        # Add time-based features
                        processed_df['hour'] = processed_df['timestamp'].dt.hour
                        processed_df['day_of_week'] = processed_df['timestamp'].dt.dayofweek
                        processed_df['month'] = processed_df['timestamp'].dt.month
                        processed_df['is_weekend'] = (processed_df['day_of_week'] >= 5).astype(int)
                        
                        # Add appliance flags (defaults)
                        processed_df['ac_running'] = ((processed_df['temperature'] > 28) & (processed_df['occupancy'] > 0.5)).astype(int)
                        processed_df['lighting'] = ((processed_df['hour'] >= 18) | (processed_df['hour'] <= 6)).astype(int)
                        processed_df['cooking'] = processed_df['hour'].isin([7, 8, 12, 13, 19, 20]).astype(int)
                        processed_df['entertainment'] = ((processed_df['hour'] >= 18) & (processed_df['is_weekend'] == 1)).astype(int)
                        
                        # Drop NaN rows
                        processed_df = processed_df.dropna()
                        
                        # Store in session state
                        st.session_state['uploaded_data'] = processed_df
                        
                        st.success(f"✅ Data processed successfully! {len(processed_df)} records loaded.")
                        st.info("👆 Select 'Uploaded Data' in the sidebar to use your data.")
                        
                        # Show processed data preview
                        st.write("**Processed data preview:**")
                        st.dataframe(processed_df.head())
                        
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        with tab2:
            st.write("**Enter consumption data manually:**")
            
            st.write("Quick entry for monthly/daily consumption:")
            
            entry_type = st.radio("Entry type", ["Monthly totals", "Daily readings"], horizontal=True)
            
            if entry_type == "Monthly totals":
                st.write("Enter your monthly consumption (kWh):")
                
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                monthly_data = {}
                cols = st.columns(4)
                for i, month in enumerate(months):
                    with cols[i % 4]:
                        monthly_data[month] = st.number_input(
                            month, min_value=0.0, max_value=10000.0, value=300.0, key=f"month_{i}"
                        )
                
                if st.button("📊 Generate Hourly Data from Monthly", type="primary"):
                    # Generate hourly data from monthly totals
                    from data_generator import generate_data
                    
                    # Generate base pattern then scale to match monthly totals
                    base_df = generate_data(days=365)
                    
                    # Scale each month to match user input
                    for i, month in enumerate(months, 1):
                        month_mask = base_df['month'] == i
                        current_sum = base_df.loc[month_mask, 'consumption_kwh'].sum()
                        if current_sum > 0:
                            scale_factor = monthly_data[month] / current_sum
                            base_df.loc[month_mask, 'consumption_kwh'] *= scale_factor
                    
                    st.session_state['uploaded_data'] = base_df
                    st.success(f"✅ Generated {len(base_df)} hourly records from your monthly data!")
                    st.info("👆 Select 'Uploaded Data' in the sidebar to use your data.")
            
            else:  # Daily readings
                st.write("Enter daily readings for the past week:")
                
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_data = {}
                
                cols = st.columns(4)
                for i, day in enumerate(days):
                    with cols[i % 4]:
                        daily_data[day] = st.number_input(
                            day, min_value=0.0, max_value=500.0, value=15.0, key=f"day_{i}"
                        )
                
                if st.button("📊 Generate Data from Daily Pattern", type="primary"):
                    # Generate a year of data based on weekly pattern
                    from data_generator import generate_data
                    base_df = generate_data(days=365)
                    
                    # Scale by day of week
                    avg_daily = sum(daily_data.values()) / 7
                    for i, day in enumerate(days):
                        day_mask = base_df['day_of_week'] == i
                        current_avg = base_df.loc[day_mask, 'consumption_kwh'].mean() * 24
                        if current_avg > 0:
                            scale_factor = daily_data[day] / current_avg
                            base_df.loc[day_mask, 'consumption_kwh'] *= scale_factor
                    
                    st.session_state['uploaded_data'] = base_df
                    st.success(f"✅ Generated {len(base_df)} hourly records from your weekly pattern!")
                    st.info("👆 Select 'Uploaded Data' in the sidebar to use your data.")
        
        with tab3:
            st.write("**Required CSV Format:**")
            
            st.markdown("""
            Your CSV file should have at minimum these columns:
            
            | Column | Description | Format |
            |--------|-------------|--------|
            | `timestamp` | Date and time | `YYYY-MM-DD HH:MM:SS` |
            | `consumption_kwh` | Energy consumed | Numeric (kWh) |
            
            **Optional columns for better predictions:**
            
            | Column | Description |
            |--------|-------------|
            | `temperature` | Ambient temperature (°C) |
            | `humidity` | Humidity percentage (%) |
            | `occupancy` | Occupancy level (0-1) |
            """)
            
            st.write("**Example CSV:**")
            example_data = pd.DataFrame({
                'timestamp': ['2025-01-01 00:00:00', '2025-01-01 01:00:00', '2025-01-01 02:00:00'],
                'consumption_kwh': [0.5, 0.3, 0.25],
                'temperature': [22, 21, 20],
                'humidity': [65, 68, 70]
            })
            st.dataframe(example_data, hide_index=True)
            
            # Download sample template
            sample_csv = example_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Template",
                data=sample_csv,
                file_name="energy_data_template.csv",
                mime="text/csv"
            )
            
            st.divider()
            
            st.write("**Where to get real data:**")
            st.markdown("""
            - **Smart Meters**: Export data from your smart meter portal
            - **Electricity Bills**: Enter monthly totals manually
            - **IoT Devices**: Export from energy monitoring devices
            - **Utility APIs**: Many utilities provide data export options
            """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🌱 <strong>Smart AI Energy Predictor</strong> | Supporting SDG 7 - Affordable and Clean Energy</p>
        <p>Built with ❤️ for a sustainable future</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
