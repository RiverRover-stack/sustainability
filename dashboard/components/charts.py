"""
Dashboard Charts Module

File Responsibility:
    Creates Plotly visualizations for the dashboard.

Inputs:
    - DataFrames with consumption/carbon data
    - Model dictionaries for forecasting

Outputs:
    - Plotly Figure objects

Assumptions:
    - Data has standard column names
    - Plotly dark theme is used

Failure Modes:
    - Empty data returns minimal chart
    - Missing columns raise KeyError
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Standard chart layout settings
CHART_LAYOUT = dict(
    template='plotly_dark',
    hovermode='x unified',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)


def create_consumption_chart(df: pd.DataFrame, period: str = 'daily') -> go.Figure:
    """
    Create consumption visualization.
    
    Purpose: Display consumption trends over time.
    
    Inputs:
        df: DataFrame with timestamp and consumption_kwh
        period: 'hourly', 'daily', or 'monthly'
        
    Outputs:
        Plotly Figure object
    """
    if period == 'hourly':
        chart_df = df.tail(168)  # Last 7 days
        fig = px.line(
            chart_df, x='timestamp', y='consumption_kwh',
            title='Hourly Consumption (Last 7 Days)',
            labels={'consumption_kwh': 'Consumption (kWh)', 'timestamp': 'Time'}
        )
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
        
        fig = px.bar(
            chart_df, x='timestamp', y='consumption_kwh',
            title='Monthly Consumption',
            labels={'consumption_kwh': 'Consumption (kWh)', 'timestamp': 'Month'}
        )
        fig.update_traces(marker_color='#1E88E5')
    
    fig.update_layout(height=400, **CHART_LAYOUT)
    return fig


def create_forecast_chart(df: pd.DataFrame, models: dict, scaler=None) -> go.Figure:
    """
    Create forecast visualization comparing models.
    
    Purpose: Show model predictions vs actuals.
    
    Inputs:
        df: Processed DataFrame with features
        models: Dict of trained models
        scaler: Optional feature scaler
        
    Outputs:
        Plotly Figure or None if no models
    """
    if not models:
        return None
    
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    try:
        from data.preprocessing import prepare_features
    except ImportError:
        from preprocessing import prepare_features
    
    df_processed = prepare_features(df)
    recent = df_processed.tail(48).copy()
    
    exclude_cols = ['timestamp', 'consumption_kwh']
    feature_cols = [col for col in recent.columns if col not in exclude_cols]
    X = recent[feature_cols].values
    
    if scaler:
        X = scaler.transform(X)
    
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=recent['timestamp'], y=recent['consumption_kwh'],
        mode='lines', name='Actual',
        line=dict(color='#1E88E5', width=2)
    ))
    
    # Model predictions
    colors = {'Linear Regression': '#43a047', 'Random Forest': '#fb8c00', 'XGBoost': '#e53935'}
    
    for name, model in models.items():
        if name in colors:
            try:
                pred = model.predict(X)
                fig.add_trace(go.Scatter(
                    x=recent['timestamp'], y=pred,
                    mode='lines', name=f'{name} Prediction',
                    line=dict(color=colors[name], width=2, dash='dot')
                ))
            except Exception:
                pass
    
    fig.update_layout(
        title='Model Predictions vs Actual',
        xaxis_title='Time', yaxis_title='Consumption (kWh)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=400, **CHART_LAYOUT
    )
    
    return fig


def create_carbon_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create carbon footprint visualization.
    
    Purpose: Display monthly CO2 emissions.
    
    Inputs:
        df: DataFrame with consumption data
        
    Outputs:
        Plotly Figure with emissions bar chart
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    try:
        from carbon.carbon_calculator import CarbonCalculator
    except ImportError:
        from carbon_calculator import CarbonCalculator
    
    calc = CarbonCalculator()
    monthly = calc.calculate_monthly_emissions(df)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly['month_name'], y=monthly['total_co2_kg'],
        marker_color='#43a047', name='CO₂ Emissions'
    ))
    
    # Benchmark line
    fig.add_hline(y=150, line_dash="dash", line_color="red",
                  annotation_text="Indian Household Average")
    
    fig.update_layout(
        title='Monthly Carbon Emissions',
        xaxis_title='Month', yaxis_title='CO₂ Emissions (kg)',
        height=400, **CHART_LAYOUT
    )
    
    return fig


def create_peak_analysis_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create peak hour analysis chart.
    
    Purpose: Highlight peak vs off-peak consumption.
    
    Inputs:
        df: DataFrame with hour and consumption columns
        
    Outputs:
        Plotly Figure with hourly bar chart
    """
    hourly = df.groupby('hour')['consumption_kwh'].mean().reset_index()
    peak_hours = list(range(6, 10)) + list(range(18, 22))
    hourly['is_peak'] = hourly['hour'].isin(peak_hours)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly['hour'], y=hourly['consumption_kwh'],
        marker_color=hourly['is_peak'].map({True: '#e53935', False: '#1E88E5'}),
        name='Consumption'
    ))
    
    fig.update_layout(
        title='Average Consumption by Hour (Red = Peak Hours)',
        xaxis_title='Hour of Day', yaxis_title='Avg Consumption (kWh)',
        xaxis=dict(tickmode='linear', dtick=2),
        height=350, **CHART_LAYOUT
    )
    
    return fig
