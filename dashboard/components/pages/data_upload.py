"""
Data Upload Page Module

File Responsibility:
    Handles the data upload page with CSV upload and manual entry.

Inputs:
    - Uploaded files
    - Manual data entry

Outputs:
    - Processed DataFrame stored in session state

Assumptions:
    - Data has timestamp and consumption columns
    - Streamlit session state available

Failure Modes:
    - Invalid file format shows error
    - Missing columns use defaults
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


def render_data_upload_page():
    """Render the data upload page."""
    st.subheader("📤 Upload Your Own Data")
    
    st.write("""
    Upload your own energy consumption data to get personalized predictions and recommendations.
    You can use data from smart meters, electricity bills, or manual tracking.
    """)
    
    tab1, tab2, tab3 = st.tabs(["📁 CSV Upload", "✍️ Manual Entry", "📋 Data Format"])
    
    with tab1:
        _render_csv_upload_tab()
    
    with tab2:
        _render_manual_entry_tab()
    
    with tab3:
        _render_data_format_tab()


def _render_csv_upload_tab():
    """Render CSV upload interface."""
    st.write("**Upload a CSV file with your energy data:**")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=['csv'],
        help="Upload a CSV file with timestamp and consumption columns"
    )
    
    if uploaded_file is not None:
        try:
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
                    index=0 if 'timestamp' not in uploaded_df.columns else 
                          uploaded_df.columns.tolist().index('timestamp')
                )
            
            with col2:
                consumption_col = st.selectbox(
                    "Consumption (kWh) column",
                    options=uploaded_df.columns.tolist(),
                    index=1 if len(uploaded_df.columns) > 1 else 0
                )
            
            # Optional columns
            st.write("**Optional columns:**")
            all_cols = ['None'] + uploaded_df.columns.tolist()
            col1, col2, col3 = st.columns(3)
            temp_col = col1.selectbox("Temperature", options=all_cols)
            humidity_col = col2.selectbox("Humidity", options=all_cols)
            occupancy_col = col3.selectbox("Occupancy", options=all_cols)
            
            if st.button("✅ Process and Use This Data", type="primary"):
                processed_df = _process_uploaded_data(
                    uploaded_df, timestamp_col, consumption_col,
                    temp_col, humidity_col, occupancy_col
                )
                st.session_state['uploaded_data'] = processed_df
                st.success(f"✅ Data processed! {len(processed_df)} records loaded.")
                st.info("👆 Select 'Uploaded Data' in the sidebar.")
                st.dataframe(processed_df.head())
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")


def _process_uploaded_data(df, ts_col, consumption_col, temp_col, humidity_col, occupancy_col):
    """Process uploaded data into standard format."""
    processed = pd.DataFrame()
    processed['timestamp'] = pd.to_datetime(df[ts_col])
    processed['consumption_kwh'] = pd.to_numeric(df[consumption_col], errors='coerce')
    
    # Optional columns
    processed['temperature'] = pd.to_numeric(df[temp_col], errors='coerce') if temp_col != 'None' else 28
    processed['humidity'] = pd.to_numeric(df[humidity_col], errors='coerce') if humidity_col != 'None' else 60
    processed['occupancy'] = pd.to_numeric(df[occupancy_col], errors='coerce') if occupancy_col != 'None' else 0.7
    
    # Time features
    processed['hour'] = processed['timestamp'].dt.hour
    processed['day_of_week'] = processed['timestamp'].dt.dayofweek
    processed['month'] = processed['timestamp'].dt.month
    processed['is_weekend'] = (processed['day_of_week'] >= 5).astype(int)
    
    # Appliance flags
    processed['ac_running'] = ((processed['temperature'] > 28) & (processed['occupancy'] > 0.5)).astype(int)
    processed['lighting'] = ((processed['hour'] >= 18) | (processed['hour'] <= 6)).astype(int)
    processed['cooking'] = processed['hour'].isin([7, 8, 12, 13, 19, 20]).astype(int)
    processed['entertainment'] = ((processed['hour'] >= 18) & (processed['is_weekend'] == 1)).astype(int)
    
    return processed.dropna()


def _render_manual_entry_tab():
    """Render manual data entry interface."""
    st.write("**Enter consumption data manually:**")
    
    entry_type = st.radio("Entry type", ["Monthly totals", "Daily readings"], horizontal=True)
    
    if entry_type == "Monthly totals":
        _render_monthly_entry()
    else:
        _render_daily_entry()


def _render_monthly_entry():
    """Render monthly data entry."""
    st.write("Enter your monthly consumption (kWh):")
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_data = {}
    cols = st.columns(4)
    
    for i, month in enumerate(months):
        with cols[i % 4]:
            monthly_data[month] = st.number_input(month, min_value=0.0, max_value=10000.0, value=300.0, key=f"month_{i}")
    
    if st.button("📊 Generate Hourly Data from Monthly", type="primary"):
        from data_generator import generate_data
        base_df = generate_data(days=365)
        
        for i, month in enumerate(months, 1):
            mask = base_df['month'] == i
            current_sum = base_df.loc[mask, 'consumption_kwh'].sum()
            if current_sum > 0:
                base_df.loc[mask, 'consumption_kwh'] *= monthly_data[month] / current_sum
        
        st.session_state['uploaded_data'] = base_df
        st.success(f"✅ Generated {len(base_df)} hourly records!")


def _render_daily_entry():
    """Render daily data entry."""
    st.write("Enter daily readings for the past week:")
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_data = {}
    cols = st.columns(4)
    
    for i, day in enumerate(days):
        with cols[i % 4]:
            daily_data[day] = st.number_input(day, min_value=0.0, max_value=500.0, value=15.0, key=f"day_{i}")
    
    if st.button("📊 Generate Data from Daily Pattern", type="primary"):
        from data_generator import generate_data
        base_df = generate_data(days=365)
        
        for i, day in enumerate(days):
            mask = base_df['day_of_week'] == i
            current_avg = base_df.loc[mask, 'consumption_kwh'].mean() * 24
            if current_avg > 0:
                base_df.loc[mask, 'consumption_kwh'] *= daily_data[day] / current_avg
        
        st.session_state['uploaded_data'] = base_df
        st.success(f"✅ Generated {len(base_df)} hourly records!")


def _render_data_format_tab():
    """Render data format info."""
    st.write("**Required CSV Format:**")
    st.markdown("""
    | Column | Description | Format |
    |--------|-------------|--------|
    | `timestamp` | Date and time | `YYYY-MM-DD HH:MM:SS` |
    | `consumption_kwh` | Energy consumed | Numeric (kWh) |
    
    **Optional columns:** `temperature`, `humidity`, `occupancy`
    """)
    
    example_data = pd.DataFrame({
        'timestamp': ['2025-01-01 00:00:00', '2025-01-01 01:00:00'],
        'consumption_kwh': [0.5, 0.3],
        'temperature': [22, 21]
    })
    st.dataframe(example_data, hide_index=True)
    
    st.download_button(
        label="📥 Download Sample Template",
        data=example_data.to_csv(index=False),
        file_name="energy_data_template.csv",
        mime="text/csv"
    )
