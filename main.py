import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Streamlit Data Visualization Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4682B4;
        text-align: center;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1E90FF;
        border-bottom: 2px solid #F0F2F6;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    .insight-box {
        background-color: #F0F8FF;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #4682B4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Streamlit Data Visualization Demo</h1>", unsafe_allow_html=True)
st.markdown("This app demonstrates various data visualization capabilities of Streamlit.")

# Create sample datasets
@st.cache_data
def load_time_series_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
    values = np.random.normal(loc=100, scale=15, size=365).cumsum() + 1000
    seasonality = 20 * np.sin(np.linspace(0, 2*np.pi, 365) * 4)
    trend = np.linspace(0, 100, 365)
    data = pd.DataFrame({
        'date': dates,
        'value': values + seasonality + trend,
        'category': np.random.choice(['A', 'B', 'C'], size=365)
    })
    # Add some events
    events = pd.DataFrame({
        'date': pd.date_range(start="2023-02-15", end="2023-11-15", freq="2M"),
        'event_name': ['Launch', 'Update 1.0', 'Marketing', 'Update 2.0', 'Partnership'],
        'impact': [50, 30, 80, 25, 60]
    })
    return data, events

@st.cache_data
def load_categorical_data():
    categories = ['Food', 'Transport', 'Entertainment', 'Utilities', 'Rent', 'Shopping', 'Healthcare', 'Education']
    values = [350, 120, 180, 200, 800, 250, 120, 100]
    return pd.DataFrame({'Category': categories, 'Amount': values})

@st.cache_data
def load_geographic_data():
    # Sample data for US states
    states = ['California', 'Texas', 'Florida', 'New York', 'Pennsylvania', 'Illinois', 'Ohio', 'Georgia', 'North Carolina', 'Michigan']
    values = [np.random.randint(100, 1000) for _ in range(len(states))]
    return pd.DataFrame({'State': states, 'Value': values})

@st.cache_data
def load_correlation_data():
    np.random.seed(42)
    n = 500
    
    # Create correlated variables
    x1 = np.random.normal(0, 1, n)
    x2 = x1 * 0.8 + np.random.normal(0, 0.5, n)
    x3 = x2 * 0.7 + np.random.normal(0, 0.6, n)
    x4 = np.random.normal(0, 1, n)
    x5 = x4 * 0.5 + np.random.normal(0, 0.8, n)
    
    df = pd.DataFrame({
        'Feature A': x1,
        'Feature B': x2,
        'Feature C': x3,
        'Feature D': x4,
        'Feature E': x5,
        'Target': x1 * 2 + x2 * 1.5 + x4 * 0.5 + np.random.normal(0, 2, n)
    })
    
    return df

@st.cache_data
def create_customer_data():
    np.random.seed(42)
    n = 1000
    
    # Age distribution
    age = np.random.normal(35, 12, n).astype(int)
    age = np.clip(age, 18, 80)
    
    # Income based partly on age
    income_base = age * 1000 + np.random.normal(30000, 20000, n)
    income = np.clip(income_base, 15000, 250000)
    
    # Spending somewhat correlated with income
    spending = income * np.random.uniform(0.1, 0.4, n) + np.random.normal(0, 5000, n)
    spending = np.clip(spending, 1000, 100000)
    
    # Customer lifetime in months
    lifetime = np.random.exponential(24, n).astype(int) + 1
    lifetime = np.clip(lifetime, 1, 120)
    
    # Products owned (1-5)
    products = np.random.randint(1, 6, n)
    
    # Satisfaction score (1-10)
    satisfaction_base = 7 + np.random.normal(0, 2, n) + (products - 3) * 0.5
    satisfaction = np.clip(satisfaction_base.astype(int), 1, 10)
    
    # Region
    regions = ['North', 'South', 'East', 'West', 'Central']
    region = np.random.choice(regions, n)
    
    # Customer type
    types = ['New', 'Regular', 'Premium', 'VIP']
    weights = [0.3, 0.4, 0.2, 0.1]
    customer_type = np.random.choice(types, n, p=weights)
    
    # Create dataframe
    df = pd.DataFrame({
        'Age': age,
        'Income': income.astype(int),
        'Spending': spending.astype(int),
        'Lifetime_Months': lifetime,
        'Products': products,
        'Satisfaction': satisfaction,
        'Region': region,
        'Type': customer_type
    })
    
    return df

# Load all datasets
time_series_data, events_data = load_time_series_data()
category_data = load_categorical_data()
geo_data = load_geographic_data()
correlation_data = load_correlation_data()
customer_data = create_customer_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = [
    "Overview",
    "Interactive Time Series Analysis",
    "Categorical Visualizations",
    "Geographic Data",
    "Correlation & Relationships",
    "Customer Segmentation",
    "Custom Analysis"
]
selection = st.sidebar.radio("Go to", pages)

# Add data info to sidebar
with st.sidebar.expander("About the Data"):
    st.write("""
    This app uses synthetic datasets to demonstrate Streamlit's visualization capabilities:
    
    - Time series data (365 days)
    - Categorical spending data
    - US state sample data
    - Correlation dataset (500 points)
    - Customer dataset (1000 records)
    """)

# Add Streamlit info
with st.sidebar.expander("Streamlit Features Shown"):
    st.write("""
    - Basic charts: line, bar, scatter
    - Interactive widgets
    - Plotly integration
    - Altair integration
    - Matplotlib/Seaborn
    - Data filtering
    - Layout controls
    - Caching
    - Custom CSS
    - Metrics and KPIs
    - Expandable sections
    - File downloading
    """)

# Overview Page
if selection == "Overview":
    st.markdown("<h2 class='section-header'>Dashboard Overview</h2>", unsafe_allow_html=True)
    
    # Show metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Time Period", "365 days", "Complete Year")
    with col2:
        current_value = time_series_data['value'].iloc[-1]
        prev_value = time_series_data['value'].iloc[-30]
        delta = ((current_value - prev_value) / prev_value) * 100
        st.metric("Current Value", f"{current_value:.1f}", f"{delta:.1f}%")
    with col3:
        avg_spending = category_data['Amount'].mean()
        st.metric("Avg. Spending", f"${avg_spending:.2f}")
    with col4:
        customer_count = len(customer_data)
        st.metric("Customers", customer_count, "Synthetic Data")
    
    # High-level preview charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Time Series Preview")
        fig = px.line(time_series_data.iloc[-90:], x='date', y='value', 
                      title="Last 90 Days Trend")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Category Distribution")
        fig = px.pie(category_data, values='Amount', names='Category', 
                     title="Spending by Category")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer data preview
    st.subheader("Customer Data Preview")
    st.dataframe(customer_data.head())
    
    # Help section
    with st.expander("How to use this app"):
        st.write("""
        This demo app showcases various data visualization capabilities of Streamlit:
        
        1. Use the sidebar navigation to explore different types of visualizations
        2. Interact with the charts using the provided controls
        3. See how Streamlit can be used for data analysis and presentation
        
        Each section demonstrates different chart types and interactive features.
        """)

# Time Series Analysis Page
elif selection == "Interactive Time Series Analysis":
    st.markdown("<h2 class='section-header'>Interactive Time Series Analysis</h2>", unsafe_allow_html=True)
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", 
                                   value=time_series_data['date'].min().date(),
                                   min_value=time_series_data['date'].min().date(), 
                                   max_value=time_series_data['date'].max().date())
    with col2:
        end_date = st.date_input("End Date", 
                                value=time_series_data['date'].max().date(),
                                min_value=time_series_data['date'].min().date(), 
                                max_value=time_series_data['date'].max().date())
    
    # Filter data based on date range
    filtered_data = time_series_data[(time_series_data['date'].dt.date >= start_date) & 
                                    (time_series_data['date'].dt.date <= end_date)]
    
    # Category filter
    categories = time_series_data['category'].unique().tolist()
    selected_categories = st.multiselect("Select Categories", categories, default=categories)
    
    if selected_categories:
        filtered_data = filtered_data[filtered_data['category'].isin(selected_categories)]
    
    # Display metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average", f"{filtered_data['value'].mean():.2f}")
    with col2:
        st.metric("Maximum", f"{filtered_data['value'].max():.2f}")
    with col3:
        st.metric("Minimum", f"{filtered_data['value'].min():.2f}")
    with col4:
        start_val = filtered_data['value'].iloc[0]
        end_val = filtered_data['value'].iloc[-1]
        change_pct = ((end_val - start_val) / start_val) * 100
        st.metric("Period Change", f"{end_val - start_val:.2f}", f"{change_pct:.2f}%")
    
    # Time series visualization
    st.subheader("Time Series Visualization")
    
    chart_type = st.radio("Select Chart Type", ["Line", "Area", "Bar"], horizontal=True)
    
    if chart_type == "Line":
        fig = px.line(filtered_data, x='date', y='value', color='category',
                     title="Time Series Data", line_shape=st.selectbox("Line Shape", 
                                                                     ["linear", "spline", "hv", "vh", "hvh", "vhv"]))
        
    elif chart_type == "Area":
        fig = px.area(filtered_data, x='date', y='value', color='category',
                     title="Time Series Data")
        
    else:  # Bar
        fig = px.bar(filtered_data, x='date', y='value', color='category',
                    title="Time Series Data")
    
    show_events = st.checkbox("Show Key Events", value=True)
    
    if show_events:
        # Filter events based on date range
        filtered_events = events_data[(events_data['date'].dt.date >= start_date) & 
                                     (events_data['date'].dt.date <= end_date)]
        
        # Add events as scatter points
        if not filtered_events.empty:
            event_trace = go.Scatter(
                x=filtered_events['date'],
                y=[filtered_data['value'].max() * 0.95] * len(filtered_events),
                mode='markers+text',
                marker=dict(symbol='star', size=15, color='red'),
                text=filtered_events['event_name'],
                textposition="top center",
                name="Key Events"
            )
            fig.add_trace(event_trace)
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Moving average analysis
    st.subheader("Moving Average Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ma_window = st.slider("Moving Average Window", min_value=3, max_value=30, value=7)
    
    with col2:
        st.write("Apply moving average to smooth the data and identify trends.")
    
    # Calculate moving average
    agg_data = filtered_data.groupby('date')['value'].mean().reset_index()
    agg_data['MA'] = agg_data['value'].rolling(window=ma_window).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=agg_data['date'],
        y=agg_data['value'],
        mode='lines',
        name='Original',
        line=dict(color='royalblue')
    ))
    
    fig.add_trace(go.Scatter(
        x=agg_data['date'],
        y=agg_data['MA'],
        mode='lines',
        name=f'{ma_window}-day MA',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(title=f"Moving Average Analysis (Window: {ma_window} days)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonality and trend analysis with some simple decomposition
    if st.checkbox("Show Seasonality and Trend Analysis"):
        # Simple decomposition
        agg_data = filtered_data.groupby('date')['value'].mean().reset_index().set_index('date')
        
        # Ensure enough data points for analysis
        if len(agg_data) > 14:
            # Compute trend using moving average
            trend = agg_data['value'].rolling(window=14, center=True).mean()
            
            # Compute seasonality and residual
            seasonal_residual = agg_data['value'] - trend
            
            # Create dataframe for visualization
            decomp_df = pd.DataFrame({
                'Original': agg_data['value'],
                'Trend': trend,
                'Seasonal + Residual': seasonal_residual
            }).reset_index()
            
            # Create subplot figure
            fig = px.line(title="Time Series Decomposition")
            
            # Add original series
            fig.add_trace(go.Scatter(
                x=decomp_df['date'], 
                y=decomp_df['Original'],
                mode='lines', 
                name='Original',
                line=dict(color='blue')
            ))
            
            # Add trend
            fig.add_trace(go.Scatter(
                x=decomp_df['date'], 
                y=decomp_df['Trend'],
                mode='lines', 
                name='Trend',
                line=dict(color='red', width=2)
            ))
            
            # Add seasonal + residual
            fig.add_trace(go.Scatter(
                x=decomp_df['date'], 
                y=decomp_df['Seasonal + Residual'],
                mode='lines', 
                name='Seasonal + Residual',
                line=dict(color='green')
            ))
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<div class='insight-box'>Trend represents the long-term progression, while the Seasonal + Residual component captures cyclical patterns and random variations.</div>", unsafe_allow_html=True)
        else:
            st.warning("Need more data points for meaningful decomposition. Please select a wider date range.")

# Categorical Visualizations Page
elif selection == "Categorical Visualizations":
    st.markdown("<h2 class='section-header'>Categorical Data Visualizations</h2>", unsafe_allow_html=True)
    
    # Allow users to modify the data
    st.subheader("Spending by Category")
    
    edited_df = st.data_editor(
        category_data,
        column_config={
            "Category": st.column_config.TextColumn("Category"),
            "Amount": st.column_config.NumberColumn("Amount ($)", min_value=0, format="$%d")
        },
        use_container_width=True,
        num_rows="fixed",
    )
    
    st.write("Select visualization type:")
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.radio(
            "Chart Type",
            ["Bar Chart", "Pie Chart", "Radar Chart", "Treemap", "Funnel Chart"],
            horizontal=False
        )
    
    with col2:
        sort_option = st.radio(
            "Sort By",
            ["Category", "Amount (Ascending)", "Amount (Descending)"],
            horizontal=False
        )
        
        if sort_option == "Category":
            sorted_df = edited_df.sort_values("Category")
        elif sort_option == "Amount (Ascending)":
            sorted_df = edited_df.sort_values("Amount")
        else:  # "Amount (Descending)"
            sorted_df = edited_df.sort_values("Amount", ascending=False)
    
    # Display the selected chart
    if chart_type == "Bar Chart":
        # Bar chart options
        orientation = st.radio("Orientation", ["Vertical", "Horizontal"], horizontal=True)
        
        if orientation == "Vertical":
            fig = px.bar(
                sorted_df,
                x="Category",
                y="Amount",
                color="Category",
                title="Spending by Category",
                text_auto=True
            )
        else:
            fig = px.bar(
                sorted_df,
                y="Category",
                x="Amount",
                color="Category",
                title="Spending by Category",
                orientation='h',
                text_auto=True
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Pie Chart":
        # Pie chart options
        hole = st.slider("Donut Hole Size", min_value=0.0, max_value=0.8, value=0.0, step=0.1)
        
        fig = px.pie(
            sorted_df,
            values="Amount",
            names="Category",
            title="Spending Distribution",
            hole=hole
        )
        fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05] * len(sorted_df))
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Radar Chart":
        # Radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=sorted_df["Amount"],
            theta=sorted_df["Category"],
            fill='toself',
            name="Amount"
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )
            ),
            title="Spending Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Treemap":
        # Treemap chart
        fig = px.treemap(
            sorted_df,
            path=["Category"],
            values="Amount",
            color="Amount",
            color_continuous_scale='RdBu',
            title="Spending Treemap"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Funnel Chart
        # Sort for funnel chart
        funnel_df = sorted_df.sort_values("Amount", ascending=False)
        
        fig = go.Figure(go.Funnel(
            y=funnel_df["Category"],
            x=funnel_df["Amount"],
            textinfo="value+percent initial"
        ))
        
        fig.update_layout(title="Spending Funnel")
        st.plotly_chart(fig, use_container_width=True)
    
    # Basic statistics
    st.subheader("Basic Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Spending", f"${edited_df['Amount'].sum()}")
    
    with col2:
        st.metric("Average per Category", f"${edited_df['Amount'].mean():.2f}")
    
    with col3:
        st.metric("Categories", f"{len(edited_df)}")
    
    # Pareto analysis
    if st.checkbox("Show Pareto Analysis"):
        st.subheader("Pareto Analysis (80/20 Rule)")
        
        # Sort by amount descending
        pareto_df = edited_df.sort_values("Amount", ascending=False).reset_index(drop=True)
        
        # Calculate cumulative percentage
        pareto_df["Cumulative"] = pareto_df["Amount"].cumsum()
        pareto_df["Cumulative Percentage"] = 100 * pareto_df["Cumulative"] / pareto_df["Amount"].sum()
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add bar chart for amounts
        fig.add_trace(go.Bar(
            x=pareto_df["Category"],
            y=pareto_df["Amount"],
            name="Amount",
            marker_color='royalblue'
        ))
        
        # Add line chart for cumulative percentage
        fig.add_trace(go.Scatter(
            x=pareto_df["Category"],
            y=pareto_df["Cumulative Percentage"],
            name="Cumulative %",
            marker_color='red',
            yaxis='y2'
        ))
        
        # Add 80% reference line
        fig.add_shape(
            type="line",
            x0=pareto_df["Category"].iloc[0],
            y0=80,
            x1=pareto_df["Category"].iloc[-1],
            y1=80,
            line=dict(
                color="green",
                width=2,
                dash="dash",
            ),
            yref='y2'
        )
        
        # Update layout
        fig.update_layout(
            title="Pareto Analysis: Which categories account for most spending?",
            yaxis=dict(title="Amount ($)"),
            yaxis2=dict(
                title="Cumulative %",
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pareto insights
        eighty_percent_categories = pareto_df[pareto_df["Cumulative Percentage"] <= 80]
        st.markdown(f"""
        <div class='insight-box'>
        <strong>Pareto Principle (80/20 Rule):</strong> {len(eighty_percent_categories)} out of {len(pareto_df)} categories 
        ({(len(eighty_percent_categories)/len(pareto_df)*100):.1f}% of categories) account for approximately 80% of total spending.
        </div>
        """, unsafe_allow_html=True)
    
    # Download data option
    csv = edited_df.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="category_spending.csv",
        mime="text/csv",
    )

# Geographic Data Page  
elif selection == "Geographic Data":
    st.markdown("<h2 class='section-header'>Geographic Data Visualization</h2>", unsafe_allow_html=True)
    
    # Allow editing the geographic data
    st.subheader("Sample US State Data")
    
    edited_geo_df = st.data_editor(
        geo_data,
        column_config={
            "State": st.column_config.TextColumn("State"),
            "Value": st.column_config.NumberColumn("Value", min_value=0, format="%d")
        },
        use_container_width=True,
        num_rows="fixed",
    )
    
    # Choropleth map using Plotly
    st.subheader("US States Choropleth Map")
    
    fig = px.choropleth(
        edited_geo_df,
        locations="State",
        locationmode="USA-states",
        color="Value",
        scope="usa",
        color_continuous_scale="Viridis",
        title="Value by US State"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Bar chart comparison
    st.subheader("State Comparison")
    
    sort_by = st.radio("Sort by", ["State Name", "Value (Ascending)", "Value (Descending)"], horizontal=True)
    
    if sort_by == "State Name":
        sorted_geo_df = edited_geo_df.sort_values("State")
    elif sort_by == "Value (Ascending)":
        sorted_geo_df = edited_geo_df.sort_values("Value")
    else:  # "Value (Descending)"
        sorted_geo_df = edited_geo_df.sort_values("Value", ascending=False)
    
    fig = px.bar(
        sorted_geo_df,
        x="State",
        y="Value",
        color="Value",
        title="Value by State",
        text_auto=True,
        color_continuous_scale="Viridis"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Custom region analysis
    st.subheader("Custom Region Analysis")
    
    # Define regions
    regions = {
        "Northeast": ["New York", "Pennsylvania"],
        "South": ["Florida", "Georgia", "North Carolina"],
        "Midwest": ["Illinois", "Ohio", "Michigan"],
        "West": ["California", "Texas"]
    }
    
    # Let user modify the regions
    st.write("Modify Region Definitions:")
    
    updated_regions = {}
    col1, col2 = st.columns(2)
    
    with col1:
        for i, (region, states) in enumerate(list(regions.items())[:2]):
            updated_regions[region] = st.multiselect(
                f"States in {region}",
                options=geo_data["State"].tolist(),
                default=states
            )
    
    with col2:
        for i, (region, states) in enumerate(list(regions.items())[2:]):
            updated_regions[region] = st.multiselect(
                f"States in {region}",
                options=geo_data["State"].tolist(),
                default=states
            )
    
    # Create region aggregated data
    region_data = []
    
    for region, states in updated_regions.items():
        region_value = edited_geo_df[edited_geo_df["State"].isin(states)]["Value"].sum()
        region_data.append({"Region": region, "Total Value": region_value})
    
    region_df = pd.DataFrame(region_data)
    
    # Display region chart
    if len(region_df) > 0:
        fig = px.pie(
            region_df,
            values="Total Value",
            names="Region",
            title="Regional Distribution",
            hole=0.4
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No regions defined. Please add states to regions.")

# Correlation and Relationships Page
elif selection == "Correlation & Relationships":
    st.markdown("<h2 class='section-header'>Data Correlation & Relationships</h2>", unsafe_allow_html=True)
    
    # Show dataset info
    st.write("This section explores relationships between variables in the correlation dataset.")
    
    # Show data preview
    with st.expander("Show Data Preview"):
        st.dataframe(correlation_data.head())
    
    # Feature selection for correlation
    st.subheader("Select Features for Analysis")
    
    all_features = correlation_data.columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("X-axis Feature", all_features, index=0)
    
    with col2:
        y_feature = st.selectbox("Y-axis Feature", all_features, index=5)  # Default to Target
    
    # Scatter plot with regression line
    st.subheader("Relationship Analysis")
    
    plot_type = st.radio("Plot Type", ["Scatter", "Hexbin", "Density Contour"], horizontal=True)
    
    if plot_type == "Scatter":
        add_trendline = st.checkbox("Add Trendline", value=True)
        
        if add_trendline:
            fig = px.scatter(
                correlation_data,
                x=x_feature,
                y=y_feature,
                trendline="ols",
                title=f"Relationship between {x_feature} and {y_feature}",
                opacity=0.7
            )
            
            # Calculate correlation
            corr = correlation_data[x_feature].corr(correlation_data[y_feature])
            st.write(f"Pearson Correlation: **{corr:.3f}**")
            
            # Add correlation interpretation
            if abs(corr) < 0.3:
                corr_strength = "weak"
            elif abs(corr) < 0.7:
                corr_strength = "moderate"
            else:
                corr_strength = "strong"
                
            corr_direction = "positive" if corr > 0 else "negative"
            
            st.markdown(f"""
            <div class='insight-box'>
            There is a {corr_strength} {corr_direction} correlation between {x_feature} and {y_feature}.
            </div>
            """, unsafe_allow_html=True)
        else:
            fig = px.scatter(
                correlation_data,
                x=x_feature,
                y=y_feature,
                title=f"Relationship between {x_feature} and {y_feature}",
                opacity=0.7
            )
    
    elif plot_type == "Hexbin":
        fig = px.density_heatmap(
            correlation_data,
            x=x_feature,
            y=y_feature,
            title=f"Hexbin Density: {x_feature} vs {y_feature}",
            nbinsx=20,
            nbinsy=20,
            color_continuous_scale="Viridis"
        )
    
    else:  # Density Contour
        fig = px.density_contour(
            correlation_data,
            x=x_feature,
            y=y_feature,
            title=f"Density Contour: {x_feature} vs {y_feature}"
        )
        
        fig.update_traces(contours_coloring="fill", contours_showlabels=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix heatmap
    st.subheader("Correlation Matrix")
    
    # Select features for correlation matrix
    selected_features = st.multiselect(
        "Select Features for Correlation Matrix",
        all_features,
        default=all_features
    )
    
    if len(selected_features) > 1:
        corr_matrix = correlation_data[selected_features].corr()
        
        # Plot using Seaborn through Matplotlib
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        cmap = st.radio("Colormap", ["coolwarm", "viridis", "plasma"], horizontal=True)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .5},
            annot=True,
            fmt=".2f",
            ax=ax
        )
        
        plt.title("Correlation Matrix")
        st.pyplot(fig)
        
        # Highest correlations
        corr_pairs = []
        
        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                feature1 = selected_features[i]
                feature2 = selected_features[j]
                correlation = corr_matrix.loc[feature1, feature2]
                corr_pairs.append((feature1, feature2, correlation))
        
        # Sort by absolute correlation value
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        st.subheader("Top Correlations")
        
        if len(corr_pairs) > 0:
            top_n = min(5, len(corr_pairs))
            
            for i in range(top_n):
                feature1, feature2, corr = corr_pairs[i]
                st.write(f"{i+1}. **{feature1}** and **{feature2}**: {corr:.3f}")
    else:
        st.warning("Please select at least two features for the correlation matrix.")
    
    # Pairplot with Plotly
    if st.checkbox("Show Interactive Pairplot"):
        # Limit to 5 features for performance reasons
        if len(selected_features) > 5:
            st.warning("Limiting pairplot to first 5 selected features for better performance.")
            pairplot_features = selected_features[:5]
        else:
            pairplot_features = selected_features
        
        if len(pairplot_features) > 1:
            fig = px.scatter_matrix(
                correlation_data,
                dimensions=pairplot_features,
                title="Interactive Pairplot"
            )
            
            fig.update_traces(diagonal_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least two features for the pairplot.")

# Customer Segmentation Page
elif selection == "Customer Segmentation":
    st.markdown("<h2 class='section-header'>Customer Segmentation Analysis</h2>", unsafe_allow_html=True)
    
    # Show data preview
    st.subheader("Customer Data Preview")
    
    with st.expander("Show Raw Data"):
        st.dataframe(customer_data.head(10))
    
    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(customer_data):,}")
    with col2:
        st.metric("Avg. Age", f"{customer_data['Age'].mean():.1f} years")
    with col3:
        st.metric("Avg. Income", f"${customer_data['Income'].mean():,.0f}")
    with col4:
        st.metric("Avg. Satisfaction", f"{customer_data['Satisfaction'].mean():.1f}/10")
    
    # Choose segmentation variables
    st.subheader("Customer Segmentation")
    
    segmentation_option = st.selectbox(
        "Select Segmentation Approach",
        ["Age vs. Income", "Spending vs. Lifetime", "Products vs. Satisfaction", "Region Analysis", "Customer Type Analysis"]
    )
    
    if segmentation_option == "Age vs. Income":
        # Age brackets
        age_bins = [18, 25, 35, 50, 65, 100]
        age_labels = ['18-24', '25-34', '35-49', '50-64', '65+']
        
        customer_data['Age Bracket'] = pd.cut(customer_data['Age'], bins=age_bins, labels=age_labels)
        
        # Income brackets
        income_bins = [0, 30000, 60000, 100000, 150000, 1000000]
        income_labels = ['<30K', '30K-60K', '60K-100K', '100K-150K', '150K+']
        
        customer_data['Income Bracket'] = pd.cut(customer_data['Income'], bins=income_bins, labels=income_labels)
        
        # Create heatmap data
        heatmap_data = pd.crosstab(
            customer_data['Age Bracket'], 
            customer_data['Income Bracket'],
            normalize='all'
        ).round(3) * 100
        
        # Plot heatmap
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Income Bracket", y="Age Bracket", color="% of Customers"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis"
        )
        
        fig.update_layout(title="Customer Distribution by Age and Income (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show average spending by segment
        pivot_data = customer_data.pivot_table(
            index='Age Bracket',
            columns='Income Bracket',
            values='Spending',
            aggfunc='mean'
        ).round(0)
        
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Income Bracket", y="Age Bracket", color="Avg. Spending ($)"),
            x=pivot_data.columns,
            y=pivot_data.index,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )
        
        fig.update_layout(title="Average Spending by Age and Income Segment ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show insights
        st.markdown("<div class='insight-box'>The heatmap reveals which age and income segments have the highest concentration of customers and their spending behaviors.</div>", unsafe_allow_html=True)
    
    elif segmentation_option == "Spending vs. Lifetime":
        # Create scatter plot
        fig = px.scatter(
            customer_data,
            x="Lifetime_Months",
            y="Spending",
            color="Type",
            size="Income",
            hover_data=["Age", "Products", "Satisfaction"],
            title="Customer Spending vs. Lifetime",
            labels={"Lifetime_Months": "Customer Lifetime (Months)", "Spending": "Annual Spending ($)"}
        )
        
        # Add quadrant lines
        avg_lifetime = customer_data["Lifetime_Months"].median()
        avg_spending = customer_data["Spending"].median()
        
        fig.add_vline(x=avg_lifetime, line_dash="dash", line_color="gray")
        fig.add_hline(y=avg_spending, line_dash="dash", line_color="gray")
        
        # Add quadrant annotations
        fig.add_annotation(x=avg_lifetime/2, y=avg_spending*1.8, text="Short-term High Value", showarrow=False)
        fig.add_annotation(x=avg_lifetime*1.8, y=avg_spending*1.8, text="Long-term High Value", showarrow=False)
        fig.add_annotation(x=avg_lifetime/2, y=avg_spending*0.2, text="Short-term Low Value", showarrow=False)
        fig.add_annotation(x=avg_lifetime*1.8, y=avg_spending*0.2, text="Long-term Low Value", showarrow=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate customer lifetime value (simplified)
        st.subheader("Customer Lifetime Value Analysis")
        
        customer_data['Monthly Value'] = customer_data['Spending'] / 12
        customer_data['CLV'] = customer_data['Monthly Value'] * customer_data['Lifetime_Months']
        
        # Group by type
        clv_by_type = customer_data.groupby('Type')['CLV'].mean().sort_values(ascending=False).reset_index()
        
        fig = px.bar(
            clv_by_type,
            x='Type',
            y='CLV',
            color='Type',
            title="Average Customer Lifetime Value by Type",
            text_auto=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif segmentation_option == "Products vs. Satisfaction":
        # Create frequency table
        product_sat = pd.crosstab(
            customer_data['Products'],
            customer_data['Satisfaction'],
            normalize='index'
        ).round(3) * 100
        
        # Plot heatmap
        fig = px.imshow(
            product_sat,
            labels=dict(x="Satisfaction Score", y="Number of Products", color="% of Customers"),
            x=product_sat.columns,
            y=product_sat.index,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdYlGn"
        )
        
        fig.update_layout(title="Customer Satisfaction Distribution by Number of Products (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate average metrics by product count
        product_metrics = customer_data.groupby('Products').agg({
            'Spending': 'mean',
            'Income': 'mean',
            'Lifetime_Months': 'mean',
            'Satisfaction': 'mean'
        }).reset_index()
        
        # Multi-line chart
        metrics_to_plot = st.multiselect(
            "Select Metrics to Plot",
            ["Spending", "Income", "Lifetime_Months", "Satisfaction"],
            default=["Spending", "Satisfaction"]
        )
        
        if metrics_to_plot:
            fig = go.Figure()
            
            for metric in metrics_to_plot:
                # Scale values to make them comparable
                if metric == "Spending" or metric == "Income":
                    scaled_values = product_metrics[metric] / product_metrics[metric].max() * 100
                    y_title = "Scaled Value (% of Maximum)"
                    hover_template = f"{metric}: " + "${%{text:.0f}}<extra></extra>"
                elif metric == "Lifetime_Months":
                    scaled_values = product_metrics[metric] / product_metrics[metric].max() * 100
                    y_title = "Scaled Value (% of Maximum)"
                    hover_template = f"{metric}: " + "%{text:.1f} months<extra></extra>"
                else:  # Satisfaction
                    scaled_values = product_metrics[metric] / 10 * 100  # Scale to percentage of max possible (10)
                    y_title = "Scaled Value (% of Maximum)"
                    hover_template = f"{metric}: " + "%{text:.1f}/10<extra></extra>"
                
                fig.add_trace(go.Scatter(
                    x=product_metrics['Products'],
                    y=scaled_values,
                    mode='lines+markers',
                    name=metric,
                    text=product_metrics[metric],
                    hovertemplate=hover_template
                ))
            
            fig.update_layout(
                title="Metrics by Number of Products (Scaled)",
                xaxis_title="Number of Products",
                yaxis_title=y_title,
                xaxis=dict(tickmode='linear')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show raw metrics
            with st.expander("Show Raw Metrics by Product Count"):
                st.dataframe(product_metrics.round(1))
    
    elif segmentation_option == "Region Analysis":
        st.subheader("Customer Analysis by Region")
        
        # Customer count by region
        region_counts = customer_data['Region'].value_counts().reset_index()
        region_counts.columns = ['Region', 'Customer Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                region_counts,
                values='Customer Count',
                names='Region',
                title="Customer Distribution by Region"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Calculate metrics by region
            region_metrics = customer_data.groupby('Region').agg({
                'Spending': 'mean',
                'Income': 'mean',
                'Satisfaction': 'mean'
            }).reset_index()
            
            # Create metrics visualization
            selected_metric = st.selectbox("Select Metric", ['Spending', 'Income', 'Satisfaction'])
            
            fig = px.bar(
                region_metrics,
                x='Region',
                y=selected_metric,
                color='Region',
                title=f"Average {selected_metric} by Region",
                text_auto=True
            )
            
            if selected_metric in ['Spending', 'Income']:
                fig.update_layout(yaxis_tickprefix="$")
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer type distribution by region
        type_region = pd.crosstab(
            customer_data['Region'],
            customer_data['Type'],
            normalize='index'
        ).round(3) * 100
        
        fig = px.bar(
            type_region.reset_index().melt(id_vars='Region', var_name='Type', value_name='Percentage'),
            x='Region',
            y='Percentage',
            color='Type',
            title="Customer Type Distribution by Region (%)",
            barmode='stack',
            text_auto=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Customer Type Analysis
        st.subheader("Customer Analysis by Type")
        
        # Customer count by type
        type_counts = customer_data['Type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Customer Count']
        
        # Calculate percentage
        type_counts['Percentage'] = (type_counts['Customer Count'] / type_counts['Customer Count'].sum() * 100).round(1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                type_counts,
                values='Customer Count',
                names='Type',
                title="Customer Distribution by Type",
                hover_data=['Percentage'],
                labels={'Percentage': 'Percentage (%)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average metrics by type
            type_metrics = customer_data.groupby('Type').agg({
                'Age': 'mean',
                'Income': 'mean',
                'Spending': 'mean',
                'Lifetime_Months': 'mean',
                'Products': 'mean',
                'Satisfaction': 'mean'
            }).reset_index().round(1)
            
            # Create metrics table
            st.dataframe(
                type_metrics,
                column_config={
                    "Type": st.column_config.TextColumn("Customer Type"),
                    "Age": st.column_config.NumberColumn("Avg. Age"),
                    "Income": st.column_config.NumberColumn("Avg. Income", format="$%d"),
                    "Spending": st.column_config.NumberColumn("Avg. Spending", format="$%d"),
                    "Lifetime_Months": st.column_config.NumberColumn("Avg. Lifetime (mo)"),
                    "Products": st.column_config.NumberColumn("Avg. Products"),
                    "Satisfaction": st.column_config.NumberColumn("Satisfaction")
                },
                use_container_width=True
            )
        
        # Radar chart for customer type profiles
        st.subheader("Customer Type Profile Comparison")
        
        # Select metrics and scale them
        radar_metrics = ['Age', 'Income', 'Spending', 'Lifetime_Months', 'Products', 'Satisfaction']
        radar_df = customer_data.groupby('Type')[radar_metrics].mean().reset_index()
        
        # Scale each metric from 0-1 for radar chart
        for metric in radar_metrics:
            max_val = customer_data[metric].max()
            radar_df[f"{metric}_scaled"] = radar_df[metric] / max_val
        
        # Create radar chart
        fig = go.Figure()
        
        for i, cust_type in enumerate(radar_df['Type']):
            fig.add_trace(go.Scatterpolar(
                r=radar_df.loc[i, [f"{m}_scaled" for m in radar_metrics]],
                theta=radar_metrics,
                fill='toself',
                name=cust_type
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Customer Type Profiles (Normalized)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Type migration analysis
        st.subheader("Type Migration Analysis")
        
        st.markdown("""
        <div class='insight-box'>
        <strong>Customer Type Progression</strong><br>
        The typical customer progression follows this path: New → Regular → Premium → VIP<br>
        Understanding what drives this progression can help with customer development strategies.
        </div>
        """, unsafe_allow_html=True)
        
        # Show conversion analysis (simplified)
        conversion_rates = {
            "New to Regular": 25,
            "Regular to Premium": 15,
            "Premium to VIP": 8
        }
        
        # Create funnel chart
        stages = ["New Customers (100%)", "Regular Customers (25%)", "Premium Customers (15%)", "VIP Customers (8%)"]
        values = [100, 25, 15, 8]
        
        fig = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textinfo="value+percent initial"
        ))
        
        fig.update_layout(title="Customer Progression Funnel (Example)")
        st.plotly_chart(fig, use_container_width=True)

# Custom Analysis Page  
elif selection == "Custom Analysis":
    st.markdown("<h2 class='section-header'>Custom Data Analysis</h2>", unsafe_allow_html=True)
    
    st.write("In this section, you can perform custom analyses by selecting variables from different datasets.")
    
    # Dataset selection
    dataset_option = st.selectbox(
        "Select Dataset",
        ["Time Series Data", "Category Data", "Customer Data", "Correlation Data"]
    )
    
    if dataset_option == "Time Series Data":
        # Show data preview
        with st.expander("Show Data Preview"):
            st.dataframe(time_series_data.head())
        
        # Time aggregation
        st.subheader("Time Aggregation")
        
        time_unit = st.radio(
            "Select Time Unit for Aggregation",
            ["Day", "Week", "Month", "Quarter"],
            horizontal=True
        )
        
        if time_unit == "Day":
            # Already at day level
            agg_data = time_series_data.copy()
            agg_data['time_period'] = agg_data['date']
        elif time_unit == "Week":
            agg_data = time_series_data.copy()
            agg_data['time_period'] = agg_data['date'].dt.to_period('W').dt.start_time
        elif time_unit == "Month":
            agg_data = time_series_data.copy()
            agg_data['time_period'] = agg_data['date'].dt.to_period('M').dt.start_time
        else:  # Quarter
            agg_data = time_series_data.copy()
            agg_data['time_period'] = agg_data['date'].dt.to_period('Q').dt.start_time
        
        # Aggregate data
        agg_function = st.selectbox(
            "Select Aggregation Function",
            ["Mean", "Sum", "Min", "Max", "Median", "Count"]
        )
        
        agg_map = {
            "Mean": "mean",
            "Sum": "sum",
            "Min": "min",
            "Max": "max",
            "Median": "median",
            "Count": "count"
        }
        
        grouped_data = agg_data.groupby(['time_period', 'category'])[['value']].agg(agg_map[agg_function]).reset_index()
        
        # Visualization
        st.subheader(f"{time_unit}ly {agg_function} Values")
        
        chart_type = st.radio(
            "Select Chart Type",
            ["Line", "Bar", "Area"],
            horizontal=True
        )
        
        if chart_type == "Line":
            fig = px.line(
                grouped_data,
                x='time_period',
                y='value',
                color='category',
                title=f"{time_unit}ly {agg_function} Values by Category"
            )
        elif chart_type == "Bar":
            fig = px.bar(
                grouped_data,
                x='time_period',
                y='value',
                color='category',
                title=f"{time_unit}ly {agg_function} Values by Category"
            )
        else:  # Area
            fig = px.area(
                grouped_data,
                x='time_period',
                y='value',
                color='category',
                title=f"{time_unit}ly {agg_function} Values by Category"
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Rolling window analysis
        st.subheader("Rolling Window Analysis")
        
        window_size = st.slider("Select Rolling Window Size", 3, 30, 7)
        
        # Prepare data for rolling analysis
        if time_unit == "Day":
            # Calculate for entire dataset, then group
            overall_avg = time_series_data.groupby('date')['value'].mean().reset_index()
            overall_avg = overall_avg.sort_values('date')
            overall_avg['rolling_avg'] = overall_avg['value'].rolling(window=window_size).mean()
            
            fig = px.line(
                overall_avg,
                x='date',
                y=['value', 'rolling_avg'],
                title=f"Original vs {window_size}-{time_unit} Rolling Average",
                labels={'value': 'Original Value', 'rolling_avg': f'{window_size}-{time_unit} Rolling Avg'}
            )
            
            fig.update_layout(legend_title_text='Series')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Rolling analysis is only available for day-level data when using {time_unit} aggregation.")
    
    elif dataset_option == "Category Data":
        # Show data preview
        with st.expander("Show Data Preview"):
            st.dataframe(category_data)
        
        # Allow custom sorting
        sort_by = st.radio(
            "Sort By",
            ["Category (A-Z)", "Amount (Low to High)", "Amount (High to Low)"],
            horizontal=True
        )
        
        if sort_by == "Category (A-Z)":
            sorted_data = category_data.sort_values("Category")
        elif sort_by == "Amount (Low to High)":
            sorted_data = category_data.sort_values("Amount")
        else:  # "Amount (High to Low)"
            sorted_data = category_data.sort_values("Amount", ascending=False)
        
        # Visualization type
        vis_type = st.radio(
            "Visualization Type",
            ["Bar Chart", "Pie Chart", "Pareto Chart"],
            horizontal=True
        )
        
        if vis_type == "Bar Chart":
            # Bar orientation
            orientation = st.radio("Bar Orientation", ["Vertical", "Horizontal"], horizontal=True)
            
            if orientation == "Vertical":
                fig = px.bar(
                    sorted_data,
                    x="Category",
                    y="Amount",
                    color="Category",
                    title="Amount by Category",
                    text_auto=True
                )
            else:
                fig = px.bar(
                    sorted_data,
                    y="Category",
                    x="Amount",
                    color="Category",
                    title="Amount by Category",
                    orientation='h',
                    text_auto=True
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif vis_type == "Pie Chart":
            # Pie chart options
            show_values = st.checkbox("Show Values", value=True)
            
            textinfo = "percent"
            if show_values:
                textinfo += "+value"
            
            fig = px.pie(
                sorted_data,
                values="Amount",
                names="Category",
                title="Amount Distribution by Category"
            )
            
            fig.update_traces(textinfo=textinfo)
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # Pareto Chart
            # Sort descending for Pareto
            pareto_data = category_data.sort_values("Amount", ascending=False).reset_index(drop=True)
            
            # Calculate cumulative percentage
            pareto_data["Cumulative"] = pareto_data["Amount"].cumsum()
            pareto_data["Cumulative Percentage"] = 100 * pareto_data["Cumulative"] / pareto_data["Amount"].sum()
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Add bar chart for amounts
            fig.add_trace(go.Bar(
                x=pareto_data["Category"],
                y=pareto_data["Amount"],
                name="Amount",
                marker_color='royalblue'
            ))
            
            # Add line chart for cumulative percentage
            fig.add_trace(go.Scatter(
                x=pareto_data["Category"],
                y=pareto_data["Cumulative Percentage"],
                name="Cumulative %",
                marker_color='red',
                yaxis='y2'
            ))
            
            # Add 80% reference line
            fig.add_shape(
                type="line",
                x0=pareto_data["Category"].iloc[0],
                y0=80,
                x1=pareto_data["Category"].iloc[-1],
                y1=80,
                line=dict(
                    color="green",
                    width=2,
                    dash="dash",
                ),
                yref='y2'
            )
            
            # Update layout
            fig.update_layout(
                title="Pareto Analysis (80/20 Rule)",
                yaxis=dict(title="Amount ($)"),
                yaxis2=dict(
                    title="Cumulative %",
                    overlaying="y",
                    side="right",
                    range=[0, 100]
                ),
                legend=dict(x=0.01, y=0.99),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Threshold analysis
            threshold = st.slider("Select cumulative percentage threshold", 0, 100, 80)
            
            # Find categories that make up the threshold
            threshold_categories = pareto_data[pareto_data["Cumulative Percentage"] <= threshold]
            
            if not threshold_categories.empty:
                st.write(f"**{len(threshold_categories)}** out of **{len(pareto_data)}** categories "
                         f"({len(threshold_categories)/len(pareto_data)*100:.1f}%) account for {threshold}% of the total amount.")
            else:
                st.write("No categories found below the threshold.")
    
    elif dataset_option == "Customer Data":
        # Show data preview
        with st.expander("Show Data Preview"):
            st.dataframe(customer_data.head())
        
        # Feature selection
        st.subheader("Feature Selection")
        
        numeric_cols = ['Age', 'Income', 'Spending', 'Lifetime_Months', 'Products', 'Satisfaction']
        categorical_cols = ['Region', 'Type']
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox(
                "Select X-axis Feature",
                numeric_cols + categorical_cols
            )
        
        with col2:
            y_axis = st.selectbox(
                "Select Y-axis Feature",
                numeric_cols,
                index=2  # Default to Spending
            )
        
        # Optional color feature
        color_by = st.selectbox(
            "Color by",
            ["None"] + categorical_cols,
            index=2  # Default to Type
        )
        
        if color_by == "None":
            color_column = None
        else:
            color_column = color_by
        
        # Chart type selection
        chart_type = st.radio(
            "Select Chart Type",
            ["Scatter", "Box Plot", "Violin Plot", "Bar Chart"],
            horizontal=True
        )
        
        # Create visualization based on selections
        if chart_type == "Scatter":
            if x_axis in numeric_cols:
                # Both numeric - create scatter plot
                fig = px.scatter(
                    customer_data,
                    x=x_axis,
                    y=y_axis,
                    color=color_column,
                    title=f"{y_axis} vs {x_axis}",
                    opacity=0.7,
                    trendline="ols" if st.checkbox("Add Trendline", value=False) else None
                )
                
                # Add axis labels with units
                unit_map = {
                    'Age': 'years',
                    'Income': '$',
                    'Spending': '$',
                    'Lifetime_Months': 'months',
                    'Products': 'count',
                    'Satisfaction': 'score'
                }
                
                x_unit = unit_map.get(x_axis, '')
                y_unit = unit_map.get(y_axis, '')
                
                if x_unit == '$':
                    fig.update_xaxes(title=f"{x_axis} ({x_unit})", tickprefix=x_unit)
                else:
                    fig.update_xaxes(title=f"{x_axis} ({x_unit})")
                
                if y_unit == '$':
                    fig.update_yaxes(title=f"{y_axis} ({y_unit})", tickprefix=y_unit)
                else:
                    fig.update_yaxes(title=f"{y_axis} ({y_unit})")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation if both numeric
                corr = customer_data[x_axis].corr(customer_data[y_axis])
                st.write(f"Correlation coefficient: **{corr:.3f}**")
            else:
                # X is categorical
                fig = px.box(
                    customer_data,
                    x=x_axis,
                    y=y_axis,
                    color=color_column,
                    title=f"{y_axis} by {x_axis}",
                    points="all" if st.checkbox("Show All Points", value=False) else False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Box Plot":
            if x_axis in categorical_cols:
                # X is categorical - create box plot
                fig = px.box(
                    customer_data,
                    x=x_axis,
                    y=y_axis,
                    color=color_column,
                    title=f"{y_axis} Distribution by {x_axis}",
                    points="all" if st.checkbox("Show All Points", value=False) else False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # X is numeric - need to bin it
                st.write(f"Binning {x_axis} for box plot...")
                
                # Create bins
                num_bins = st.slider("Number of Bins", 2, 10, 5)
                
                # Calculate bins and labels
                min_val = customer_data[x_axis].min()
                max_val = customer_data[x_axis].max()
                bin_width = (max_val - min_val) / num_bins
                
                bins = [min_val + i * bin_width for i in range(num_bins + 1)]
                labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(num_bins)]
                
                # Create binned column
                binned_data = customer_data.copy()
                binned_data[f"{x_axis} Binned"] = pd.cut(
                    binned_data[x_axis],
                    bins=bins,
                    labels=labels
                )
                
                # Create box plot
                fig = px.box(
                    binned_data,
                    x=f"{x_axis} Binned",
                    y=y_axis,
                    color=color_column,
                    title=f"{y_axis} Distribution by {x_axis} (Binned)",
                    category_orders={f"{x_axis} Binned": labels}
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Violin Plot":
            if x_axis in categorical_cols:
                # X is categorical - create violin plot
                fig = px.violin(
                    customer_data,
                    x=x_axis,
                    y=y_axis,
                    color=color_column,
                    title=f"{y_axis} Distribution by {x_axis}",
                    box=True,
                    points="all" if st.checkbox("Show All Points", value=False) else False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Violin plots work best with categorical X-axis. Please select a categorical feature for X-axis.")
        
        else:  # Bar Chart
            if x_axis in categorical_cols:
                # Aggregate data
                agg_function = st.selectbox(
                    "Aggregation Function",
                    ["Mean", "Median", "Sum", "Count", "Min", "Max"]
                )
                
                agg_map = {
                    "Mean": "mean",
                    "Median": "median",
                    "Sum": "sum",
                    "Count": "count",
                    "Min": "min",
                    "Max": "max"
                }
                
                # Create aggregation
                agg_data = customer_data.groupby(x_axis)[y_axis].agg(agg_map[agg_function]).reset_index()
                
                # Create bar chart
                fig = px.bar(
                    agg_data,
                    x=x_axis,
                    y=y_axis,
                    color=x_axis,
                    title=f"{agg_function} {y_axis} by {x_axis}",
                    text_auto=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Bar charts work best with categorical X-axis. Please select a categorical feature for X-axis.")
        
        # Advanced analysis section
        st.subheader("Advanced Analysis")
        
        advanced_analysis = st.selectbox(
            "Select Analysis Type",
            ["None", "Group Comparison", "Segmentation Analysis", "Outlier Detection"]
        )
        
        if advanced_analysis == "Group Comparison":
            # Select grouping variable
            group_var = st.selectbox(
                "Group By",
                categorical_cols,
                index=1  # Default to Type
            )
            
            # Select metric to compare
            compare_var = st.selectbox(
                "Compare",
                numeric_cols,
                index=2  # Default to Spending
            )
            
            # Calculate group statistics
            group_stats = customer_data.groupby(group_var)[compare_var].agg(['mean', 'median', 'std', 'count']).reset_index()
            group_stats.columns = [group_var, 'Mean', 'Median', 'Std Dev', 'Count']
            
            # Round values
            for col in ['Mean', 'Median', 'Std Dev']:
                group_stats[col] = group_stats[col].round(2)
            
            # Display stats
            st.write("Group Statistics:")
            st.dataframe(group_stats, use_container_width=True)
            
            # ANOVA-like visualization
            fig = go.Figure()
            
            for group in customer_data[group_var].unique():
                subset = customer_data[customer_data[group_var] == group][compare_var]
                fig.add_trace(go.Box(
                    y=subset,
                    name=group,
                    boxmean=True  # adds mean as dashed line
                ))
            
            fig.update_layout(
                title=f"Distribution of {compare_var} by {group_var}",
                yaxis_title=compare_var
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # T-test like insight
            st.markdown("""
            <div class='insight-box'>
            Group comparison allows you to see if there are significant differences between categories.
            Look for non-overlapping box plots to identify potential significant differences.
            </div>
            """, unsafe_allow_html=True)
        
        elif advanced_analysis == "Segmentation Analysis":
            # Select two variables to create a 2D segmentation
            seg_var1 = st.selectbox(
                "Segmentation Variable 1",
                numeric_cols,
                index=1  # Default to Income
            )
            
            seg_var2 = st.selectbox(
                "Segmentation Variable 2",
                numeric_cols,
                index=2  # Default to Spending
            )
            
            # Number of segments
            num_segments = st.slider("Number of Segments", 2, 6, 4)
            
            # Create simple segments using quantiles
            seg_data = customer_data.copy()
            
            # Create segment labels
            seg_data[f"{seg_var1} Segment"] = pd.qcut(
                seg_data[seg_var1],
                q=num_segments,
                labels=[f"{seg_var1} Q{i+1}" for i in range(num_segments)]
            )
            
            seg_data[f"{seg_var2} Segment"] = pd.qcut(
                seg_data[seg_var2],
                q=num_segments,
                labels=[f"{seg_var2} Q{i+1}" for i in range(num_segments)]
            )
            
            # Create combined segment
            seg_data["Segment"] = seg_data[f"{seg_var1} Segment"].astype(str) + " + " + seg_data[f"{seg_var2} Segment"].astype(str)
            
            # Show segment sizes
            segment_sizes = seg_data["Segment"].value_counts().reset_index()
            segment_sizes.columns = ["Segment", "Count"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Segment Sizes:")
                st.dataframe(segment_sizes, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    segment_sizes,
                    values="Count",
                    names="Segment",
                    title="Segment Distribution"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Segment profile
            st.write("Segment Profiles:")
            
            # Select profile variable
            profile_var = st.selectbox(
                "Profile Variable",
                numeric_cols,
                index=5  # Default to Satisfaction
            )
            
            # Calculate segment averages
            segment_profile = seg_data.groupby("Segment")[profile_var].mean().reset_index()
            segment_profile = segment_profile.sort_values(profile_var, ascending=False)
            
            fig = px.bar(
                segment_profile,
                x="Segment",
                y=profile_var,
                color="Segment",
                title=f"Average {profile_var} by Segment",
                text_auto=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif advanced_analysis == "Outlier Detection":
            # Select variable for outlier detection
            outlier_var = st.selectbox(
                "Detect Outliers in",
                numeric_cols,
                index=2  # Default to Spending
            )
            
            # Calculate outlier thresholds using IQR method
            Q1 = customer_data[outlier_var].quantile(0.25)
            Q3 = customer_data[outlier_var].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Flag outliers
            outliers = customer_data[(customer_data[outlier_var] < lower_bound) | (customer_data[outlier_var] > upper_bound)]
            
            # Show stats
            st.write(f"**Outlier Analysis for {outlier_var}:**")
            st.write(f"- Lower threshold: {lower_bound:.2f}")
            st.write(f"- Upper threshold: {upper_bound:.2f}")
            st.write(f"- Number of outliers detected: {len(outliers)} ({len(outliers)/len(customer_data)*100:.1f}% of data)")
            
            # Plot distribution with outliers highlighted
            fig = px.histogram(
                customer_data,
                x=outlier_var,
                title=f"Distribution of {outlier_var} with Outlier Thresholds",
                nbins=30
            )
            
            # Add threshold lines
            fig.add_vline(x=lower_bound, line_dash="dash", line_color="red", annotation_text="Lower Threshold")
            fig.add_vline(x=upper_bound, line_dash="dash", line_color="red", annotation_text="Upper Threshold")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show outliers
            if len(outliers) > 0:
                st.write("Outlier Records:")
                st.dataframe(outliers, use_container_width=True)
    
    else:  # Correlation Data
        # Show data preview
        with st.expander("Show Data Preview"):
            st.dataframe(correlation_data.head())
        
        # Feature selection
        st.subheader("Correlation Analysis")
        
        # Select features to include
        all_corr_features = correlation_data.columns.tolist()
        
        selected_features = st.multiselect(
            "Select Features for Analysis",
            all_corr_features,
            default=all_corr_features
        )
        
        if len(selected_features) > 1:
            # Calculate correlation matrix
            corr_matrix = correlation_data[selected_features].corr().round(3)
            
            # Visualization option
            vis_option = st.radio(
                "Visualization Type",
                ["Heatmap", "Pairplot", "Network Graph"],
                horizontal=True
            )
            
            if vis_option == "Heatmap":
                # Customization options
                cmap = st.selectbox(
                    "Color Scheme",
                    ["coolwarm", "viridis", "RdBu", "YlGnBu", "plasma"]
                )
                
                # Plot with Plotly
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale=cmap,
                    title="Correlation Matrix"
                )
                
                fig.update_layout(width=800, height=800)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show strongest correlations
                st.subheader("Strongest Correlations")
                
                # Get pairs of features and their correlations
                corr_pairs = []
                for i in range(len(selected_features)):
                    for j in range(i+1, len(selected_features)):
                        feature1 = selected_features[i]
                        feature2 = selected_features[j]
                        corr_value = corr_matrix.loc[feature1, feature2]
                        corr_pairs.append((feature1, feature2, corr_value))
                
                # Sort by absolute correlation
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Display top correlations
                for i, (feature1, feature2, corr_value) in enumerate(corr_pairs[:5]):
                    st.write(f"{i+1}. **{feature1}** and **{feature2}**: {corr_value:.3f}")
                    
                    # Interpretation
                    if abs(corr_value) < 0.3:
                        strength = "weak"
                    elif abs(corr_value) < 0.7:
                        strength = "moderate"
                    else:
                        strength = "strong"
                    
                    direction = "positive" if corr_value > 0 else "negative"
                    
                    st.write(f"   - {strength.capitalize()} {direction} correlation")
            
            elif vis_option == "Pairplot":
                # Limit to 5 features for performance
                if len(selected_features) > 5:
                    st.warning("Limiting to first 5 features for performance reasons.")
                    plot_features = selected_features[:5]
                else:
                    plot_features = selected_features
                
                # Create pairplot
                fig = px.scatter_matrix(
                    correlation_data[plot_features],
                    dimensions=plot_features,
                    title="Pairplot Matrix"
                )
                
                fig.update_traces(diagonal_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # Network Graph
                # Create network-like visualization for correlations
                st.write("Correlation Network Graph")
                
                # Prepare data for network graph
                network_data = []
                
                for i in range(len(selected_features)):
                    for j in range(i+1, len(selected_features)):
                        feature1 = selected_features[i]
                        feature2 = selected_features[j]
                        corr_value = corr_matrix.loc[feature1, feature2]
                        
                        # Only include if correlation is significant
                        if abs(corr_value) > 0.3:
                            network_data.append({
                                "source": feature1,
                                "target": feature2,
                                "value": abs(corr_value),
                                "color": "blue" if corr_value > 0 else "red"
                            })
                
                if network_data:
                    # Create network visualization using Plotly
                    nodes = list(set([item["source"] for item in network_data] + [item["target"] for item in network_data]))
                    
                    # Simple force-directed layout (very basic)
                    import math
                    import random
                    
                    # Place nodes in a circle
                    angle_step = 2 * math.pi / len(nodes)
                    radius = 1
                    
                    node_positions = {}
                    for i, node in enumerate(nodes):
                        angle = i * angle_step
                        x = radius * math.cos(angle)
                        y = radius * math.sin(angle)
                        node_positions[node] = (x, y)
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add edges
                    for edge in network_data:
                        source = edge["source"]
                        target = edge["target"]
                        value = edge["value"]
                        color = edge["color"]
                        
                        x0, y0 = node_positions[source]
                        x1, y1 = node_positions[target]
                        
                        # Line width based on correlation strength
                        width = value * 5
                        
                        fig.add_trace(go.Scatter(
                            x=[x0, x1, None],
                            y=[y0, y1, None],
                            mode="lines",
                            line=dict(width=width, color=color),
                            hoverinfo="text",
                            text=f"{source} - {target}: {value:.3f}",
                            showlegend=False
                        ))
                    
                    # Add nodes
                    for node, (x, y) in node_positions.items():
                        fig.add_trace(go.Scatter(
                            x=[x],
                            y=[y],
                            mode="markers+text",
                            marker=dict(size=15, color="lightgrey", line=dict(width=2, color="black")),
                            text=node,
                            textposition="top center",
                            name=node
                        ))
                    
                    fig.update_layout(
                        title="Correlation Network (|r| > 0.3)",
                        showlegend=False,
                        hovermode="closest",
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class='insight-box'>
                    <strong>Network Interpretation:</strong><br>
                    - <span style='color:blue'>Blue lines</span>: Positive correlations<br>
                    - <span style='color:red'>Red lines</span>: Negative correlations<br>
                    - Line thickness indicates correlation strength
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No significant correlations (|r| > 0.3) found between selected features.")
        else:
            st.warning("Please select at least two features for correlation analysis.")
        
        # Feature vs. Target analysis
        if "Target" in selected_features and len(selected_features) > 1:
            st.subheader("Feature vs. Target Analysis")
            
            other_features = [f for f in selected_features if f != "Target"]
            target_feature = st.selectbox(
                "Select Feature to Compare with Target",
                other_features
            )
            
            # Create scatter plot with regression line
            fig = px.scatter(
                correlation_data,
                x=target_feature,
                y="Target",
                trendline="ols",
                title=f"Relationship between {target_feature} and Target"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation and metrics
            corr = correlation_data[target_feature].corr(correlation_data["Target"])
            
            # Simple linear regression
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score
            
            X = correlation_data[target_feature].values.reshape(-1, 1)
            y = correlation_data["Target"].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            predictions = model.predict(X)
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Correlation", f"{corr:.3f}")
            
            with col2:
                st.metric("R²", f"{r2:.3f}")
            
            with col3:
                st.metric("MSE", f"{mse:.3f}")
            
            st.markdown(f"""
            <div class='insight-box'>
            <strong>Linear Relationship:</strong><br>
            The variable <strong>{target_feature}</strong> explains {r2*100:.1f}% of the variance in the Target variable.
            <br>
            Relationship: Target = {model.intercept_:.3f} + {model.coef_[0]:.3f} × {target_feature}
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### About This Demo")
st.markdown('''
This demo showcases various data visualization capabilities of Streamlit. The data used is synthetic and generated for demonstration purposes only.

**Key Features Demonstrated:**
- Interactive charts and visualizations
- Data filtering and aggregation
- Custom component layouts
- User inputs and controls
- Multiple dataset analysis
''')