import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import BytesIO
import base64
import json
import urllib.parse

# Page configuration for mobile responsiveness
st.set_page_config(
    page_title="Data Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile optimization
st.markdown("""
<style>
    /* Mobile-friendly spacing */
    .stApp {
        max-width: 100%;
    }
    
    /* Larger touch targets for mobile */
    .stButton>button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    /* Better mobile table display */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Responsive charts */
    .js-plotly-plot {
        width: 100% !important;
    }
    
    /* Mobile-friendly selectboxes */
    .stSelectbox {
        margin-bottom: 1rem;
    }
    
    /* Compact headers on mobile */
    @media (max-width: 768px) {
        h1 {
            font-size: 1.5rem;
        }
        h2 {
            font-size: 1.25rem;
        }
        h3 {
            font-size: 1.1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chart_config' not in st.session_state:
    st.session_state.chart_config = {}
if 'saved_charts' not in st.session_state:
    st.session_state.saved_charts = []
if 'current_chart' not in st.session_state:
    st.session_state.current_chart = None

def load_data(uploaded_file):
    """Load CSV or Excel file into DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_numeric_columns(df):
    """Get list of numeric columns from DataFrame"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df):
    """Get list of categorical columns from DataFrame"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def calculate_statistics(df, column):
    """Calculate basic statistics for a column"""
    if pd.api.types.is_numeric_dtype(df[column]):
        return {
            'Mean': df[column].mean(),
            'Median': df[column].median(),
            'Mode': df[column].mode()[0] if not df[column].mode().empty else None,
            'Std Dev': df[column].std(),
            'Min': df[column].min(),
            'Max': df[column].max(),
            'Count': df[column].count(),
            'Missing': df[column].isna().sum()
        }
    else:
        return {
            'Unique Values': df[column].nunique(),
            'Most Common': df[column].mode()[0] if not df[column].mode().empty else None,
            'Count': df[column].count(),
            'Missing': df[column].isna().sum()
        }

def get_color_palette(color_scheme):
    """Get color palette based on scheme selection"""
    palettes = {
        'plotly': px.colors.qualitative.Plotly,
        'viridis': px.colors.sequential.Viridis,
        'cividis': px.colors.sequential.Cividis,
        'rainbow': px.colors.cyclical.IceFire,
        'pastel': px.colors.qualitative.Pastel,
        'bold': px.colors.qualitative.Bold
    }
    return palettes.get(color_scheme, px.colors.qualitative.Plotly)

def create_chart(df, chart_type, x_col, y_col, color_col=None, title="", color_scheme="plotly"):
    """Create interactive Plotly chart"""
    try:
        color_palette = get_color_palette(color_scheme)
        
        if chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=color_palette)
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=color_palette)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=color_palette)
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_col, values=y_col, title=title, color_discrete_sequence=color_palette)
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=color_palette)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, color=color_col, title=title, color_discrete_sequence=color_palette)
        else:
            return None
        
        # Mobile-friendly layout
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(size=12),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def main():
    # Check if there's a shared chart in the URL
    if "chart" in st.query_params:
        try:
            chart_encoded = st.query_params["chart"]
            chart_json = base64.urlsafe_b64decode(chart_encoded).decode()
            shared_config = json.loads(chart_json)
            
            # Show notification about shared chart
            st.info(f"📊 **Shared Chart Configuration Detected:** {shared_config.get('title', 'Untitled Chart')}")
            st.info("⬆️ Upload a dataset with matching columns to recreate this visualization")
            
            # Store shared config in session state
            if 'shared_config' not in st.session_state:
                st.session_state.shared_config = shared_config
        except Exception as e:
            st.warning(f"Could not load shared chart: {str(e)}")
    
    # Header
    st.title("📊 Data Analysis Tool")
    st.markdown("**Analyze data and create visualizations on the go**")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Supports CSV and Excel files"
        )
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df)} rows")
        
        st.divider()
        
        # Navigation
        if st.session_state.df is not None:
            page = st.radio(
                "Navigate",
                ["📋 Data View", "📈 Statistics", "📊 Visualizations"],
                label_visibility="collapsed"
            )
        else:
            page = None
            st.info("Upload a file to get started")
    
    # Main content area
    if st.session_state.df is not None:
        df = st.session_state.df
        
        if page == "📋 Data View":
            st.header("📋 Data View")
            
            # Column filter
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect(
                "Select columns to display",
                all_columns,
                default=all_columns[:10] if len(all_columns) > 10 else all_columns
            )
            
            if selected_columns:
                # Row filter
                col1, col2 = st.columns(2)
                with col1:
                    start_row = st.number_input("From row", min_value=0, max_value=len(df)-1, value=0)
                with col2:
                    end_row = st.number_input("To row", min_value=start_row+1, max_value=len(df), value=min(100, len(df)))
                
                # Display filtered data
                st.dataframe(
                    df[selected_columns].iloc[start_row:end_row],
                    use_container_width=True,
                    height=400
                )
                
                # Data info
                st.subheader("Dataset Info")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        elif page == "📈 Statistics":
            st.header("📈 Statistical Analysis")
            
            # Overall statistics
            st.subheader("Dataset Overview")
            
            numeric_cols = get_numeric_columns(df)
            categorical_cols = get_categorical_columns(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Numeric Columns", len(numeric_cols))
            with col2:
                st.metric("Categorical Columns", len(categorical_cols))
            
            # Column-specific statistics
            st.subheader("Column Statistics")
            
            selected_col = st.selectbox("Select a column to analyze", df.columns.tolist())
            
            if selected_col:
                stats = calculate_statistics(df, selected_col)
                
                # Display statistics in a clean format
                cols = st.columns(2)
                for idx, (key, value) in enumerate(stats.items()):
                    with cols[idx % 2]:
                        if isinstance(value, (int, np.integer)):
                            st.metric(key, f"{value:,}")
                        elif isinstance(value, (float, np.floating)):
                            st.metric(key, f"{value:.2f}")
                        else:
                            st.metric(key, str(value))
                
                # Distribution visualization for numeric columns
                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    st.subheader("Distribution")
                    fig = px.histogram(df, x=selected_col, marginal="box", title=f"Distribution of {selected_col}")
                    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.subheader("Value Counts")
                    value_counts = df[selected_col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                title=f"Top 10 Values in {selected_col}",
                                labels={'x': selected_col, 'y': 'Count'})
                    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix for numeric data
            if len(numeric_cols) > 1:
                st.subheader("Correlation Matrix")
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, 
                               text_auto=True,
                               aspect="auto",
                               color_continuous_scale='RdBu_r',
                               title="Correlation Heatmap")
                fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
        
        elif page == "📊 Visualizations":
            st.header("📊 Create Visualizations")
            
            # Check if there's a shared config to apply
            if 'shared_config' in st.session_state and st.session_state.shared_config:
                st.info("🔗 **Shared Chart Available!** Click below to recreate it with your uploaded data.")
                if st.button("📊 Apply Shared Chart Configuration", type="primary", use_container_width=True):
                    config = st.session_state.shared_config
                    try:
                        recreated_fig = create_chart(
                            df, 
                            config['type'], 
                            config['x'], 
                            config['y'], 
                            config.get('color'), 
                            config['title'],
                            config.get('color_scheme', 'plotly')
                        )
                        if recreated_fig:
                            st.plotly_chart(recreated_fig, use_container_width=True)
                            st.success(f"✅ Successfully recreated: {config['title']}")
                            # Clear the shared config after applying
                            st.session_state.shared_config = None
                        else:
                            st.error("Could not recreate the chart. Please check if your data has the required columns.")
                    except Exception as e:
                        st.error(f"Error recreating shared chart: {str(e)}")
                        st.info("Make sure your dataset has matching column names.")
                st.divider()
            
            # Chart type selection
            chart_type = st.selectbox(
                "Select chart type",
                ["Line Chart", "Bar Chart", "Scatter Plot", "Pie Chart", "Box Plot", "Histogram"]
            )
            
            # Dynamic column selection based on chart type
            numeric_cols = get_numeric_columns(df)
            all_cols = df.columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if chart_type == "Pie Chart":
                    x_col = st.selectbox("Category column", all_cols)
                    y_col = st.selectbox("Value column", numeric_cols)
                elif chart_type == "Histogram":
                    x_col = st.selectbox("Column to analyze", numeric_cols)
                    y_col = None
                else:
                    x_col = st.selectbox("X-axis", all_cols)
                    y_col = st.selectbox("Y-axis", numeric_cols if chart_type in ["Line Chart", "Bar Chart", "Scatter Plot", "Box Plot"] else all_cols)
            
            with col2:
                if chart_type != "Histogram":
                    color_col = st.selectbox("Color by (optional)", [None] + all_cols)
                else:
                    color_col = st.selectbox("Color by (optional)", [None] + all_cols)
            
            # Chart customization
            with st.expander("🎨 Customize Chart"):
                chart_title = st.text_input("Chart title", f"{chart_type}: {x_col} vs {y_col if y_col else ''}")
                color_scheme = st.selectbox("Color scheme", ["plotly", "viridis", "cividis", "rainbow", "pastel", "bold"])
            
            # Create and display chart
            if st.button("Generate Visualization", type="primary", use_container_width=True):
                with st.spinner("Creating visualization..."):
                    fig = create_chart(df, chart_type, x_col, y_col, color_col, chart_title, color_scheme)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save chart config to session state
                        chart_config = {
                            'type': chart_type,
                            'x': x_col,
                            'y': y_col,
                            'color': color_col,
                            'title': chart_title,
                            'color_scheme': color_scheme
                        }
                        st.session_state.chart_config = chart_config
                        st.session_state.current_chart = fig
                        
                        # Success message
                        st.success("✅ Visualization created!")
                        
                       # ==============================
# 📤 SHARE & EXPORT SECTION
# ==============================
st.divider()
st.subheader("📤 Share & Export")

import io
import json
import base64
import os
import plotly.io as pio
from datetime import datetime

col1, col2 = st.columns(2)

with col1:
    try:
        # Export HTML (TEXT-BASED BUFFER)
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, include_plotlyjs='cdn')
        html_data = html_buffer.getvalue()

        st.download_button(
            label="📥 Download HTML",
            data=html_data,
            file_name=f"{chart_title.replace(' ', '_')}.html",
            mime="text/html",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"⚠️ HTML Export Failed: {e}")

with col2:
    try:
        # Export PNG (requires kaleido)
        png_bytes = pio.to_image(fig, format="png", width=1200, height=800)
        st.download_button(
            label="📸 Download PNG",
            data=png_bytes,
            file_name=f"{chart_title.replace(' ', '_')}.png",
            mime="image/png",
            use_container_width=True
        )
    except Exception as e:
        st.info("ℹ️ PNG Export unavailable — Kaleido may not be installed or allowed here.")
        st.caption(f"Error detail: {e}")

# ==============================
# 💾 SAVE CHART TO SESSION
# ==============================
if st.button("💾 Save to My Charts", use_container_width=True):
    st.session_state.saved_charts.append({
        'config': chart_config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    st.success(f"✅ Chart saved! You now have {len(st.session_state.saved_charts)} saved chart(s).")

# ==============================
# 🔗 SHAREABLE LINK
# ==============================
st.subheader("🔗 Shareable Link")

try:
    # Encode chart config to base64 for URL
    config_json = json.dumps(chart_config)
    config_encoded = base64.urlsafe_b64encode(config_json.encode()).decode()

    # Determine base URL
    replit_url = os.getenv('REPLIT_DEV_DOMAIN') or os.getenv('REPLIT_DEPLOYMENT_URL')
    base_url = f"https://{replit_url}" if replit_url else "http://localhost:8501"

    shareable_url = f"{base_url}?chart={config_encoded}"

    st.text_input(
        "Copy this link to share:",
        value=shareable_url,
        key="shareable_link",
        help="Anyone with this link can view the chart configuration. They'll need the same dataset to recreate it."
    )
    st.caption("📋 Share this link with colleagues who have the same dataset to recreate this visualization.")
except Exception as e:
    st.error(f"Share link creation failed: {e}")

# ==============================
# 📄 OPTIONAL: VIEW CHART CONFIG JSON
# ==============================
with st.expander("📄 View Chart Configuration (Alternative)"):
    st.code(json.dumps(chart_config, indent=2), language="json")
    st.caption("Copy and paste this JSON configuration to share manually.")

# ==============================
# 📊 DISPLAY SAVED CHARTS
# ==============================
if st.session_state.saved_charts:
    st.divider()
    st.subheader("💾 My Saved Charts")

    for idx, saved_chart in enumerate(st.session_state.saved_charts):
        with st.expander(f"Chart {idx + 1}: {saved_chart['config']['title']} - {saved_chart['timestamp']}"):
            st.json(saved_chart['config'])
            col1, col2 = st.columns(2)

            with col1:
                if st.button(f"📊 Recreate", key=f"recreate_{idx}"):
                    config = saved_chart['config']
                    recreated_fig = create_chart(
                        df,
                        config['type'],
                        config['x'],
                        config['y'],
                        config['color'],
                        config['title'],
                        config.get('color_scheme', 'plotly')
                    )
                    if recreated_fig:
                        st.plotly_chart(recreated_fig, use_container_width=True)

            with col2:
                if st.button(f"🗑️ Delete", key=f"delete_{idx}"):
                    st.session_state.saved_charts.pop(idx)
                    st.rerun()

st.info("💡 **Tip:** Right-click the chart to download as PNG or interact using touch gestures on mobile.")

    
    # ============================================
# 📊 MAIN APPLICATION ENTRY POINT
# ============================================

def main():
    st.set_page_config(page_title="Data Visualizer Chatbot", layout="wide")
    st.title("📈 Data Visualizer Chatbot")

    # Initialize session state
    if "df" not in st.session_state:
        st.session_state.df = None
    if "saved_charts" not in st.session_state:
        st.session_state.saved_charts = []

    # Sidebar upload
    uploaded_file = st.sidebar.file_uploader("📤 Upload your data file (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Load the uploaded dataset
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
            # Continue to visualization interface
            show_data_interface(df)
        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")

    else:
        # ==============================
        # WELCOME SCREEN (no file yet)
        # ==============================
        st.markdown("""
        ### Welcome to the Data Analysis Tool! 👋
        
        This mobile-friendly tool helps you:
        - 📤 Upload CSV and Excel files
        - 🔍 Explore your data with interactive tables
        - 📊 Generate statistical insights
        - 📈 Create beautiful, shareable visualizations
        
        **Get started by uploading a data file using the sidebar.**
        """)

        # Sample data option
        if st.button("📂 Try with Sample Data"):
            sample_df = pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=50, freq='D'),
                'Sales': np.random.randint(1000, 5000, 50),
                'Customers': np.random.randint(50, 200, 50),
                'Region': np.random.choice(['North', 'South', 'East', 'West'], 50),
                'Product': np.random.choice(['Product A', 'Product B', 'Product C'], 50)
            })
            st.session_state.df = sample_df
            st.rerun()


# ============================================
# 🚀 APP LAUNCHER
# ============================================
if __name__ == "__main__":
    main()

