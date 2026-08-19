import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add src to python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import ensure_directories, generate_synthetic_data
from src.preprocessing import clean_data, create_basket
from src.association_rules import mine_frequent_itemsets, generate_association_rules, compare_algorithms
from src.eda import get_kpis, plot_top_products, plot_top_categories, plot_basket_distribution, plot_sales_trend
from src.recommendation_engine import get_recommendations

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Market Basket Insights",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    /* Main body background and text */
    .reportview-container {
        background: #0f1116;
    }
    
    /* Title and Subtitle */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 50%, #45B649 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #8a8d98;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom KPI Cards */
    .kpi-card {
        background-color: #1a1e29;
        border: 1px solid #2d3345;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: #FF8E53;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #8a8d98;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Interactive recommender styles */
    .rec-box {
        background-color: #1a1e29;
        border-left: 5px solid #45B649;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Data Loading Helper -----------------
@st.cache_data
def load_and_preprocess_uploaded_data(file_path_or_buffer):
    """Load and clean data (cached)."""
    df = pd.read_csv(file_path_or_buffer)
    if 'Transaction Date' in df.columns:
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
    df_cleaned = clean_data(df)
    return df_cleaned

# Make sure folders exist
ensure_directories()
sample_file = 'data/synthetic_transactions.csv'
if not os.path.exists(sample_file):
    generate_synthetic_data(sample_file, num_transactions=2500)

# ----------------- Sidebar Configuration -----------------
st.sidebar.markdown("### 🛍️ Market Basket Setup")

# File Upload Options
data_source = st.sidebar.radio("Select Data Source", ["Synthetic Dataset (Sample)", "Upload Custom CSV"])
uploaded_file = None

if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Retail Transactions (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = load_and_preprocess_uploaded_data(uploaded_file)
            st.sidebar.success("Custom dataset loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")
            df = load_and_preprocess_uploaded_data(sample_file)
    else:
        st.sidebar.info("Awaiting file upload. Using sample dataset...")
        df = load_and_preprocess_uploaded_data(sample_file)
else:
    df = load_and_preprocess_uploaded_data(sample_file)

# Algorithm Choice & Hyperparameters
st.sidebar.markdown("### ⚙️ Rule Mining Hyperparameters")
mining_algo = st.sidebar.selectbox("Mining Algorithm", ["FP-Growth (Fast)", "Apriori"])
min_support = st.sidebar.slider("Minimum Support (min_support)", 0.005, 0.10, 0.02, step=0.005, help="Minimum percentage of transactions that must contain the itemset.")
min_confidence = st.sidebar.slider("Minimum Confidence (min_confidence)", 0.05, 1.0, 0.3, step=0.05, help="Minimum reliability of the generated rules.")
min_lift = st.sidebar.slider("Minimum Lift (min_lift)", 1.0, 5.0, 1.2, step=0.1, help="Filter out independent or negatively correlated item relationships.")

# ----------------- Main Layout -----------------
st.markdown('<div class="main-title">Market Basket Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Extract actionable shopping associations and recommendations using Apriori and FP-Growth algorithms</div>', unsafe_allow_html=True)

# Tabs
tab_eda, tab_mining, tab_recommender, tab_benchmark = st.tabs([
    "📊 Insights & EDA", 
    "⚙️ Association Rules Explorer", 
    "🎯 Recommendation Simulator", 
    "⚡ Performance Benchmarks"
])

# ----------------- Tab 1: Insights & EDA -----------------
with tab_eda:
    # 1. KPIs
    kpis = get_kpis(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{kpis['total_transactions']:,}</div>
            <div class="kpi-label">Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{kpis['unique_products']:,}</div>
            <div class="kpi-label">Unique Products</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">£{kpis['total_sales']:,.2f}</div>
            <div class="kpi-label">Total Sales Revenue</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{kpis['avg_basket_size']:.2f}</div>
            <div class="kpi-label">Avg Basket Size</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 2. EDA Visualizations
    col_left, col_right = st.columns(2)
    with col_left:
        # Top 10 products
        fig_prod = plot_top_products(df, top_n=10)
        st.plotly_chart(fig_prod, use_container_width=True)
        
        # Basket size distribution
        fig_dist = plot_basket_distribution(df)
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with col_right:
        # Top Categories
        fig_cat = plot_top_categories(df, top_n=10)
        if fig_cat:
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No category data found in dataset.")
            
        # Sales Trend
        fig_trend = plot_sales_trend(df)
        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No timestamp data found to plot shopping patterns.")

# ----------------- Tab 2: Association Rules Explorer -----------------
with tab_mining:
    st.markdown("### Generate and Filter Rules")
    
    # Generate Basket representation
    basket_df, te = create_basket(df)
    
    # Mining trigger button
    algo_clean = 'fpgrowth' if "FP-Growth" in mining_algo else 'apriori'
    
    with st.spinner(f"Mining frequent itemsets and rules using {mining_algo}..."):
        frequent_itemsets, elapsed_time = mine_frequent_itemsets(basket_df, min_support=min_support, algorithm=algo_clean)
        rules = generate_association_rules(frequent_itemsets, min_confidence=min_confidence, min_lift=min_lift)
        
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Mining Time", f"{elapsed_time:.4f} sec")
    col_m2.metric("Frequent Itemsets Found", len(frequent_itemsets))
    col_m3.metric("Association Rules Generated", len(rules))
    
    if not rules.empty:
        # Filter rules interactively
        st.markdown("#### Search and Filter Rules Table")
        
        # Text search input
        search_query = st.text_input("Filter rules containing specific product (case-insensitive):", "")
        
        filtered_rules = rules.copy()
        if search_query:
            query = search_query.strip().lower()
            filtered_rules = filtered_rules[
                filtered_rules['antecedent_str'].str.lower().str.contains(query) |
                filtered_rules['consequent_str'].str.lower().str.contains(query)
            ]
            
        st.write(f"Showing {len(filtered_rules)} rules matching criteria.")
        
        # Format rules for nice displaying
        display_rules = filtered_rules[[
            'antecedent_str', 'consequent_str', 'support', 'confidence', 'lift', 'leverage', 'conviction'
        ]].copy()
        display_rules.columns = ['Antecedent (IF)', 'Consequent (THEN)', 'Support', 'Confidence', 'Lift', 'Leverage', 'Conviction']
        
        # Show rules table
        st.dataframe(display_rules.style.format({
            'Support': '{:.4f}',
            'Confidence': '{:.2%}',
            'Lift': '{:.3f}',
            'Leverage': '{:.4f}',
            'Conviction': '{:.3f}'
        }), use_container_width=True)
        
        # Download button
        csv_rules = filtered_rules.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Rules CSV",
            data=csv_rules,
            file_name="association_rules_output.csv",
            mime="text/csv"
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("#### Scatter Plot: Support vs Confidence (colored by Lift)")
        
        fig_scatter = px.scatter(
            filtered_rules,
            x='support',
            y='confidence',
            size='lift',
            color='lift',
            hover_data=['antecedent_str', 'consequent_str'],
            title='Association Rules Distribution',
            labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'},
            color_continuous_scale='Viridis'
        )
        fig_scatter.update_layout(template='plotly_dark')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    else:
        st.warning("No association rules found with the selected support and confidence thresholds. Try lowering the thresholds in the sidebar.")

# ----------------- Tab 3: Recommendation Simulator -----------------
with tab_recommender:
    st.markdown("### Interactive Shopping Cart Simulator")
    st.write("Add items to your customer cart to see real-time recommendations generated by the association rules mined above.")
    
    unique_products = sorted(df['Product Name'].unique().tolist())
    
    col_cart, col_recs = st.columns([1, 1])
    
    with col_cart:
        st.subheader("🛒 Customer Basket")
        selected_items = st.multiselect(
            "Select items to add to the cart:",
            options=unique_products,
            default=[]
        )
        
        if selected_items:
            st.markdown("##### Current Cart Contents:")
            for item in selected_items:
                st.write(f"- 📦 {item}")
        else:
            st.info("Cart is currently empty. Select products from the dropdown to start simulating.")
            
    with col_recs:
        st.subheader("✨ Intelligent Recommendations")
        
        if selected_items:
            if not rules.empty:
                recommendations = get_recommendations(rules, selected_items, top_n=5)
                
                if not recommendations.empty:
                    st.success(f"Found {len(recommendations)} smart recommendations!")
                    
                    for idx, row in recommendations.iterrows():
                        st.markdown(f"""
                        <div class="rec-box">
                            <strong>⭐ Recommended Product: {row['Recommended Product']}</strong><br/>
                            <span>Based on association rule containing: <em>{row['Based On']}</em></span><br/>
                            <span>Confidence: <strong>{row['Confidence']:.1%}</strong> | Lift: <strong>{row['Lift']:.2f}</strong></span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recommendations found for the current cart combinations. Try adding more items or lowering min support/confidence thresholds.")
            else:
                st.warning("Please configure support/confidence in the sidebar to generate rules first.")
        else:
            st.write("Awaiting cart inputs to generate recommendations...")

# ----------------- Tab 4: Performance Benchmarks -----------------
with tab_benchmark:
    st.markdown("### Apriori vs FP-Growth Execution Comparison")
    st.write("Compare the execution times of Apriori and FP-Growth algorithms across various support levels on the loaded dataset.")
    
    if st.button("🚀 Run Benchmark Analysis"):
        with st.spinner("Running benchmarks... (this might take a few seconds)"):
            supports_range = [0.05, 0.03, 0.02, 0.015, 0.01]
            benchmark_df = compare_algorithms(basket_df, min_support_range=supports_range)
            
            st.markdown("#### Benchmark Results Table")
            st.dataframe(benchmark_df.style.format({
                'Min Support': '{:.2%}',
                'Apriori Time (s)': '{:.4f}',
                'FP-Growth Time (s)': '{:.4f}'
            }), use_container_width=True)
            
            # Plot comparisons
            st.markdown("#### Execution Speed comparison (lower is better)")
            
            fig_bench = go.Figure()
            fig_bench.add_trace(go.Scatter(
                x=benchmark_df['Min Support'],
                y=benchmark_df['Apriori Time (s)'],
                mode='lines+markers',
                name='Apriori Algorithm',
                line=dict(color='#EF553B', width=2)
            ))
            fig_bench.add_trace(go.Scatter(
                x=benchmark_df['Min Support'],
                y=benchmark_df['FP-Growth Time (s)'],
                mode='lines+markers',
                name='FP-Growth Algorithm',
                line=dict(color='#00CC96', width=2)
            ))
            
            fig_bench.update_layout(
                xaxis_title='Minimum Support Threshold',
                yaxis_title='Execution Time (Seconds)',
                xaxis=dict(autorange="reverse"),  # Showing high support to low support
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_bench, use_container_width=True)
            
            st.markdown("""
            > [!TIP]
            > **Key Takeaway:** Notice that as the **Minimum Support** threshold decreases (moving from right to left on the X-axis), the execution time for the **Apriori** algorithm grows exponentially due to the excessive candidate itemsets generated. The **FP-Growth** algorithm uses a tree structure (FP-Tree) and runs significantly faster and scales much better at lower support thresholds!
            """, unsafe_allow_html=True)
    else:
        st.info("Click the button above to run the algorithmic comparison benchmarking.")
