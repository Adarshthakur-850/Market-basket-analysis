import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def get_kpis(df):
    """
    Calculate core KPIs for the dataset.
    """
    total_sales = (df['Quantity'] * df['Unit Price']).sum() if 'Quantity' in df.columns and 'Unit Price' in df.columns else 0.0
    total_transactions = df['Invoice ID'].nunique()
    total_items = df['Quantity'].sum() if 'Quantity' in df.columns else len(df)
    unique_products = df['Product Name'].nunique()
    
    # Average items per invoice (basket size)
    basket_sizes = df.groupby('Invoice ID')['Product Name'].count()
    avg_basket_size = basket_sizes.mean() if not basket_sizes.empty else 0.0
    
    return {
        'total_sales': total_sales,
        'total_transactions': total_transactions,
        'total_items': total_items,
        'unique_products': unique_products,
        'avg_basket_size': avg_basket_size
    }

def get_top_products(df, top_n=10):
    """
    Get top N most selling products.
    """
    if 'Quantity' in df.columns:
        top_prods = df.groupby('Product Name')['Quantity'].sum().reset_index()
        top_prods = top_prods.sort_values(by='Quantity', ascending=False).head(top_n)
        top_prods.columns = ['Product Name', 'Total Quantity Sold']
    else:
        top_prods = df['Product Name'].value_counts().reset_index().head(top_n)
        top_prods.columns = ['Product Name', 'Transaction Count']
    return top_prods

def get_top_categories(df, top_n=10):
    """
    Get top N categories.
    """
    if 'Product Category' not in df.columns:
        return pd.DataFrame()
        
    if 'Quantity' in df.columns:
        top_cats = df.groupby('Product Category')['Quantity'].sum().reset_index()
        top_cats = top_cats.sort_values(by='Quantity', ascending=False).head(top_n)
        top_cats.columns = ['Category', 'Total Quantity Sold']
    else:
        top_cats = df['Product Category'].value_counts().reset_index().head(top_n)
        top_cats.columns = ['Category', 'Transaction Count']
    return top_cats

def get_basket_distribution(df):
    """
    Get basket size (number of items per transaction) distribution.
    """
    basket_sizes = df.groupby('Invoice ID')['Product Name'].count().reset_index()
    basket_sizes.columns = ['Invoice ID', 'Basket Size']
    return basket_sizes

def get_hourly_trend(df):
    """
    Get transaction frequency hourly distribution.
    """
    if 'Transaction Date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['Transaction Date']):
        return pd.DataFrame()
        
    df_hours = df.copy()
    df_hours['Hour'] = df_hours['Transaction Date'].dt.hour
    hourly = df_hours.groupby('Hour')['Invoice ID'].nunique().reset_index()
    hourly.columns = ['Hour', 'Number of Invoices']
    return hourly

def get_monthly_trend(df):
    """
    Get transaction frequency monthly distribution.
    """
    if 'Transaction Date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['Transaction Date']):
        return pd.DataFrame()
        
    df_months = df.copy()
    df_months['Month'] = df_months['Transaction Date'].dt.strftime('%b')
    df_months['Month_Num'] = df_months['Transaction Date'].dt.month
    monthly = df_months.groupby(['Month_Num', 'Month'])['Invoice ID'].nunique().reset_index()
    monthly = monthly.sort_values('Month_Num').drop(columns=['Month_Num'])
    monthly.columns = ['Month', 'Number of Invoices']
    return monthly

def plot_top_products(df, top_n=10):
    """Generate Plotly bar chart for top selling products."""
    data = get_kpis(df)
    top_df = get_top_products(df, top_n)
    
    y_col = top_df.columns[1] # 'Total Quantity Sold' or 'Transaction Count'
    
    fig = px.bar(
        top_df, 
        x='Product Name', 
        y=y_col,
        title=f"Top {top_n} Most Purchased Products",
        labels={'Product Name': 'Product', y_col: 'Sales volume'},
        color=y_col,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(xaxis_tickangle=-45, template='plotly_dark')
    return fig

def plot_top_categories(df, top_n=10):
    """Generate Plotly pie chart for top selling categories."""
    top_df = get_top_categories(df, top_n)
    if top_df.empty:
        return None
        
    y_col = top_df.columns[1]
    fig = px.pie(
        top_df,
        values=y_col,
        names='Category',
        title=f"Top {top_n} Sales Categories",
        color_discrete_sequence=px.colors.sequential.Plasma_r
    )
    fig.update_layout(template='plotly_dark')
    return fig

def plot_basket_distribution(df):
    """Generate Plotly histogram for basket sizes."""
    dist_df = get_basket_distribution(df)
    fig = px.histogram(
        dist_df,
        x='Basket Size',
        nbins=20,
        title='Items per Basket (Transaction Size Distribution)',
        color_discrete_sequence=['#ff7f0e']
    )
    fig.update_layout(
        xaxis_title='Basket Size (No. of Unique Products)',
        yaxis_title='Number of Transactions',
        template='plotly_dark'
    )
    return fig

def plot_sales_trend(df):
    """Generate sales hourly/monthly trends."""
    hourly = get_hourly_trend(df)
    if hourly.empty:
        return None
    
    fig = px.line(
        hourly,
        x='Hour',
        y='Number of Invoices',
        title='Hourly Shopping Patterns (Transaction Count)',
        markers=True
    )
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        xaxis_title='Hour of the Day',
        yaxis_title='Number of Transactions',
        template='plotly_dark'
    )
    return fig
