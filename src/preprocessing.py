import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder

def load_data(filepath):
    """
    Load a CSV retail transactional dataset.
    """
    try:
        df = pd.read_csv(filepath)
        # Parse Transaction Date if column exists
        if 'Transaction Date' in df.columns:
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
        return df
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {str(e)}")

def clean_data(df):
    """
    Perform complete preprocessing including:
    - Missing value handling
    - Duplicate transaction removal
    - Outlier filtering (e.g. Quantity <= 0 or Unit Price < 0)
    - Product name cleaning (strip whitespaces, remove junk)
    """
    df_clean = df.copy()
    
    # 1. Handle missing values
    df_clean = df_clean.dropna(subset=['Invoice ID', 'Product Name'])
    
    # Fill remaining missing numeric values
    if 'Quantity' in df_clean.columns:
        df_clean['Quantity'] = df_clean['Quantity'].fillna(1).astype(int)
    if 'Unit Price' in df_clean.columns:
        df_clean['Unit Price'] = df_clean['Unit Price'].fillna(0.0).astype(float)
        
    # Fill remaining text values
    if 'Customer ID' in df_clean.columns:
        df_clean['Customer ID'] = df_clean['Customer ID'].fillna('Anonymous')
    if 'Country/Region' in df_clean.columns:
        df_clean['Country/Region'] = df_clean['Country/Region'].fillna('Unknown')
    if 'Product Category' in df_clean.columns:
        df_clean['Product Category'] = df_clean['Product Category'].fillna('General')
        
    # 2. Product Name normalization
    df_clean['Product Name'] = df_clean['Product Name'].astype(str).str.strip()
    
    # Remove rows where Quantity <= 0 or Unit Price <= 0 (e.g. returns/adjustments)
    if 'Quantity' in df_clean.columns:
        df_clean = df_clean[df_clean['Quantity'] > 0]
    if 'Unit Price' in df_clean.columns:
        df_clean = df_clean[df_clean['Unit Price'] >= 0]
        
    # 3. Duplicate removal
    df_clean = df_clean.drop_duplicates()
    
    return df_clean

def create_basket(df, groupby_col='Invoice ID', item_col='Product Name'):
    """
    Group items by invoice and encode them into a binary/boolean matrix format.
    Returns:
        basket_df: Boolean DataFrame where rows are transactions and columns are products.
        te: Fitted TransactionEncoder instance.
    """
    # Group products by Invoice ID
    transactions = df.groupby(groupby_col)[item_col].apply(list).tolist()
    
    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    
    # Create the dataframe as boolean (required by mlxtend >= 0.17)
    basket_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    return basket_df, te
