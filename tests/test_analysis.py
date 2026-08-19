import os
import pytest
import pandas as pd
import numpy as np
from src.utils import generate_synthetic_data, ensure_directories
from src.preprocessing import clean_data, create_basket
from src.association_rules import mine_frequent_itemsets, generate_association_rules
from src.recommendation_engine import get_recommendations

@pytest.fixture
def sample_transactions():
    """Fixture returning a simple DataFrame of transactional data."""
    data = [
        {'Invoice ID': 'INV1', 'Product Name': 'Bread', 'Product Category': 'Bakery', 'Quantity': 1, 'Unit Price': 2.50},
        {'Invoice ID': 'INV1', 'Product Name': 'Butter', 'Product Category': 'Dairy', 'Quantity': 1, 'Unit Price': 3.49},
        {'Invoice ID': 'INV1', 'Product Name': 'Jam', 'Product Category': 'Pantry', 'Quantity': 1, 'Unit Price': 4.20},
        {'Invoice ID': 'INV2', 'Product Name': 'Bread', 'Product Category': 'Bakery', 'Quantity': 2, 'Unit Price': 2.50},
        {'Invoice ID': 'INV2', 'Product Name': 'Butter', 'Product Category': 'Dairy', 'Quantity': 1, 'Unit Price': 3.49},
        {'Invoice ID': 'INV3', 'Product Name': 'Coffee', 'Product Category': 'Pantry', 'Quantity': 1, 'Unit Price': 7.99},
        {'Invoice ID': 'INV3', 'Product Name': 'Sugar', 'Product Category': 'Pantry', 'Quantity': 1, 'Unit Price': 2.10},
        {'Invoice ID': 'INV4', 'Product Name': 'Bread', 'Product Category': 'Bakery', 'Quantity': 1, 'Unit Price': 2.50},
        {'Invoice ID': 'INV4', 'Product Name': 'Jam', 'Product Category': 'Pantry', 'Quantity': 1, 'Unit Price': 4.20},
        {'Invoice ID': 'INV5', 'Product Name': 'Bread', 'Product Category': 'Bakery', 'Quantity': 1, 'Unit Price': 2.50},
        {'Invoice ID': 'INV5', 'Product Name': 'Butter', 'Product Category': 'Dairy', 'Quantity': 1, 'Unit Price': 3.49},
        {'Invoice ID': 'INV5', 'Product Name': 'Jam', 'Product Category': 'Pantry', 'Quantity': 1, 'Unit Price': 4.20},
    ]
    return pd.DataFrame(data)

def test_generate_synthetic_data():
    """Test generating synthetic transaction data."""
    test_filepath = 'data/test_synthetic.csv'
    if os.path.exists(test_filepath):
        os.remove(test_filepath)
        
    df = generate_synthetic_data(filepath=test_filepath, num_transactions=20)
    assert os.path.exists(test_filepath)
    assert len(df) > 0
    assert 'Invoice ID' in df.columns
    assert 'Product Name' in df.columns
    
    # Cleanup
    os.remove(test_filepath)

def test_clean_data(sample_transactions):
    """Test cleaning process removes invalid quantities and handles columns."""
    # Add a row with negative quantity and a duplicate row
    bad_row = pd.DataFrame([{'Invoice ID': 'INV6', 'Product Name': 'Soda', 'Product Category': 'Beverages', 'Quantity': -1, 'Unit Price': 1.0}])
    dup_row = pd.DataFrame([{'Invoice ID': 'INV1', 'Product Name': 'Bread', 'Product Category': 'Bakery', 'Quantity': 1, 'Unit Price': 2.50}])
    
    test_df = pd.concat([sample_transactions, bad_row, dup_row], ignore_index=True)
    cleaned_df = clean_data(test_df)
    
    # Assert negative quantity is removed
    assert not (cleaned_df['Quantity'] <= 0).any()
    # Assert duplicate is removed
    assert len(cleaned_df[cleaned_df['Invoice ID'] == 'INV1']) == 3

def test_create_basket(sample_transactions):
    """Test encoding transactions into boolean transaction matrix."""
    basket_df, te = create_basket(sample_transactions)
    
    assert isinstance(basket_df, pd.DataFrame)
    assert basket_df.shape[0] == 5 # 5 unique invoices (INV1 to INV5)
    assert basket_df.dtypes.to_dict()['Bread'] == bool # Must be boolean matrix
    
    # Test specific values
    # INV3 has Coffee and Sugar, but not Bread
    # Let's index by Invoice ID to verify
    basket_indexed = basket_df.copy()
    basket_indexed.index = ['INV1', 'INV2', 'INV3', 'INV4', 'INV5']
    assert basket_indexed.loc['INV3', 'Coffee'] == True
    assert basket_indexed.loc['INV3', 'Bread'] == False

def test_mining_and_rules(sample_transactions):
    """Test association mining finds rules for synthetic data fixture."""
    basket_df, te = create_basket(sample_transactions)
    
    # Mine itemsets with low support so we definitely find some
    frequent_itemsets, _ = mine_frequent_itemsets(basket_df, min_support=0.2, algorithm='fpgrowth')
    assert not frequent_itemsets.empty
    
    # Generate rules
    rules = generate_association_rules(frequent_itemsets, min_confidence=0.2, min_lift=1.0)
    assert not rules.empty
    assert 'antecedent_str' in rules.columns
    assert 'consequent_str' in rules.columns
    assert 'confidence' in rules.columns

def test_recommendation_engine(sample_transactions):
    """Test recommender returns expected suggestions."""
    basket_df, te = create_basket(sample_transactions)
    frequent_itemsets, _ = mine_frequent_itemsets(basket_df, min_support=0.2, algorithm='fpgrowth')
    rules = generate_association_rules(frequent_itemsets, min_confidence=0.2, min_lift=1.0)
    
    # Bread + Butter should recommend Jam (since INV1 and INV5 contain Bread, Butter, Jam)
    recs = get_recommendations(rules, ['Bread', 'Butter'], top_n=2)
    assert not recs.empty
    assert 'Jam' in recs['Recommended Product'].values
    
    # Check that recommendation does not contain items already in cart
    assert 'Bread' not in recs['Recommended Product'].values
    assert 'Butter' not in recs['Recommended Product'].values
