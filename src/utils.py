import os
import random
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def ensure_directories():
    """Ensure that all standard project folders exist."""
    directories = ['data', 'models', 'outputs', 'tests', 'src', 'dashboard']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_model(obj, filepath):
    """Save an object (e.g. trained rules) using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_model(filepath):
    """Load an object using pickle."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def generate_synthetic_data(filepath='data/synthetic_transactions.csv', num_transactions=2000):
    """
    Generate a highly realistic retail dataset of transactions with embedded association rules.
    
    Embedded rules to be discovered:
    - {Bread, Butter} -> {Jam}
    - {Coffee} -> {Sugar, Milk}
    - {Diapers} -> {Beer}
    - {Pasta} -> {Tomato Sauce, Cheese}
    - {Wine} -> {Cheese}
    - {Tea} -> {Cookies}
    """
    ensure_directories()
    
    # Product master with categories and unit prices
    product_catalog = {
        'Bakery': {
            'Bread': 2.50,
            'Croissant': 1.80,
            'Bagel': 1.50,
            'Cookies': 3.00,
            'Cake': 12.00
        },
        'Dairy & Eggs': {
            'Milk': 1.99,
            'Butter': 3.49,
            'Cheese': 5.99,
            'Eggs': 2.99,
            'Yogurt': 1.20
        },
        'Pantry': {
            'Jam': 4.20,
            'Coffee': 7.99,
            'Tea': 4.50,
            'Sugar': 2.10,
            'Pasta': 1.49,
            'Tomato Sauce': 2.79,
            'Olive Oil': 9.99,
            'Rice': 3.20
        },
        'Beverages': {
            'Beer': 8.99,
            'Wine': 14.99,
            'Soda': 1.79,
            'Water': 0.99,
            'Juice': 3.50
        },
        'Baby & Personal Care': {
            'Diapers': 19.99,
            'Baby Wipes': 3.99,
            'Soap': 1.50,
            'Shampoo': 4.99
        }
    }
    
    # Flatten catalog for easy lookup
    all_products = []
    product_details = {}
    for category, products in product_catalog.items():
        for prod_name, price in products.items():
            all_products.append(prod_name)
            product_details[prod_name] = {'category': category, 'price': price}
            
    # Base probabilities for starting items
    base_probs = {
        'Bread': 0.35,
        'Milk': 0.30,
        'Coffee': 0.25,
        'Diapers': 0.15,
        'Pasta': 0.20,
        'Wine': 0.18,
        'Tea': 0.22,
        'Soda': 0.28,
        'Water': 0.40,
        'Eggs': 0.25,
        'Croissant': 0.15,
        'Yogurt': 0.20
    }
    
    # Add other products with standard low base probability
    for prod in all_products:
        if prod not in base_probs:
            base_probs[prod] = 0.10
            
    # Generate transactions
    records = []
    start_date = datetime(2025, 1, 1)
    countries = ['United Kingdom', 'Germany', 'France', 'Spain', 'Italy', 'Netherlands']
    country_weights = [0.70, 0.10, 0.08, 0.05, 0.04, 0.03]
    
    # Keep track of generated IDs
    invoice_num = 536365
    
    for i in range(num_transactions):
        invoice_id = f"INV{invoice_num + i}"
        customer_id = f"CUST{random.randint(10000, 19999)}"
        country = random.choices(countries, weights=country_weights)[0]
        # Random transaction date in the last year
        tx_date = start_date + timedelta(
            days=random.randint(0, 364),
            hours=random.randint(8, 20),
            minutes=random.randint(0, 59)
        )
        
        # Decide items in this basket based on rules
        basket = set()
        
        # 1. Evaluate base products
        for prod, prob in base_probs.items():
            if random.random() < prob:
                basket.add(prod)
                
        # 2. Inject conditional rules with high probability to create associations
        # Rule: Bread -> Butter (75% confidence), {Bread, Butter} -> Jam (60% confidence)
        if 'Bread' in basket:
            if random.random() < 0.75:
                basket.add('Butter')
                if random.random() < 0.60:
                    basket.add('Jam')
                    
        # Rule: Coffee -> Sugar (70% confidence), Coffee -> Milk (65% confidence)
        if 'Coffee' in basket:
            if random.random() < 0.70:
                basket.add('Sugar')
            if random.random() < 0.65:
                basket.add('Milk')
                
        # Rule: Diapers -> Beer (50% confidence), Diapers -> Baby Wipes (80% confidence)
        if 'Diapers' in basket:
            if random.random() < 0.50:
                basket.add('Beer')
            if random.random() < 0.80:
                basket.add('Baby Wipes')
                
        # Rule: Pasta -> Tomato Sauce (85% confidence), {Pasta, Tomato Sauce} -> Cheese (60% confidence)
        if 'Pasta' in basket:
            if random.random() < 0.85:
                basket.add('Tomato Sauce')
                if random.random() < 0.60:
                    basket.add('Cheese')
                    
        # Rule: Wine -> Cheese (55% confidence)
        if 'Wine' in basket:
            if random.random() < 0.55:
                basket.add('Cheese')
                
        # Rule: Tea -> Cookies (65% confidence)
        if 'Tea' in basket:
            if random.random() < 0.65:
                basket.add('Cookies')
                
        # Handle empty basket edge case
        if len(basket) == 0:
            basket.add(random.choice(all_products))
            
        # Write to records
        for product in basket:
            details = product_details[product]
            qty = random.choices([1, 2, 3, 4, 5, 10, 12], weights=[0.5, 0.25, 0.1, 0.05, 0.05, 0.03, 0.02])[0]
            records.append({
                'Invoice ID': invoice_id,
                'Product Name': product,
                'Product Category': details['category'],
                'Quantity': qty,
                'Transaction Date': tx_date.strftime('%Y-%m-%d %H:%M'),
                'Customer ID': customer_id,
                'Country/Region': country,
                'Unit Price': details['price']
            })
            
    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
    print(f"Generated {num_transactions} transactions ({len(df)} records) saved to '{filepath}'")
    return df
