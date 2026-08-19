import pandas as pd

def get_recommendations(rules_df, current_basket, top_n=5):
    """
    Generate product recommendations based on items in the current basket.
    
    Args:
        rules_df: DataFrame of generated association rules.
        current_basket: List of product names currently in the user's cart.
        top_n: Number of recommendations to return.
        
    Returns:
        recommendations: DataFrame of recommended items with scoring metrics.
    """
    if rules_df.empty or not current_basket:
        return pd.DataFrame(columns=['Recommended Product', 'Confidence', 'Lift', 'Support', 'Based On'])
        
    # Convert current basket items to lowercase/stripped for comparison (or exact matching)
    cart_set = set(item.strip() for item in current_basket)
    
    recommendations_list = []
    
    for idx, row in rules_df.iterrows():
        antecedents = set(row['antecedent_items'])
        consequents = set(row['consequent_items'])
        
        # Check if antecedents is a subset of the customer's cart
        if antecedents.issubset(cart_set):
            # Recommend consequences that are NOT already in the cart
            rec_items = consequents - cart_set
            
            for item in rec_items:
                recommendations_list.append({
                    'Recommended Product': item,
                    'Confidence': row['confidence'],
                    'Lift': row['lift'],
                    'Support': row['support'],
                    'Based On': ", ".join(row['antecedent_items'])
                })
                
    if not recommendations_list:
        return pd.DataFrame(columns=['Recommended Product', 'Confidence', 'Lift', 'Support', 'Based On'])
        
    # Convert to DataFrame
    rec_df = pd.DataFrame(recommendations_list)
    
    # Aggregate multiple rules pointing to the same product (keep the one with highest Lift)
    rec_df = rec_df.sort_values(by=['Lift', 'Confidence'], ascending=[False, False])
    rec_df = rec_df.drop_duplicates(subset=['Recommended Product'], keep='first')
    
    return rec_df.head(top_n).reset_index(drop=True)
