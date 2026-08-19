import time
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

def mine_frequent_itemsets(basket_df, min_support=0.02, algorithm='fpgrowth'):
    """
    Mine frequent itemsets using the specified algorithm (apriori or fpgrowth).
    Returns frequent_itemsets, execution_time (seconds).
    """
    # Ensure basket is strictly boolean
    basket_bool = basket_df.astype(bool)
    
    start_time = time.time()
    if algorithm.lower() == 'apriori':
        frequent_itemsets = apriori(basket_bool, min_support=min_support, use_colnames=True)
    elif algorithm.lower() == 'fpgrowth':
        frequent_itemsets = fpgrowth(basket_bool, min_support=min_support, use_colnames=True)
    else:
        raise ValueError("Algorithm must be 'apriori' or 'fpgrowth'")
    
    elapsed_time = time.time() - start_time
    return frequent_itemsets, elapsed_time

def generate_association_rules(frequent_itemsets, min_confidence=0.1, min_lift=1.0):
    """
    Generate association rules from frequent itemsets and apply filters.
    """
    if frequent_itemsets.empty:
        return pd.DataFrame()
        
    try:
        # mlxtend association_rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        if rules.empty:
            return pd.DataFrame()
            
        # Apply lift filter
        rules = rules[rules['lift'] >= min_lift]
        
        if rules.empty:
            return pd.DataFrame()
            
        # Add readability fields: convert frozenset to string & lists
        rules['antecedent_items'] = rules['antecedents'].apply(lambda x: list(x))
        rules['consequent_items'] = rules['consequents'].apply(lambda x: list(x))
        
        rules['antecedent_str'] = rules['antecedent_items'].apply(lambda x: ", ".join(x))
        rules['consequent_str'] = rules['consequent_items'].apply(lambda x: ", ".join(x))
        
        rules['antecedent_len'] = rules['antecedent_items'].apply(len)
        rules['consequent_len'] = rules['consequent_items'].apply(len)
        
        # Sort by lift and confidence
        rules = rules.sort_values(by=['lift', 'confidence'], ascending=[False, False])
        
        # Reorder columns for user-friendliness
        cols_order = [
            'antecedent_str', 'consequent_str', 
            'antecedent_support', 'consequent_support', 
            'support', 'confidence', 'lift', 'leverage', 'conviction',
            'antecedent_items', 'consequent_items', 'antecedent_len', 'consequent_len'
        ]
        
        # Make sure all columns exist (conviction is sometimes inf or NaN)
        cols_order = [c for c in cols_order if c in rules.columns]
        rules = rules[cols_order].reset_index(drop=True)
        
        return rules
        
    except Exception as e:
        print(f"Error during rule generation: {str(e)}")
        return pd.DataFrame()

def compare_algorithms(basket_df, min_support_range=[0.05, 0.03, 0.02, 0.01]):
    """
    Benchmark Apriori vs FP-Growth algorithms across different support levels.
    """
    results = []
    
    # Ensure basket is boolean
    basket_bool = basket_df.astype(bool)
    
    for sup in min_support_range:
        # Run Apriori
        try:
            start = time.time()
            ap_sets = apriori(basket_bool, min_support=sup, use_colnames=True)
            ap_time = time.time() - start
            ap_count = len(ap_sets)
        except Exception:
            ap_time = None
            ap_count = 0
            
        # Run FP-Growth
        try:
            start = time.time()
            fp_sets = fpgrowth(basket_bool, min_support=sup, use_colnames=True)
            fp_time = time.time() - start
            fp_count = len(fp_sets)
        except Exception:
            fp_time = None
            fp_count = 0
            
        results.append({
            'Min Support': sup,
            'Apriori Time (s)': ap_time,
            'Apriori Count': ap_count,
            'FP-Growth Time (s)': fp_time,
            'FP-Growth Count': fp_count
        })
        
    return pd.DataFrame(results)
