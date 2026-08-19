import os
import argparse
import pandas as pd
from src.utils import ensure_directories, generate_synthetic_data, save_model
from src.preprocessing import load_data, clean_data, create_basket
from src.association_rules import mine_frequent_itemsets, generate_association_rules
from src.eda import get_kpis

def run_pipeline(data_path, min_support, min_confidence, algorithm, output_dir):
    """
    Run the entire market basket analysis pipeline.
    """
    print("=" * 60)
    print("      MARKET BASKET ANALYSIS - PIPELINE EXECUTION      ")
    print("=" * 60)
    
    # 1. Ensure directories and data exist
    ensure_directories()
    if not os.path.exists(data_path):
        print(f"Data file '{data_path}' not found. Generating realistic synthetic data...")
        generate_synthetic_data(data_path, num_transactions=2500)
        
    # 2. Preprocess Data
    print(f"\n[Step 1/4] Loading and cleaning transactional data from: {data_path}...")
    raw_df = load_data(data_path)
    cleaned_df = clean_data(raw_df)
    
    kpi_dict = get_kpis(cleaned_df)
    print(f"  - Total invoices: {kpi_dict['total_transactions']}")
    print(f"  - Unique products: {kpi_dict['unique_products']}")
    print(f"  - Total records after cleaning: {len(cleaned_df)}")
    print(f"  - Average basket size: {kpi_dict['avg_basket_size']:.2f}")
    
    # 3. Create Basket Matrix
    print(f"\n[Step 2/4] Formatting transactional data into boolean matrix...")
    basket_df, te = create_basket(cleaned_df)
    print(f"  - Matrix Shape: {basket_df.shape} (Transactions x Products)")
    
    # 4. Mine Frequent Itemsets
    print(f"\n[Step 3/4] Mining frequent itemsets using {algorithm.upper()} (min_support={min_support})...")
    frequent_itemsets, mine_time = mine_frequent_itemsets(basket_df, min_support=min_support, algorithm=algorithm)
    print(f"  - Found {len(frequent_itemsets)} frequent itemsets in {mine_time:.4f} seconds.")
    
    # 5. Generate Association Rules
    print(f"\n[Step 4/4] Generating association rules (min_confidence={min_confidence})...")
    rules_df = generate_association_rules(frequent_itemsets, min_confidence=min_confidence)
    print(f"  - Generated {len(rules_df)} association rules.")
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    fitemsets_path = os.path.join(output_dir, 'frequent_itemsets.csv')
    rules_path = os.path.join(output_dir, 'association_rules.csv')
    model_path = os.path.join('models', 'association_rules_model.pkl')
    
    frequent_itemsets.to_csv(fitemsets_path, index=False)
    rules_df.to_csv(rules_path, index=False)
    
    # Save model model dictionaries for recommendation engine
    model_data = {
        'rules': rules_df,
        'unique_products': sorted(cleaned_df['Product Name'].unique().tolist()),
        'kpis': kpi_dict
    }
    save_model(model_data, model_path)
    
    print("\n" + "=" * 60)
    print(f"SUCCESS: Pipeline completed.")
    print(f"  - Saved Frequent Itemsets to: {fitemsets_path}")
    print(f"  - Saved Association Rules to: {rules_path}")
    print(f"  - Saved Serialization Model to: {model_path}")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Market Basket Analysis Pipeline")
    parser.add_argument('--data-path', type=str, default='data/synthetic_transactions.csv', help='Path to transaction CSV file')
    parser.add_argument('--generate-data', action='store_true', help='Force generation of synthetic data')
    parser.add_argument('--min-support', type=float, default=0.02, help='Minimum support threshold (0.0 to 1.0)')
    parser.add_argument('--min-confidence', type=float, default=0.2, help='Minimum confidence threshold (0.0 to 1.0)')
    parser.add_argument('--algorithm', type=str, default='fpgrowth', choices=['apriori', 'fpgrowth'], help='Mining algorithm to use')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save outputs')
    
    args = parser.parse_args()
    
    if args.generate_data:
        print("Generating new synthetic data...")
        generate_synthetic_data(args.data_path, num_transactions=2500)
    else:
        run_pipeline(args.data_path, args.min_support, args.min_confidence, args.algorithm, args.output_dir)
