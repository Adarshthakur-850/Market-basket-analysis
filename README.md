# 🛒 Market Basket Analysis

> **An end-to-end retail analytics and recommendation system for discovering product associations, mining frequent itemsets, generating association rules, and producing data-driven product recommendations.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![MLxtend](https://img.shields.io/badge/MLxtend-Association%20Mining-orange.svg)](https://rasbt.github.io/mlxtend/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-purple.svg)](https://plotly.com/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Project Overview

**Market Basket Analysis** is a data mining and machine learning project designed to identify relationships between products purchased together.

The system processes transactional retail data, converts transactions into a basket representation, discovers frequently purchased item combinations, generates association rules, evaluates those rules using statistical metrics, and uses the resulting rules to provide product recommendations.

The project is designed as an end-to-end analytical pipeline rather than a single notebook. It includes:

* Data loading and validation
* Transaction cleaning and preprocessing
* Transaction-to-basket transformation
* Frequent itemset mining
* Apriori and FP-Growth algorithms
* Association-rule generation
* Rule filtering using Support, Confidence and Lift
* Exploratory Data Analysis
* Business KPIs
* Product recommendation engine
* Interactive Streamlit dashboard
* CSV output generation
* Model serialization
* Automated tests
* Docker support

The repository is structured so that the analytical components can be used independently or executed together through the main pipeline.

---

## 🎯 Problem Statement

Retail and e-commerce businesses generate large amounts of transactional data. A transaction usually contains multiple products purchased by a customer, but the raw transaction history does not directly reveal which products have meaningful relationships.

For example:

```text
Customer Basket:
Milk + Bread
```

If many customers repeatedly purchase another product together with this combination, the business can use that relationship for:

* Product recommendations
* Cross-selling
* Product bundling
* Store layout optimization
* Targeted promotions
* Inventory planning
* Personalized shopping experiences

The objective of this project is therefore to transform raw transaction data into actionable product-association knowledge.

---

# 🧠 What is Market Basket Analysis?

Market Basket Analysis (MBA) is a data-mining technique used to discover relationships between products within transactional datasets.

The central question is:

> **"If a customer purchases one or more products, what other products are they likely to purchase?"**

Association Rule Mining represents these relationships using rules such as:

```text
{Product A, Product B} → {Product C}
```

This means that customers who purchase **Product A** and **Product B** may have an increased tendency to purchase **Product C**.

The project uses frequent itemset mining followed by association-rule generation to discover these relationships.

---

# 🔬 How the System Works

The complete workflow is:

```text
                    ┌─────────────────────┐
                    │   Transaction Data  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    │ & Validation        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Transaction Basket  │
                    │ Representation      │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │ Frequent Itemset Mining        │
              │                                │
              │  Apriori       /    FP-Growth  │
              └────────────────┬───────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Association Rules  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Rule Evaluation     │
                    │ Support             │
                    │ Confidence          │
                    │ Lift                │
                    │ Leverage            │
                    │ Conviction          │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴───────────────┐
                ▼                              ▼
      ┌─────────────────────┐       ┌─────────────────────┐
      │ Business Analytics  │       │ Recommendation      │
      │ & Visualization     │       │ Engine              │
      └──────────┬──────────┘       └──────────┬──────────┘
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                    ┌─────────────────────┐
                    │ Streamlit Dashboard │
                    └─────────────────────┘
```

---

# ✨ Key Features

## 1. Data Preprocessing

The preprocessing module prepares raw retail transactions for association-rule mining.

The pipeline handles:

* Missing values
* Invalid transactions
* Duplicate records
* Invalid quantities
* Invalid prices
* Product-name normalization
* Transaction-date parsing
* Unknown customer information
* Unknown country/region information
* Unknown product categories

The implementation expects transaction-oriented fields such as:

```text
Invoice ID
Product Name
Quantity
Unit Price
Customer ID
Country/Region
Product Category
Transaction Date
```

The preprocessing layer also removes records with invalid quantities and normalizes product names before basket creation.

---

# 2. Transaction Basket Creation

Association-rule algorithms require transactions to be represented in a basket format.

The project groups products by transaction/invoice and converts the resulting transaction lists into a boolean matrix using `TransactionEncoder`.

Example:

### Original Transactions

```text
Invoice 101 → Milk, Bread
Invoice 102 → Milk, Eggs
Invoice 103 → Bread, Eggs
```

### Basket Matrix

| Transaction | Milk | Bread | Eggs |
| ----------- | ---: | ----: | ---: |
| 101         |    1 |     1 |    0 |
| 102         |    1 |     0 |    1 |
| 103         |    0 |     1 |    1 |

This representation is then passed to the frequent-itemset mining algorithms.

---

# 3. Frequent Itemset Mining

The project supports two major algorithms:

### Apriori

Apriori progressively discovers item combinations that satisfy the minimum support threshold.

```text
1-itemsets
    ↓
2-itemsets
    ↓
3-itemsets
    ↓
...
```

It is straightforward to understand and useful for smaller datasets, but computational complexity can increase significantly as the number of products grows.

### FP-Growth

FP-Growth is also supported and provides an alternative approach for finding frequent itemsets.

The implementation allows the algorithm to be selected dynamically:

```python
algorithm = "apriori"
```

or:

```python
algorithm = "fpgrowth"
```

The mining module measures execution time and returns both the discovered frequent itemsets and mining duration.

---

# 4. Association Rule Mining

After frequent itemsets are discovered, the project generates association rules.

A rule follows the form:

```text
Antecedent → Consequent
```

Example:

```text
Bread → Butter
```

or:

```text
Milk + Bread → Butter
```

The system filters generated rules according to configurable thresholds such as:

* Minimum confidence
* Minimum lift

Rules are also sorted according to their analytical strength, with Lift and Confidence used to prioritize the most useful relationships.

---

# 📊 Association Rule Metrics

Understanding the metrics is critical because not every frequent relationship is necessarily useful.

## Support

Support measures how frequently an itemset appears across all transactions.

```text
Support(A) =
Number of transactions containing A
------------------------------------
Total number of transactions
```

For an itemset `{Milk, Bread}`:

```text
Support(Milk, Bread)
=
Transactions containing Milk and Bread
/
Total Transactions
```

Higher support indicates that the relationship occurs more frequently.

---

## Confidence

Confidence measures how often the consequent appears when the antecedent appears.

```text
Confidence(A → B)
=
Support(A ∪ B)
----------------
Support(A)
```

For example:

```text
Milk → Bread
Confidence = 0.75
```

means that among transactions containing Milk, approximately 75% also contain Bread.

---

## Lift

Lift measures how much stronger the association is compared with the expected occurrence of the consequent alone.

```text
Lift(A → B)
=
Confidence(A → B)
-----------------
Support(B)
```

Interpretation:

```text
Lift > 1  → Positive association
Lift = 1  → Approximately independent
Lift < 1  → Negative association
```

Lift is especially useful for avoiding misleading rules that appear strong simply because the consequent is already very common.

---

## Leverage

Leverage compares the observed co-occurrence of two itemsets with the co-occurrence expected if they were independent.

```text
Leverage(A → B)
=
Support(A ∪ B)
-
Support(A) × Support(B)
```

A positive leverage value indicates that the products occur together more frequently than expected under independence.

---

## Conviction

Conviction measures the degree to which the presence of the antecedent implies the consequent.

It is particularly useful alongside confidence and lift when ranking rules.

---

# 🤖 Recommendation Engine

The project contains a recommendation engine that converts association rules into product recommendations.

The recommendation process is:

```text
Customer Basket
      ↓
Find rules whose antecedents
are contained in the basket
      ↓
Remove products already
in the basket
      ↓
Rank candidate products
      ↓
Return Top-N recommendations
```

For example:

```text
Current Basket:
Milk
Bread
```

The engine searches for rules such as:

```text
Milk + Bread → Butter
Milk + Bread → Eggs
```

and returns products that are not already in the customer's basket.

Recommendations contain:

* Recommended Product
* Confidence
* Lift
* Support
* Products that triggered the recommendation

The implementation ranks recommendations using Lift and Confidence and removes duplicate recommendations for the same product.

---

# 📈 Exploratory Data Analysis

The project includes an EDA layer for understanding the underlying retail dataset.

The dashboard provides analytical views including:

* Total transactions
* Unique products
* Total sales revenue
* Average basket size
* Top products
* Top product categories
* Basket-size distribution
* Sales trends
* Association-rule distributions

The dashboard calculates and displays these KPIs directly from the processed transaction data.

---

# 📊 Interactive Streamlit Dashboard

The project includes a Streamlit-based interactive dashboard.

The dashboard is organized around three main analytical areas:

### 1. Business Overview

Displays:

```text
Transactions
Unique Products
Total Sales Revenue
Average Basket Size
```

along with interactive visualizations.

### 2. Association Rules Explorer

Users can configure the mining process and inspect generated rules.

The dashboard provides:

* Algorithm selection
* Minimum support
* Minimum confidence
* Minimum lift
* Rule filtering
* Association-rule table
* Rule metrics
* Support vs Confidence visualization
* Lift-based visualization
* CSV export

The current implementation supports both Apriori and FP-Growth through the dashboard.

### 3. Recommendation Simulator

The dashboard provides an interactive shopping-cart simulator.

Users select products from the available catalog and the system generates recommendations based on the mined association rules.

This makes the project useful not only for analyzing historical transactions but also for demonstrating how association rules can power a basic recommendation workflow.

---

# 🗂️ Project Structure

```text
Market-basket-analysis/
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── synthetic_transactions.csv
│
├── outputs/
│   ├── association_rules.csv
│   └── frequent_itemsets.csv
│
├── src/
│   ├── association_rules.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── recommendation_engine.py
│   └── utils.py
│
├── tests/
│   └── test_analysis.py
│
├── Dockerfile
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

The repository currently follows this modular structure, separating the dashboard, data, analytical outputs, source modules, and tests.

---

# 🧩 Module Description

## `main.py`

The main entry point for the complete analytical pipeline.

It performs:

1. Directory preparation
2. Dataset loading
3. Synthetic-data generation when the configured dataset is unavailable
4. Data cleaning
5. Basket creation
6. Frequent itemset mining
7. Association-rule generation
8. Output generation
9. Model serialization

The pipeline saves frequent itemsets, association rules, and serialized recommendation data.

---

## `src/preprocessing.py`

Responsible for:

* Loading transaction data
* Cleaning records
* Handling missing values
* Removing invalid transactions
* Normalizing product names
* Removing duplicates
* Creating the transaction basket

It uses `TransactionEncoder` from MLxtend for transaction encoding.

---

## `src/association_rules.py`

Responsible for:

* Apriori mining
* FP-Growth mining
* Frequent-itemset generation
* Association-rule generation
* Confidence filtering
* Lift filtering
* Rule ranking
* Rule-metric formatting

---

## `src/recommendation_engine.py`

Converts association rules into product recommendations.

It checks whether the antecedent of a rule is contained within the customer's current basket and recommends products from the corresponding consequent.

---

## `src/eda.py`

Contains analytical functions used to calculate KPIs and generate visualizations for understanding customer transaction behavior.

---

## `src/utils.py`

Provides supporting functionality such as directory management, synthetic data generation, and model serialization.

---

## `dashboard/app.py`

Contains the Streamlit application.

It combines:

```text
Data Processing
      +
EDA
      +
Association Mining
      +
Rule Exploration
      +
Recommendation Simulation
```

into a single interactive interface.

---

## `tests/test_analysis.py`

Contains automated tests for validating project functionality.

Keeping analysis code and tests separate makes it easier to verify changes without manually checking every pipeline stage.

---

# 📦 Technology Stack

| Technology       | Purpose                                      |
| ---------------- | -------------------------------------------- |
| **Python**       | Core programming language                    |
| **Pandas**       | Data manipulation and analysis               |
| **NumPy**        | Numerical operations                         |
| **MLxtend**      | Frequent itemset and association-rule mining |
| **Streamlit**    | Interactive dashboard                        |
| **Plotly**       | Interactive visualizations                   |
| **Matplotlib**   | Data visualization                           |
| **Seaborn**      | Statistical visualization                    |
| **Scikit-learn** | Supporting machine-learning utilities        |
| **Pytest**       | Automated testing                            |
| **Docker**       | Containerized execution                      |

The repository's current `requirements.txt` specifies Pandas, NumPy, MLxtend, Streamlit, Plotly, Matplotlib, Seaborn, Scikit-learn, and Pytest.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Adarshthakur-850/Market-basket-analysis.git
cd Market-basket-analysis
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Analysis Pipeline

The complete pipeline can be executed using:

```bash
python main.py
```

The pipeline loads the configured transaction dataset and performs the complete workflow:

```text
Load
  ↓
Clean
  ↓
Create Basket
  ↓
Mine Itemsets
  ↓
Generate Rules
  ↓
Save Results
```

---

# ⚙️ Pipeline Configuration

The pipeline supports configurable parameters including:

```bash
python main.py --help
```

Important parameters include:

```text
--data-path
--min-support
--min-confidence
--algorithm
--output-dir
```

Example:

```bash
python main.py \
    --data-path data/synthetic_transactions.csv \
    --min-support 0.02 \
    --min-confidence 0.10 \
    --algorithm fpgrowth
```

On Windows PowerShell:

```powershell
python main.py `
    --data-path data/synthetic_transactions.csv `
    --min-support 0.02 `
    --min-confidence 0.10 `
    --algorithm fpgrowth
```

---

# 🖥️ Running the Dashboard

Start the Streamlit dashboard with:

```bash
streamlit run dashboard/app.py
```

After startup, Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# 🐳 Running with Docker

The repository includes a `Dockerfile`, allowing the application to be packaged into a container.

Build the image:

```bash
docker build -t market-basket-analysis .
```

Run the container:

```bash
docker run -p 8501:8501 market-basket-analysis
```

Then open:

```text
http://localhost:8501
```

---

# 🧪 Running Tests

Run the test suite with:

```bash
pytest
```

Or:

```bash
python -m pytest
```

For more detailed output:

```bash
pytest -v
```

---

# 📁 Outputs

After running the pipeline, the project can generate analytical outputs such as:

```text
outputs/
│
├── frequent_itemsets.csv
└── association_rules.csv
```

## Frequent Itemsets

Contains frequently occurring combinations of products.

Typical information includes:

```text
support
itemsets
```

## Association Rules

Contains generated rules and their evaluation metrics.

Typical fields include:

```text
Antecedent
Consequent
Support
Confidence
Lift
Leverage
Conviction
```

These outputs can be used for further business analysis or downstream recommendation systems.

---

# 💼 Business Applications

The results of this project can support several retail and e-commerce use cases.

## Cross-Selling

Recommend complementary products.

```text
Laptop → Laptop Bag
```

## Product Bundling

Identify products that frequently occur together.

```text
Coffee + Sugar + Milk
```

can become a promotional bundle.

## Store Layout Optimization

Frequently associated products can potentially be positioned closer together to simplify shopping and encourage complementary purchases.

## Promotional Campaigns

Association rules can identify combinations that could be targeted through discounts or bundled promotions.

## Recommendation Systems

The recommendation engine can provide basic "customers who bought these products may also be interested in..." functionality.

## Inventory Planning

Strong product associations can provide additional context for demand planning and stock availability.

---

# 📐 Example Association Rule

Suppose the mining process produces:

```text
{Milk, Bread} → {Butter}
```

with:

```text
Support    = 0.08
Confidence = 0.65
Lift       = 1.75
```

The interpretation would be:

* The combination occurs in approximately 8% of all transactions.
* Among customers buying Milk and Bread, approximately 65% also purchase Butter.
* The Lift of 1.75 indicates that Butter is purchased with Milk and Bread more frequently than would be expected based only on Butter's overall transaction frequency.

This type of rule can therefore be used as a candidate for product recommendation or cross-selling.

---

# ⚖️ Why Use Both Apriori and FP-Growth?

Different datasets can behave differently depending on:

* Number of transactions
* Number of unique products
* Basket size
* Minimum support
* Number of frequent combinations

The project supports both algorithms so their performance and resulting itemsets can be compared.

### Apriori

Advantages:

* Easy to understand
* Straightforward implementation
* Useful for teaching and smaller datasets

Limitations:

* Can require many candidate-generation steps
* May become expensive with large item spaces

### FP-Growth

Advantages:

* Avoids explicit candidate generation
* Can be substantially faster for suitable datasets
* Useful for larger transaction collections

Limitations:

* More complex internal representation
* Performance still depends on dataset characteristics and thresholds

The project measures mining execution time, making algorithm comparison possible.

---

# 🧮 Choosing Support and Confidence

The thresholds significantly affect the number and usefulness of generated rules.

### If Minimum Support is Too High

You may get:

```text
Very few frequent itemsets
↓
Very few rules
↓
Potentially useful relationships disappear
```

### If Minimum Support is Too Low

You may get:

```text
Large number of itemsets
↓
Large number of rules
↓
More computational cost
↓
More noise
```

The same principle applies to confidence.

Therefore, thresholds should be selected according to:

* Dataset size
* Product diversity
* Transaction frequency
* Business objective
* Acceptable number of rules

The dashboard allows these parameters to be adjusted interactively.

---

# 🔍 Limitations

This project focuses on association-based recommendations and therefore has several limitations.

### 1. Association Does Not Mean Causation

If two products frequently occur together, this does not mean that purchasing one causes the purchase of the other.

### 2. Cold Start

A product with little historical transaction data may not generate enough associations for strong recommendations.

### 3. Threshold Sensitivity

Different support, confidence, and lift thresholds can produce substantially different rule sets.

### 4. Large Product Catalogs

Datasets with thousands or millions of unique products can generate extremely large search spaces.

### 5. Basic Recommendation Strategy

The current recommendation engine primarily relies on association-rule matching rather than incorporating:

* Customer demographics
* Individual customer history
* Product embeddings
* Temporal behavior
* Price sensitivity
* Personalized ranking models

This makes the system interpretable and relatively simple, but it also means it is not intended to replace a full-scale personalized recommender system.

---

# 🔐 Data and Security

Do not commit the following into the repository:

```text
.env
API keys
Passwords
Authentication tokens
Private datasets
Credentials
Cloud secrets
```

Use `.gitignore` and environment variables for sensitive configuration.

---

# 🛠️ Development Roadmap

Potential future improvements include:

### Phase 1 — Data Engineering

* Support multiple transaction schemas
* Add stronger data-validation rules
* Add configurable preprocessing
* Support larger datasets

### Phase 2 — Association Mining

* Add ECLAT
* Improve algorithm benchmarking
* Add rule-ranking strategies
* Add configurable metric combinations

### Phase 3 — Recommendation System

* Customer-level personalization
* Hybrid recommendation models
* Collaborative filtering
* Product similarity
* Customer purchase history
* Personalized ranking

### Phase 4 — Analytics

* Customer segmentation
* RFM analysis
* Cohort analysis
* Customer lifetime value
* Product profitability analysis

### Phase 5 — Production Deployment

* REST API
* Model/version management
* Database integration
* Scheduled retraining
* CI/CD
* Cloud deployment
* Monitoring

---

# 📚 Learning Objectives

This project demonstrates practical understanding of:

* Data preprocessing
* Exploratory data analysis
* Transactional data modeling
* Association-rule mining
* Frequent itemset generation
* Apriori
* FP-Growth
* Support
* Confidence
* Lift
* Leverage
* Conviction
* Recommendation systems
* Data visualization
* Streamlit application development
* Automated testing
* Docker-based deployment
* Modular Python project architecture

---

# 🧑‍💻 Project Architecture

The project follows a modular architecture:

```text
                ┌──────────────┐
                │    main.py   │
                └───────┬──────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
   preprocessing   association      eda
       module        rules         module
          │             │             │
          └─────────────┼─────────────┘
                        │
                        ▼
             recommendation_engine
                        │
                        ▼
                   dashboard
```

This separation makes the project easier to test, maintain, extend, and reuse.

---

# 🧪 Reproducibility

For reproducible results:

1. Use the same dataset.
2. Use the same preprocessing configuration.
3. Use the same minimum-support threshold.
4. Use the same minimum-confidence threshold.
5. Use the same mining algorithm.
6. Keep the Python dependency versions consistent.

The project's `requirements.txt` defines minimum dependency versions for the core analytical stack.

---

# 📌 Quick Start

For users who simply want to run the project:

```bash
git clone https://github.com/Adarshthakur-850/Market-basket-analysis.git

cd Market-basket-analysis

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python main.py

streamlit run dashboard/app.py
```

---

# 🌟 Why This Project Matters

Market Basket Analysis is a practical example of how data mining can transform raw transactional records into business intelligence.

Instead of simply answering:

> "What products were sold?"

the system attempts to answer:

> "Which products are associated with each other, how strong are those associations, and how can those relationships be used to recommend products?"

That transition from raw data to interpretable business recommendations is the central purpose of this project.

---

# 👨‍💻 Author

**Adarsh Thakur**

GitHub:
[Adarshthakur-850](https://github.com/Adarshthakur-850)

Project Repository:
[Market Basket Analysis](https://github.com/Adarshthakur-850/Market-basket-analysis)

---

# 📄 License

This project is intended for educational, research, and portfolio purposes.

If a license file is included in the repository, refer to that file for the applicable licensing terms.

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

Contributions, suggestions, and improvements are welcome.
