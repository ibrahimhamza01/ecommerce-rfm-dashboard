# E-Commerce Customer Behavior & RFM Analytics

This project analyzes customer purchasing behavior using transactional retail data. It implements a complete end-to-end analytical workflow from raw data processing to statistical modeling, dimensionality reduction, customer segmentation, and interactive visualization.

---

## Live Dashboard (Deployed on GCP)

Access the live application:  
https://ecommerce-rfm-dashboard-701238022699.us-east1.run.app/

Note: Initial load may take a few seconds due to Cloud Run cold start.

---

## Author

Syed Ibrahim Hamza  
DATS 6401 – Visualization of Complex Data

---

## Dataset

- Online Retail II Dataset (UCI / Kaggle)  
- ~1 million transactions (2009–2011)  
- UK-based online gift retailer  

### Key Challenges

- Missing customer IDs (~22%)  
- Duplicate records  
- Negative values (returns/cancellations)  
- Highly skewed, heavy-tailed distributions  
- High-cardinality categorical variables  

---

## Project Objective

Transform raw transactional data into actionable insights through:

- RFM (Recency, Frequency, Monetary) analysis  
- Exploratory Data Analysis (EDA)  
- Statistical analysis and hypothesis testing  
- Normality testing and data transformation  
- Dimensionality reduction (PCA)  
- Customer segmentation and clustering  
- Interactive dashboards using Dash  

---

## Project Structure

```

ecommerce-rfm-dashboard/
│
├── data/
│   ├── online_retail_II.csv
│   ├── preprocessed_transactions.csv
│   ├── cleaned_iqr.csv
│   ├── rfm_table.csv
│   └── rfm_pca.csv
│
├── preprocessing/
│   ├── data_loader.py
│   └── run_pipeline.py
│
├── phase1_static/
│   ├── numerical_eda/
│   ├── categorical_eda/
│   ├── outlier_detection/
│   ├── normality_tests/
│   ├── transformation_outputs/
│   ├── pca_analysis/
│   ├── subplots/
│   ├── 01_eda_numerical.py
│   ├── 02_eda_categorical.py
│   ├── 03_outlier_detection.py
│   ├── 04_normality_tests.py
│   ├── 05_transformation.py
│   ├── 06_pca_analysis.py
│   └── 07_subplots_storytelling.py
│
├── analysis/
│   ├── clustering.py
│   ├── statistical_analysis.py
│   └── statistics_outputs/
│
├── layouts/
├── callbacks/
│
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md

````

---

## Installation

```bash
pip install -r requirements.txt
````

### Core Libraries

* pandas, numpy
* scipy, statsmodels
* scikit-learn
* matplotlib, seaborn
* plotly, dash, dash-bootstrap-components, dash-daq

---

## How to Run (Local)

### 1. Add Dataset

Place dataset in:

```
data/online_retail_II.csv
```

---

### 2. Run Data Pipeline

```bash
python preprocessing/run_pipeline.py
```

Generates:

* preprocessed_transactions.csv
* cleaned_iqr.csv
* rfm_table.csv
* rfm_pca.csv

---

### 3. Run Static Analysis

```bash
python phase1_static/01_eda_numerical.py
python phase1_static/02_eda_categorical.py
python phase1_static/03_outlier_detection.py
python phase1_static/04_normality_tests.py
python phase1_static/05_transformation.py
python phase1_static/06_pca_analysis.py
python phase1_static/07_subplots_storytelling.py
```

---

### 4. Run Statistical Analysis

```bash
python analysis/statistical_analysis.py
```

Generates:

* Descriptive statistics
* Correlation matrices (Pearson & Spearman)
* Heatmaps and scatter matrix
* Multivariate KDE plots
* Hypothesis testing results

---

### 5. Run Dashboard

```bash
python app.py
```

---

## Deployment (Layer 11)

This project is deployed using:

* Docker
* Google Cloud Run

### Deployment Command

```bash
gcloud run deploy ecommerce-rfm-dashboard \
  --source . \
  --region us-east1 \
  --allow-unauthenticated
```

---

## Data Processing Overview

### Data Cleaning

* Missing value handling
* Duplicate removal
* Timestamp conversion
* Business-rule handling for negative values

### Feature Engineering

* LineTotal = Price × Quantity
* TransactionStatus (Completed / Cancelled)
* PurchaseQuarter (seasonality)
* PriceCategory (binned pricing)

---

## RFM Modeling

* Recency: Days since last purchase
* Frequency: Number of purchases
* MonetaryValue: Total spending

---

## Analytical Methods

### Outlier Detection

* IQR (primary)
* Z-score
* Isolation Forest

### Normality Testing

* Shapiro–Wilk
* Kolmogorov–Smirnov
* D’Agostino K²

### Data Transformation

* Log transformation
* Box-Cox transformation
* Standardization
* MinMax scaling

---

## Dimensionality Reduction (PCA)

* Scree plot and explained variance
* 2D and 3D projections
* PCA loadings interpretation

---

## Clustering

* K-Means on RFM features
* Customer segmentation into behavioral groups

---

## Statistical Analysis

* Descriptive statistics
* Pearson and Spearman correlations
* Correlation heatmaps
* Scatter matrix
* Multivariate KDE

### Hypothesis Testing

* T-test
* ANOVA
* Chi-square test

---

## Dashboard Overview

The interactive dashboard provides:

* Data loading and inspection
* Data cleaning
* Outlier detection
* Normality testing
* Data transformation
* PCA visualization
* Numerical and categorical analysis
* Statistical testing

### Features

* Interactive Plotly visualizations
* Dynamic transformations
* PCA clustering
* Dash DAQ components
* Callback-driven updates

---

## Key Insights

* Retail data is highly skewed with heavy tails
* Most customers are low-frequency, low-value buyers
* A small segment contributes disproportionately to revenue
* Frequency is the strongest driver of customer value
* Recency inversely correlates with activity and spending
* Real-world data remains non-normal despite transformation

---

## Outputs

| File                          | Description                |
| ----------------------------- | -------------------------- |
| preprocessed_transactions.csv | Cleaned dataset            |
| cleaned_iqr.csv               | Outlier-treated dataset    |
| rfm_table.csv                 | Customer-level RFM metrics |
| rfm_pca.csv                   | PCA-transformed features   |

---

## Reproducibility

```bash
git clone https://github.com/ibrahimhamza01/ecommerce-rfm-dashboard
cd ecommerce-rfm-dashboard

pip install -r requirements.txt

# Add dataset to /data
python preprocessing/run_pipeline.py
python app.py
```

---

## Final Goal

Develop an interactive system that communicates:

* Customer segmentation
* Behavioral patterns
* Statistical insights
* Data-driven business recommendations

---

## Contact

Syed Ibrahim Hamza
DATS 6401 – Visualization of Complex Data
- Or make a **short 1-page portfolio version of this README**
```
