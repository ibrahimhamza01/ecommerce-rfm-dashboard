# E-Commerce Customer Behavior & RFM Analytics

This project analyzes customer purchasing behavior using transactional retail data. It builds an end-to-end analytical pipeline from raw data to visualization using **RFM modeling, statistical analysis, PCA, clustering, and interactive dashboards**.

---

## Author

**Syed Ibrahim Hamza**

---

## Dataset

* **Online Retail II Dataset (UCI / Kaggle)**
* ~1 million transactions (2009–2011)
* UK-based online gift retailer

### Key Characteristics

* Missing customer IDs (~22%)
* Duplicate transactions
* Negative values (returns/cancellations)
* Highly skewed distributions
* High-cardinality categorical features

---

## Project Objective

Transform raw transactional data into actionable insights through:

* RFM (Recency, Frequency, Monetary) analysis
* Exploratory Data Analysis (EDA)
* Statistical analysis & distribution testing
* Data transformation (log scaling, normalization)
* Dimensionality reduction (PCA)
* Customer segmentation & clustering
* Interactive visualization (Dash)

---

## Project Structure

```
ecommerce-rfm-dashboard/
│
├── data/
│   ├── online_retail_II.csv
│   ├── preprocessed_transactions.csv
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
│   ├── 01_eda_numerical.py
│   └── 02_eda_categorical.py
│
├── analysis/
│   ├── clustering.py
│   └── statistics.py
│
├── layouts/
├── callbacks/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

### Core Libraries

* pandas, numpy
* scipy, statsmodels
* scikit-learn
* matplotlib, seaborn
* plotly, dash

---

## How to Run

### 1. Add Dataset

Place the dataset in:

```
data/online_retail_II.csv
```

---

### 2. Run Data Pipeline

```bash
python preprocessing/run_pipeline.py
```

Generates:

* `preprocessed_transactions.csv`
* `rfm_table.csv`
* `rfm_pca.csv`

---

### 3. Run EDA

```bash
python phase1_static/01_eda_numerical.py
python phase1_static/02_eda_categorical.py
```

Outputs:

* Distribution plots
* Boxplots & violin plots
* Scatter & regression plots
* Density & correlation visualizations

Saved in:

```
phase1_static/numerical_eda/
phase1_static/categorical_eda/
```

---

### 4. Run Dashboard

```bash
python app.py
```

---

## Data Processing Overview

* Data cleaning (duplicates, missing values)
* Feature engineering:

  * `LineTotal`
  * `TransactionStatus`
  * `PurchaseQuarter`
  * `PriceCategory`
* RFM computation (completed transactions only)
* Outlier detection (IQR, Z-score)
* Log transformations for skewed variables
* PCA for dimensionality reduction
* Correlation analysis

---

## Key Insights

* Strong **right-skewed distributions** in transaction values
* Presence of **high-value customer outliers**
* **Frequency positively correlates with MonetaryValue**
* **Recency inversely relates to engagement**
* Customer behavior aligns with standard **RFM segmentation patterns**

---

## Data Notes

* Negative values are preserved to represent **returns/cancellations**
* Cancelled invoices (`Invoice` starting with `'C'`) are flagged
* RFM metrics are computed using **valid completed transactions only**

---

## Outputs

| File                          | Description                |
| ----------------------------- | -------------------------- |
| preprocessed_transactions.csv | Cleaned transaction data   |
| rfm_table.csv                 | Customer-level RFM metrics |
| rfm_pca.csv                   | PCA-transformed features   |

---

## Reproducibility

```bash
git clone <repo-url>
cd ecommerce-rfm-dashboard
pip install -r requirements.txt

# Add dataset to /data
python preprocessing/run_pipeline.py
python phase1_static/01_eda_numerical.py
python phase1_static/02_eda_categorical.py
```

---

## Final Goal

Develop an interactive dashboard that communicates:

* Customer segmentation
* Behavioral trends
* Statistical insights
* Business recommendations

---

## Contact

**Syed Ibrahim Hamza**
DATS 6401 – Visualization of Complex Data
