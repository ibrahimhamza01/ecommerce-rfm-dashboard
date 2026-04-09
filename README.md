# E-Commerce Customer Behavior & RFM Analytics

This project is part of **DATS 6401 – Visualization of Complex Data** and focuses on analyzing customer purchasing behavior using transactional retail data. It builds a complete analytical pipeline from raw data to advanced visualization using **RFM modeling, statistical analysis, PCA, clustering, and Dash**.

---

## Author
**Syed Ibrahim Hamza**

---

## Dataset
- **Online Retail II Dataset (UCI / Kaggle CSV version)**
- ~1 million transactions (2009–2011)
- UK-based online gift retailer

### Key Characteristics:
- Missing customer identifiers (~22%)
- Duplicate transactions
- Negative values (returns, cancellations)
- Highly skewed distributions
- High-cardinality categorical variables

---

## Project Objective
To transform raw transactional data into meaningful customer insights using:
- **Recency–Frequency–Monetary (RFM) analysis**
- **Comprehensive Exploratory Data Analysis (EDA)**
- **Outlier detection & visualization**
- **Normality testing**
- **Data transformation**
- **Principal Component Analysis (PCA)**
- **Clustering & segmentation**
- **Interactive dashboard (Dash)**

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
├── assets/
│   └── style.css
│
├── preprocessing/
│   ├── data_loader.py
│   └── run_pipeline.py
│
├── phase1_static/
│   └── numerical_eda/
│
├── analysis/
│   ├── clustering.py
│   └── statistics.py
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
Install required dependencies:
```bash
pip install -r requirements.txt
````

### Main Libraries:

* pandas, numpy
* scipy, statsmodels
* scikit-learn
* matplotlib, seaborn
* plotly, dash, dash-bootstrap-components, dash-daq
* openpyxl, tabulate

---

## How to Run

### Step 1 — Add Dataset

Place the dataset in:

```
data/online_retail_II.csv
```

---

### Step 2 — Run Preprocessing Pipeline

```bash
python preprocessing/run_pipeline.py
```

This generates:

* `preprocessed_transactions.csv`
* `rfm_table.csv`
* `rfm_pca.csv`

---

### Step 3 — Run Numerical EDA (Layer 2)

```bash
python phase1_static/01_eda_numerical.py
```

Generates:

* Histograms, KDE plots
* Distribution plots
* Boxplots & boxen plots
* Violin plots
* Scatter & regression plots
* Joint & pair plots
* QQ plots (normality testing)
* Hexbin plots (density)
* Rug plots
* Area & line plots
* 3D visualization
* Contour density plots

All outputs are saved in:

```
phase1_static/numerical_eda/
```

---

### Step 4 — Run Dashboard

```bash
python app.py
```

---

## Layer 1 — Core Data Pipeline

Implemented in:

```
preprocessing/data_loader.py
```

### Includes:

* Data loading and type correction
* Duplicate removal
* Missing value handling
* Feature engineering:

  * `LineTotal`
  * `TransactionStatus`
  * `PurchaseQuarter`
  * `PriceCategory`
* RFM computation
* Outlier detection (IQR, Z-score)
* Data normalization (log, standardization, min-max)
* PCA (dimensionality reduction)
* Correlation analysis (Pearson, Spearman, p-values)
* Descriptive statistics (mean, std, skew, kurtosis)

---

## Layer 2 — Numerical EDA (COMPLETED)

Implemented in:

```
phase1_static/01_eda_numerical.py
```

### Includes:

* Univariate analysis:

  * Histograms + KDE
  * Dist plots
  * Filled KDE
  * Boxplots
  * Violin plots
  * Rug plots
  * QQ plots

* Multivariate analysis:

  * Scatter plots
  * Regression plots
  * Joint plots
  * Pair plots
  * Hexbin plots
  * Contour plots
  * 3D plots

### Key Insights:

* Strong **right-skewed distributions** in Frequency and MonetaryValue
* Significant **outliers** in high-value customers
* **Negative relationship** between Recency and Frequency
* **Positive relationship** between Frequency and MonetaryValue
* Customer behavior aligns with **RFM theory**

---

## Business Logic

* Negative `Quantity` and `Price` values are **preserved**

  * Represent returns, cancellations, or adjustments
* Transactions with invoice starting with `'C'` are marked as **Cancelled**
* **RFM is computed using only completed transactions**

---

## Pipeline Outputs

| File                            | Description                   |
| ------------------------------- | ----------------------------- |
| `preprocessed_transactions.csv` | Cleaned + engineered dataset  |
| `rfm_table.csv`                 | Customer-level RFM metrics    |
| `rfm_pca.csv`                   | PCA-transformed customer data |

---

## Data Availability

Large data files are **not included** due to GitHub size limits.

To reproduce:

1. Download dataset
2. Place in `data/`
3. Run pipeline

---

## Project Roadmap

### Completed

* [x] Layer 0 — Setup
* [x] Layer 1 — Data pipeline
* [x] Layer 2 — Numerical EDA

### Upcoming

* [ ] Categorical EDA
* [ ] Advanced outlier visualization
* [ ] Data transformation strategies
* [ ] PCA analysis interpretation
* [ ] Customer clustering (K-Means, Hierarchical)
* [ ] Statistical testing
* [ ] Interactive dashboard (Dash)
* [ ] Deployment (Docker + GCP)

---

## Reproducibility

```bash
git clone <repo-url>
cd ecommerce-rfm-dashboard
pip install -r requirements.txt
# Add dataset to /data
python preprocessing/run_pipeline.py
python phase1_static/01_eda_numerical.py
```

---

## Notes

* Modular, reusable functions
* Data pipeline is the **single source of truth**
* All outputs are generated dynamically
* Structured for scalability and extension

---

## Final Goal

A fully deployed interactive dashboard that communicates:

* Customer segmentation
* Behavioral insights
* Statistical validation
* Business recommendations

---

## Contact

**Syed Ibrahim Hamza**
DATS 6401 – Visualization of Complex Data
👉 or help you write **EDA report section (for submission)**
```
