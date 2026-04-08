# E-Commerce Customer Behavior & RFM Analytics
This project is part of **DATS 6401 – Visualization of Complex Data** and focuses on analyzing customer purchasing behavior using transactional retail data. It builds a complete analytical pipeline from raw data to interactive visualization using **RFM modeling, statistical analysis, PCA, clustering, and Dash**.

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
- **Exploratory Data Analysis (EDA)**
- **Outlier detection**
- **Normality testing**
- **Data transformation**
- **Principal Component Analysis (PCA)**
- **Clustering**
- **Interactive dashboard (Dash)**

---

## Project Structure
```
ecommerce-rfm-dashboard/
│
├── data/
│   └── online_retail_II.csv   # (not included in repo due to size)
│
├── assets/
│   └── style.css
│
├── preprocessing/
│   ├── data_loader.py
│   └── run_pipeline.py
│
├── phase1_static/
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
```

---

## Installation
Install required dependencies:
```bash
pip install -r requirements.txt
```

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
* `processed_transactions.csv`
* `rfm_table.csv`
* `rfm_pca.csv`

---

### Step 3 — Run Dashboard
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

## Business Logic
* Negative `Quantity` and `Price` values are **preserved**
  * Represent returns, cancellations, or adjustments
* Transactions with invoice starting with `'C'` are marked as **Cancelled**
* **RFM is computed using only completed transactions**

---

## 📊 Pipeline Outputs

| File                         | Description                   |
| ---------------------------- | ----------------------------- |
| `processed_transactions.csv` | Cleaned + engineered dataset  |
| `rfm_table.csv`              | Customer-level RFM metrics    |
| `rfm_pca.csv`                | PCA-transformed customer data |

---

## Data Availability
Large data files are **not included in this repository** due to GitHub size limits.

To reproduce results:
1. Download dataset
2. Place in `data/`
3. Run preprocessing pipeline

---

## Project Roadmap

### Completed
* [x] Layer 0 — Setup
* [x] Layer 1 — Data pipeline

### In Progress / Upcoming
* [ ] Numerical EDA
* [ ] Categorical EDA
* [ ] Outlier visualization
* [ ] Normality testing
* [ ] Data transformation
* [ ] PCA analysis
* [ ] Clustering
* [ ] Statistical analysis
* [ ] Dash dashboard
* [ ] Deployment (Docker + GCP)

---

## Reproducibility
To reproduce the project:
```bash
git clone <repo-url>
cd ecommerce-rfm-dashboard
pip install -r requirements.txt
# add dataset to /data
python preprocessing/run_pipeline.py
```

---

## Notes
* Functions are designed to be reusable and modular
* Data pipeline is the **single source of truth**
* Processed CSVs are generated, not stored
* All later layers depend on Layer 1 outputs

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
````
