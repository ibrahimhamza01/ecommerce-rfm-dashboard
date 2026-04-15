# E-Commerce Customer Behavior & RFM Analytics

This project analyzes customer purchasing behavior using transactional retail data. It builds a complete end-to-end analytical pipeline from raw data to advanced visualization using **RFM modeling, statistical analysis, data transformation, PCA, clustering, and interactive dashboards**.

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
* Heavy-tailed numerical features
* High-cardinality categorical variables

---

## Project Objective

Transform raw transactional data into actionable insights through:

* **RFM (Recency, Frequency, Monetary) analysis**
* Exploratory Data Analysis (EDA)
* Statistical analysis & **normality testing**
* **Data transformation (Log, Box-Cox, Scaling)**
* Dimensionality reduction (**PCA**)
* Customer segmentation & clustering
* Interactive visualization using **Dash**

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
│   ├── 01_eda_numerical.py
│   ├── 02_eda_categorical.py
│   ├── 03_outlier_detection.py
│   ├── 04_normality_tests.py
│   └── 05_transformation.py
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
* `cleaned_iqr.csv`
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
* Boxplots, violin plots
* Scatter & regression plots
* Pair plots and density visualizations

Saved in:

```
phase1_static/numerical_eda/
phase1_static/categorical_eda/
```

---

### 4. Run Outlier Detection

```bash
python phase1_static/03_outlier_detection.py
```

Includes:

* IQR (primary method)
* Z-score comparison
* Isolation Forest (anomaly detection)
* Before/after visualization
* Percentage of data removed

Saved in:

```
phase1_static/outlier_detection/
```

---

### 5. Run Normality Testing

```bash
python phase1_static/04_normality_tests.py
```

Includes:

* Shapiro-Wilk test
* Kolmogorov-Smirnov test
* D’Agostino K² test
* QQ plots
* Statistical tables
* Interpretation of results

Saved in:

```
phase1_static/normality_tests/
```

---

### 6. Run Data Transformation

```bash
python phase1_static/05_transformation.py
```

Includes:

* Log transformation
* Box-Cox transformation
* Standardization
* MinMax scaling
* Before/after distribution comparison
* Re-evaluated normality tests
* Skewness and kurtosis analysis
* Best transformation selection

Saved in:

```
phase1_static/transformation_outputs/
```

---

### 7. Run Dashboard

```bash
python app.py
```

---

## Data Processing Overview

* Data cleaning:

  * Missing values
  * Duplicate removal
* Feature engineering:

  * `LineTotal`
  * `TransactionStatus`
  * `PurchaseQuarter`
  * `PriceCategory`
* RFM computation (completed transactions only)

### Statistical Analysis

* Outlier detection:

  * IQR (primary)
  * Z-score
  * Isolation Forest
* Normality testing:

  * Shapiro-Wilk
  * Kolmogorov-Smirnov
  * D’Agostino K²

### Data Transformation

* Applied to strictly positive values only
* Methods:

  * Log transformation
  * Box-Cox transformation
  * Standardization
  * MinMax scaling
* Evaluation using:

  * Skewness
  * Kurtosis
  * Re-run normality tests (sample-based)

---

## Key Insights

* Retail transaction data is **highly right-skewed** with heavy tails
* All key variables (**Quantity, Price, LineTotal**) are **non-normal**
* Outliers significantly distort statistical analysis
* IQR effectively stabilizes extreme values
* Log transformation reduces skewness moderately
* **Box-Cox consistently produces near-symmetric distributions**
* Normality tests remain near zero due to large sample size
* Skewness and kurtosis are more reliable indicators for improvement
* Standardization is essential for **PCA and clustering**, not normalization

---

## Data Notes

* Negative values represent **returns and cancellations**
* Cancelled invoices (`Invoice` starting with `'C'`) are preserved and labeled
* Transformation is applied only to **valid positive transactions**
* RFM metrics use **completed transactions only**

---

## Outputs

| File                            | Description                           |
| ------------------------------- | ------------------------------------- |
| preprocessed_transactions.csv   | Cleaned transaction data              |
| cleaned_iqr.csv                 | Outlier-treated dataset               |
| rfm_table.csv                   | Customer-level RFM metrics            |
| rfm_pca.csv                     | PCA-transformed features              |
| transformation_results.csv      | Transformation comparison results     |
| best_transformation_summary.csv | Best method per feature               |
| observations.txt                | Feature-level transformation insights |

---

## Reproducibility

```bash
git clone [<repo-url>](https://github.com/ibrahimhamza01/ecommerce-rfm-dashboard)
cd ecommerce-rfm-dashboard
pip install -r requirements.txt

# Add dataset to /data
python preprocessing/run_pipeline.py
python phase1_static/01_eda_numerical.py
python phase1_static/02_eda_categorical.py
python phase1_static/03_outlier_detection.py
python phase1_static/04_normality_tests.py
python phase1_static/05_transformation.py
```

---

## Final Goal

Develop an interactive dashboard that communicates:

* Customer segmentation
* Behavioral trends
* Statistical insights
* Data-driven business recommendations

---

## Contact

**Syed Ibrahim Hamza**
DATS 6401 – Visualization of Complex Data
