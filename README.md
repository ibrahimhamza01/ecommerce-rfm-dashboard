# E-Commerce Customer Behavior & RFM Analytics

This project analyzes customer purchasing behavior using transactional retail data. It implements a complete end-to-end analytics pipeline from raw data processing to statistical analysis, dimensionality reduction, customer segmentation, and interactive visualization.

---

## Author

**Syed Ibrahim Hamza**

---

## Dataset

- **Online Retail II Dataset (UCI / Kaggle)**
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

- **RFM (Recency, Frequency, Monetary) analysis**
- Exploratory Data Analysis (EDA)
- Statistical analysis and correlation study
- Normality testing and transformation
- Dimensionality reduction (**PCA**)
- Customer segmentation and clustering
- Interactive dashboards using **Dash**

---

## Project Structure

```text
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
├── requirements.txt
└── README.md
````

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

```text
data/online_retail_II.csv
```

### 2. Run Data Pipeline

```bash
python preprocessing/run_pipeline.py
```

This generates:

* `preprocessed_transactions.csv`
* `cleaned_iqr.csv`
* `rfm_table.csv`

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

### 4. Run Layer 9 Statistical Analysis

```bash
python analysis/statistical_analysis.py
```

This generates:

* descriptive statistics table
* Pearson correlation matrix
* Spearman correlation matrix
* Pearson correlation heatmap
* Spearman correlation heatmap
* scatter matrix
* multivariate KDE plots
* `observations.txt`

Saved in:

```text
analysis/statistics_outputs/
```

### 5. Run Dashboard

```bash
python app.py
```

---

## Data Processing Overview

* Data cleaning:

  * missing values handling
  * duplicate removal

* Feature engineering:

  * `LineTotal`
  * `TransactionStatus`
  * `PurchaseQuarter`
  * `PriceCategory`

* RFM metrics computed using **completed transactions only**

---

## Statistical Analysis

### Outlier Detection

* IQR (primary)
* Z-score
* Isolation Forest

### Normality Testing

* Shapiro-Wilk
* Kolmogorov-Smirnov
* D’Agostino K²

### Correlation & Statistical Profiling

* Pearson correlation
* Spearman correlation
* correlation heatmaps
* scatter matrix
* multivariate KDE
* descriptive statistics table

---

## Data Transformation

* Log transformation
* Box-Cox transformation
* Standardization
* MinMax scaling

Evaluation based on:

* skewness
* kurtosis
* normality test comparison

---

## PCA Insights

* First two components explain **~92% of total variance**
* Customer behavior can be represented effectively in **2D space**
* **Frequency and MonetaryValue are strongly positively related**
* **Recency is inversely related** to customer value and activity
* PCA confirms meaningful behavioral structure suitable for segmentation
* Low condition number indicates stable PCA results

---

## Layer 9 Highlights

The Layer 9 statistical analysis focuses on customer-level **RFM relationships**.

### Descriptive Statistics

* `Recency` shows a right-skewed distribution with a long tail of inactive customers
* `Frequency` is strongly right-skewed, indicating that most customers purchase infrequently
* `MonetaryValue` is also right-skewed, showing that a small number of customers account for disproportionately high spending

### Correlation Findings

* **Frequency and MonetaryValue** show a strong positive relationship

  * Pearson: **0.77**
  * Spearman: **0.80**

* **Recency and Frequency** show a moderate negative relationship

  * Pearson: **-0.42**
  * Spearman: **-0.48**

* **Recency and MonetaryValue** show a moderate negative relationship

  * Pearson: **-0.38**
  * Spearman: **-0.42**

### Interpretation

* Customers who buy more often tend to spend more
* Recently active customers are generally more valuable
* Spearman correlations are slightly stronger than Pearson, indicating monotonic but not perfectly linear relationships

---

## Key Insights

* Retail customer behavior is **highly skewed** with heavy tails
* Most customers are **low-frequency, low-value buyers**
* A smaller customer segment contributes a large share of revenue
* **Frequency is the strongest driver of MonetaryValue**
* **Recency is inversely associated** with both spending and activity
* Statistical profiling confirms that RFM is effective for customer behavior analysis
* Transformation improves symmetry, but large real-world retail data remains non-normal

---

## Outputs

| File                                                          | Description                     |
| ------------------------------------------------------------- | ------------------------------- |
| `preprocessed_transactions.csv`                               | Cleaned transaction data        |
| `cleaned_iqr.csv`                                             | Outlier-treated dataset         |
| `rfm_table.csv`                                               | Customer-level RFM metrics      |
| `rfm_pca.csv`                                                 | PCA-transformed features        |
| `transformation_results.csv`                                  | Transformation comparison       |
| `best_transformation_summary.csv`                             | Best method per feature         |
| `phase1_static/subplots/*`                                    | Storytelling subplot figures and narratives |
| `pca_analysis/*`                                              | PCA plots and summaries         |
| `analysis/statistics_outputs/*`                               | Storytelling Statistical Outputs |
| `observations.txt`                                            | Layer-wise analytical observations |

---

## Reproducibility

```bash
git clone https://github.com/ibrahimhamza01/ecommerce-rfm-dashboard
cd ecommerce-rfm-dashboard
pip install -r requirements.txt

# Add dataset to /data
python preprocessing/run_pipeline.py
python phase1_static/01_eda_numerical.py
python phase1_static/02_eda_categorical.py
python phase1_static/03_outlier_detection.py
python phase1_static/04_normality_tests.py
python phase1_static/05_transformation.py
python phase1_static/06_pca_analysis.py
python phase1_static/07_subplots_storytelling.py
python analysis/statistical_analysis.py
```

---

## Final Goal

Build an interactive dashboard that communicates:

* customer segmentation
* behavioral patterns
* statistical insights
* data-driven business recommendations

---

## Contact

**Syed Ibrahim Hamza**
DATS 6401 – Visualization of Complex Data
If you want, I can make this even better by adding a small **Results Preview** section with 3–4 bullets specifically summarizing Layer 9 and PCA together.
```
