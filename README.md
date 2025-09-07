# Marketing Campaign Analysis

**Project:** Customer Purchase & Marketing Campaign Analysis

**Dataset:** `marketing_data.csv` (source: [Kaggle — Jack Daoud — Marketing Data](https://www.kaggle.com/jackdaoud/marketing-data))

---

## 1. Project Overview

This project performs exploratory data analysis (EDA), feature engineering, preprocessing, visualization, and unsupervised segmentation (K-Means clustering) on a marketing dataset. The goal is to understand customer purchasing behaviour, evaluate campaign effectiveness, identify underperforming channels, and create customer segments for targeted marketing.

Key deliverables in this repo/notebook:

* Data ingestion and cleaning
* Feature engineering (e.g., total purchases, customer age)
* Missing value imputation (KNN Imputer)
* Outlier detection and removal
* Data visualization (boxplots, jointplots, heatmaps, violin plots)
* Clustering (K-Means) and analysis of cluster profiles
* Persisted artifacts: `kmeans_model.h5` and `kmeans_scaler.h5`

---

## 2. Files in this Project

* `marketing_data.csv` — raw dataset (from Kaggle)
* `marketing_analysis_notebook.ipynb` — main notebook with EDA, preprocessing and clustering code
* `kmeans_model.h5` — saved KMeans model
* `kmeans_scaler.h5` — saved StandardScaler
* `README.md` — this file

---

## 3. Environment & Tools

* Python 3.x
* Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `datasist`, `joblib`
* IDE: Jupyter / Colab

Install dependencies (example):

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn datasist joblib
```

---

## 4. Data Loading & Initial Inspection

* Loaded `marketing_data.csv` using `pd.read_csv()` and inspected using `df.info()`, `df.describe()`, `df.sample()`.
* Explored unique values for columns such as `Education`, `Marital_Status`, `Country`, and numeric fields.

---

## 5. Feature Engineering

Created several useful features:

* `totalpurchases` = sum of product spend columns (`MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`).
* `Totalkids` = `Kidhome` + `Teenhome`.
* Converted `Dt_Customer` to datetime and extracted `Year`, `Month`, `Day`.
* Calculated `age_Customer` using `Year - Year_Birth`.

These features improved downstream analysis and interpretation of customer behavior.

---

## 6. Data Cleaning & Transformations

* Cleaned the `Income` field: removed `$` and `,` then converted to `float`.
* Dropped the original `Income` column with spaced name after transformation.
* Checked missing values with `df.isna().sum()` and imputed numeric features using `KNNImputer` from scikit-learn.
* Dropped non-informative or redundant columns prior to modeling: `ID`, `Year_Birth`, `Totalkids`, `Year`, `Month`, `Day`, `age_Customer`, `Recency`, `Dt_Customer` (note: keep whichever are meaningful for your downstream use case).

---

## 7. Outlier Handling

* Used `datasist.structdata.detect_outliers` to identify outlier indices in numeric columns and removed those rows to reduce skew/influence on clustering.

---

## 8. Exploratory Data Analysis & Visualizations

Visualizations used to understand distributions and relationships:

* Boxplots and stripplots for numeric columns (Income, product spends, visits, purchases).
* Heatmaps for correlation analysis with target variables like `NumStorePurchases` and `totalpurchases`.
* Jointplots to visualize pairwise relationships (e.g., `totalpurchases` vs `Income`, `MntGoldProds` vs campaigns).
* Violinplots and swarmplots to examine campaign acceptance by categorical groups (`Education`, `Marital_Status`, `Country`).

### Key EDA Findings

* Top-performing product categories by spend: `MntWines`, `MntMeatProducts`, `MntFishProducts`, `MntFruits`, `MntSweetProducts`, `MntGoldProds` (in that order).
* Campaigns `AcceptedCmp5` and `AcceptedCmp1` appear most successful; `AcceptedCmp2`, `AcceptedCmp3`, `AcceptedCmp4` underperformed.
* Geographic differences: campaign success varies by `Country` — regional effects are present and should be considered in targeting.
* Customers with children tend to have lower `totalpurchases` (negative effect observed).

---

## 9. Encoding & Scaling

* Converted categorical variables (`Education`, `Marital_Status`, `Country`) to dummy variables via `pd.get_dummies(..., drop_first=True)`.
* Standardized numeric features with `StandardScaler` prior to clustering.

---

## 10. Clustering (K-Means)

* Performed an elbow-method scan for `k` from 1 to 99 using `inertia_` to inform the number of clusters.
* Final model used `KMeans(n_clusters=25)` (note: 25 is quite high — consider using silhouette score, gap statistic or domain-driven k selection for production).
* Saved trained model and scaler using `joblib.dump()`:

  * `kmeans_model.h5`
  * `kmeans_scaler.h5`

### Cluster Analysis

* Added `clusters` column to original dataframe and examined cluster-wise aggregates for `Mnt*` fields, `totalpurchases` and `Income`.
* Visualized cluster profiles using violin/swarm/strip plots and heatmaps of cluster correlations.

---

## 11. Model Usage Example

Example: Predict cluster for a new customer (ensure features order matches trained dataframe after encoding & scaling):

```python
# load
import joblib
model = joblib.load('kmeans_model.h5')
scaler = joblib.load('kmeans_scaler.h5')

# example feature vector (must match col order used in training)
sample = [0,0,0,2000,200,200,100,10000,223,333,344,3334,0,0,1,0,0,0,2,0,15000,44244,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0]
cluster = model.predict(scaler.transform([sample]))
print(cluster)
```

> **Important:** The sample vector must match the exact column order and dummies used during training. If you change columns or encoding, re-fit the scaler and model.

---

## 12. Business Insights & Recommendations

1. **Target high-value customers**: Focus retention and premium campaigns on segments with high `totalpurchases` and `Income` (e.g., clusters with high `MntWines` and `MntMeatProducts`).
2. **Revise underperforming channels**: Re-assess campaign strategies for `AcceptedCmp2`, `AcceptedCmp3`, and `AcceptedCmp4` — consider A/B testing creative, offers, or targeting rules.
3. **Regional customization**: Tailor campaigns per country/region since acceptance and product preferences differ geographically.
4. **Family-focused offers**: Customers with children showed lower spends — consider family bundles or promotions targeted at households with kids.
5. **Feature and model improvements**: Consider preserving recency (`Recency`) and `age_Customer` for time-aware models (they were removed here to simplify clustering). Use silhouette score or domain constraints to choose `k`.

---

## 13. Limitations & Next Steps

* `KNNImputer` fills missing numeric values using nearest neighbours — validate imputed values for business plausibility.
* Outlier removal was heuristic — consider winsorizing or modeling with robust algorithms instead.
* KMeans assumes spherical clusters and equal variance — consider Gaussian Mixture Models or hierarchical clustering as alternatives.
* Evaluate cluster stability and business actionability before operationalizing.

---

