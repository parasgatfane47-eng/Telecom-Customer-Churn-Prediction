# Predicting Customer Churn Before It Happens

**A data-driven retention program for a telecom operator ("Company A"), built on 100,000 customer usage and billing records.**

> Consulting Proof of Concept · GCI World 2026 · July 2026

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Results](#key-results)
3. [Repository Structure](#repository-structure)
4. [Dataset](#dataset)
5. [Solution Approach](#solution-approach)
6. [Model Results](#model-results)
7. [Business Impact](#business-impact)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Risks and Limitations](#risks-and-limitations)
10. [Tech Stack](#tech-stack)
11. [How to Run](#how-to-run)
12. [Future Work](#future-work)
13. [Author](#author)

---

## Project Overview

Customer churn is a major profit problem for telecom operators. Industry benchmarks put annual churn at 20–50%, and acquiring a new customer costs roughly 6–7x more than retaining one. Reacting after a customer leaves is expensive; intervening before they leave is not.

This project builds a **churn risk-scoring model** and turns it into an **actionable, quantified retention program**:

- Explore usage, billing and handset data to understand *who* churns and *why*.
- Train and compare four classifiers to rank customers by churn probability.
- Convert the risk score into a **top-decile targeting strategy** with an estimated financial impact.
- Propose a phased rollout with go/no-go gates and risk mitigations.

**Business objective:** reduce avoidable revenue loss by contacting the customers most likely to churn, with an equipment-upgrade and loyalty offer, before they leave.

**ML task:** binary classification. The target is `churn` (1 = churned 31–60 days after observation).

---

## Key Results

| Metric | Value |
|---|---|
| Best model | Tuned **XGBoost** (best of 4 models tested) |
| ROC-AUC (3-fold CV, all 100K customers) | **0.6927** |
| Accuracy / F1 at 0.5 cut-off | 63.8% / 63.7% |
| Churn rate in top-risk decile | **78.5%** vs. 49.6% baseline |
| Lift over random targeting | **1.58x** |
| Est. net annual benefit (per 100K customers) | **~$628K** |
| Est. Year-1 program ROI | **~125.6%** |

The business case rests on planning assumptions (20% save rate, $50 outreach cost), not measured outcomes. See [Business Impact](#business-impact) and [Risks and Limitations](#risks-and-limitations).

---

## Repository Structure

```
.
├── Paras2005.ipynb                  # Full analysis: EDA, feature engineering, modelling, business simulation
├── Paras2005.pptx                   # Final consulting presentation (15 slides)
├── Company_A_Churn_Retention_Proposal.pptx   # Earlier iteration of the deck (Gradient Boosting, AUC 0.68)
├── README.md
└── data/                            # NOT included — see "Dataset" below
    ├── Client.csv
    └── Record.csv
```

> Adjust the file names above to match your repository layout.

---

## Dataset

Two internal tables provided for the PoC engagement:

| File | Contents | Shape |
|---|---|---|
| `Record.csv` | Usage history: calls, minutes, overage, drops, roaming, customer-care contacts, plus the `churn` flag and `months` | 100,000 × 51 |
| `Client.csv` | Customer profile: handset, equipment age, billing, credit class, demographics | 100,000 × 50 |

- Joined on `Customer_ID` (no duplicate IDs, no rows lost) → **100,000 customers × 100 raw columns**.
- **Missing data:** 43 of 100 columns contain missing values, mostly demographic fields (e.g. `numbcars` ~49% missing, `dwllsize` ~38%, `HHstatin` ~38%, `ownrent` ~34%). Missing values are imputed or flagged rather than rows dropped.
- **Class balance:** 49.56% of customers are flagged as churned. A near 50/50 split does not occur in a live customer base, where monthly churn is typically 1–2%. The extract was **deliberately balanced for modelling**. The risk ranking and drivers generalise, but dollar figures are anchored to benchmark assumptions rather than this raw rate.

> The data is proprietary and is **not** included in this repository. Place `Client.csv` and `Record.csv` in the working directory (or update the paths in the notebook) to reproduce results.

---

## Solution Approach

### 1. Data preparation
- Load both tables and verify there are no duplicate `Customer_ID`s.
- Inner join on `Customer_ID` → 100,000 × 100 frame.

### 2. Exploratory data analysis
Key findings:

- **Aging equipment is the #1 churn signal.** `eqpdays` is the strongest numeric correlate of churn. Churners carry equipment ~16% older (421 vs. 363 days), and their handsets are worth ~12% less ($96 vs. $108). Churn rate rises steadily with equipment age, which matches the industry trend toward longer device-replacement cycles.
- **Churners disengage quietly.** They contact customer care ~19% *less* than retained customers and pay ~7% less in monthly recurring charges. Total monthly revenue is nearly identical between groups (within ~1%), so this is a *behavioural* signal, not simply a low-value-customer effect.
- **Geography matters.** Churn by service area ranges from ~56.9% (Northwest/Rocky Mountain) to ~46.0% (DC/Maryland/Virginia).

### 3. Feature engineering
Raw columns are turned into **131 model-ready features**.

| Group | Details |
|---|---|
| **Ratio and flag features** | `drop_call_ratio`, `complete_ratio`, `revenue_per_min`, `custcare_flag`, `eqp_years`, `mou_trend_ratio` (3-month vs. 6-month usage), `subs_ratio`, `mou_ratio`, `overage_burden` (overage revenue as a share of revenue). These normalise raw counts so heavy users don't mislead the model. |
| **Numeric features** | 43 usage, billing, handset and demographic variables. |
| **Categorical encoding** | One-hot encoding (`drop_first=True`) of `area`, `crclscod`, `asl_flag`, `new_cell`, `dualband`, `refurb_new`, `hnd_webcap`, `ownrent`, `creditcd`. |
| **Missing values** | Numeric gaps filled with the **column median** (robust to usage outliers). Missing categoricals coded as an explicit `"missing"` category, since non-response can itself be informative. |

### 4. Model selection and evaluation
- **Four classifiers** compared: Logistic Regression (scaled), Random Forest, Histogram Gradient Boosting, and XGBoost.
- **3-fold stratified cross-validation** with out-of-fold predictions (`cross_val_predict`) across all 100,000 customers, so every customer contributes to the reported metrics.
- **ROC-AUC** is the primary metric. It is threshold-independent and measures how well the model *ranks* customers by risk, which is how the business will use it.

**Final XGBoost configuration:**

```python
XGBClassifier(
    n_estimators=800, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_lambda=1.5, reg_alpha=0.1,
    eval_metric="logloss", random_state=42, n_jobs=-1,
)
```

Shallow trees, a low learning rate, and L1/L2 regularisation control overfitting.

### 5. From score to action
Customers are ranked by predicted churn probability and split into deciles. The **top decile (riskiest 10%)** becomes the retention target list, which also skews toward above-average revenue customers.

### 6. Business simulation
Top-decile churn rate, average revenue, an assumed save rate and an assumed outreach cost are combined into a net-benefit and ROI estimate. See [Business Impact](#business-impact).

---

## Model Results

### Model comparison (3-fold stratified CV, 0.5 threshold)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.5941 | 0.5846 | 0.6297 |
| Random Forest | 0.6111 | 0.6223 | 0.6586 |
| Gradient Boosting (Hist.) | 0.6258 | 0.6289 | 0.6792 |
| **XGBoost (tuned)** | **0.6377** | **0.6374** | **0.6927** |

XGBoost is strongest on every metric tracked. Gradient-boosted trees handle non-linear thresholds well (churn risk does not rise linearly with equipment age) and cope with mixed continuous, ratio and dummy features.

### Feature importance
Equipment age (`eqp_years`) remains the strongest driver, which independently confirms the EDA. Two engineered features earned real importance: `mou_trend_ratio` (accelerating usage decline) and `overage_burden`. One-hot categorical flags can rank higher under XGBoost's "gain" metric than their real-world prevalence suggests, so they should be treated as directional rather than definitive.

### Risk-decile lift (out-of-fold XGBoost predictions)

| Decile | Customers | Actual churn rate | Avg. monthly revenue |
|---|---|---|---|
| **1 (highest risk)** | 10,000 | **78.5%** | $59.85 |
| 2 | 10,000 | 67.0% | $55.72 |
| 3 | 10,000 | 62.3% | $55.32 |
| 4 | 10,000 | 56.1% | $55.04 |
| 5 | 10,000 | 52.7% | $55.11 |
| 6 | 10,000 | 47.9% | $56.56 |
| 7 | 10,000 | 43.0% | $58.11 |
| 8 | 10,000 | 37.7% | $62.05 |
| 9 | 10,000 | 30.8% | $65.00 |
| 10 (lowest risk) | 10,000 | 19.8% | $56.38 |

Top-decile churn is **78.5% vs. a 49.6% baseline, a 1.58x lift**. The top decile averages $59.85/month in revenue vs. $57.91 overall: high risk *and* high value.

---

## Business Impact

A monthly risk-scored retention program, sized **per 100,000 customers**.

**Assumptions**

| Parameter | Value |
|---|---|
| Customers targeted | Top decile = 10,000 |
| Churn rate in target group | 78.5% (out-of-fold actual) |
| Retention success rate | 20% of would-be churners saved *(planning assumption)* |
| Outreach/offer cost | $50 per customer *(planning assumption)* |
| Avg. monthly revenue | $59.85 per customer |

**Outcome**

| Item | Value |
|---|---|
| Expected churners in target group | ~7,853 |
| Customers saved | ~1,571 |
| Gross annual revenue preserved | ~$1.13M |
| Program cost | $500K |
| **Net annual benefit** | **~$628K** |
| **Year-1 ROI** | **~125.6%** (every $1 spent preserves ~$2.26 in revenue) |

Figures scale roughly linearly with customer base (about 10x for a 1M-customer operator). Because the training sample is balanced and the save rate is assumed, treat these as **planning estimates to be validated in a pilot**, not forecasts.

---

## Implementation Roadmap

| Phase | Timing | Activities |
|---|---|---|
| **1. Pilot** | Weeks 1–6 | Score the full base monthly. Contact the top-risk decile in **one region** with an equipment-upgrade and loyalty offer. Track the realised save rate against the 20% assumption. |
| **2. Scale** | Weeks 7–14 | Roll out nationally **if pilot save rate ≥ 15%**. Integrate scoring into the CRM/retention workflow and automate the monthly refresh. |
| **3. Optimise** | Ongoing | Retrain quarterly on fresh data. A/B test offer variants (device subsidy vs. plan discount) by segment. Expand beyond the top decile as ROI data accumulates. |

**Go/no-go metric at each gate:** realised save rate vs. the 20% planning assumption, measured against a **holdout control group**.

---

## Risks and Limitations

| Risk | Mitigation |
|---|---|
| **Moderate discrimination (AUC ≈ 0.69)** | Good enough to rank and prioritise, not to make unilateral decisions. Use as one input alongside agent judgement, not as an auto-cutoff. |
| **Artificially balanced training sample (~50% churn)** | Drivers and rank-ordering transfer to production. Dollar estimates are anchored to benchmark assumptions and must be re-validated on live traffic. |
| **Outreach fatigue / customer experience** | Cap contact frequency, rotate offer types, and exclude customers in an active service complaint. |
| **Data privacy and compliance** | Handle usage and demographic data under existing telecom privacy/consent policies. Have legal/compliance review the scoring before go-live. |
| **Assumed save rate and cost** | Treated as hypotheses. The pilot's control group is designed to test them. |

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3 |
| Data handling | pandas, NumPy |
| Visualisation | Matplotlib, seaborn |
| Modelling | scikit-learn (Logistic Regression, Random Forest, HistGradientBoosting, pipelines, `StratifiedKFold`, `cross_val_predict`, metrics), **XGBoost** |
| Environment | Jupyter / Google Colab |
| Deliverable | PowerPoint deck (`.pptx`) |

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter

# 4. Add the data files (not included) next to the notebook
#    Client.csv, Record.csv

# 5. Launch the notebook
jupyter notebook Paras2005.ipynb
```

The notebook was authored in Google Colab, so it also runs there: upload `Client.csv` and `Record.csv` to the session and run all cells.

> **Note:** the notebook also imports `lightgbm` and `train_test_split`, which are not used in the final pipeline. They can be removed, or install `lightgbm` if you prefer to keep the imports.

---

## Future Work

- Measure actual save rates via the pilot's holdout group and replace the 20% assumption.
- Calibrate probabilities and re-score on a **non-balanced, live** customer extract.
- Systematic hyperparameter search (e.g. Optuna) and probability calibration.
- Add temporal features (usage trajectory over time) and more handset/network quality signals.
- Uplift modelling, to target customers *who respond to an offer*, not just those likely to churn.
- Model explainability (SHAP) to give agents plain-language reasons for each flagged customer.
- Cost-sensitive threshold optimisation tied to offer cost and customer lifetime value.

---

## Author

**Paras** — Associate Consulting Team, GCI World 2026

Feel free to open an issue or reach out with questions or suggestions.