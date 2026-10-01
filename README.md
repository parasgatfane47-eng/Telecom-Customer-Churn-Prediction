# Telecom Customer Churn Prediction & Retention Program

> Consulting proof of concept (GCI World 2026): a machine-learning risk model plus a business case for a targeted, monthly retention program, built on 100,000 customer usage and billing records.

## Overview

Telecom operators lose a large share of revenue to churn, and acquiring a new customer costs far more than keeping one. This project builds a model that **ranks customers by churn risk** so a fixed retention budget can be spent on the customers most likely to leave. It then **translates the model into a quantified business case** (cost, customers saved, revenue preserved, ROI) and a phased rollout plan.

**Deliverables**
- Jupyter notebook: end-to-end analysis (EDA, feature engineering, modeling, business simulation)
- 15-slide executive deck: findings, impact estimate, roadmap, risks

## Headline Results

| Metric | Result |
|---|---|
| Best model | Tuned XGBoost |
| ROC-AUC (3-fold stratified CV, all 100k customers) | **0.693** |
| Accuracy / F1 at 0.5 cut-off | 63.8% / 63.7% |
| Churn rate in top-risk decile | **78.5%** vs 49.6% baseline |
| Lift over random targeting (top decile) | **1.58x** |
| Illustrative net annual benefit (per 100k customers) | **~$628K** |
| Illustrative Year-1 ROI | **~125.6%** |

## Data

Two internal tables, joined on `Customer_ID` (no duplicates, no lost rows):

| File | Shape | Contents |
|---|---|---|
| `Record.csv` | 100,000 × 51 | Usage history: calls, minutes, overage, dropped/blocked calls, roaming, customer-care contacts, `churn` flag |
| `Client.csv` | 100,000 × 50 | Customer profile: handset, equipment age, tenure, billing, demographics |

- Merged dataset: 100,000 customers × ~100 raw columns; 43 of 100 columns contain missing values (mostly demographics).
- Target: `churn` (1 = churned 31–60 days after observation).
- **The extract is artificially balanced (49.6% churn).** Real monthly churn is ~1–2%. Risk *rankings* and *drivers* transfer to production; absolute dollar figures are illustrative scenarios and should be re-validated on live data.

> The raw data is client-confidential and **is not included in this repository.** See [How to Run](#how-to-run).

## Approach

### 1. Exploratory Data Analysis
- **Aging equipment is the strongest churn signal.** Churners hold equipment ~16% older (421 vs 363 days) and lower-value handsets (~$96 vs ~$108).
- **Churners disengage quietly.** They contact customer care ~19% less and pay ~7% less in monthly recurring charges, while total revenue is within ~1% of retained customers. This is a behavioral signal, not simply a "low-value customer" effect.
- Churn varies by region (e.g., Northwest/Rocky Mountain ≈ 56.9% vs Midwest ≈ 45.9%).

### 2. Feature Engineering (~100 raw columns → 131 model-ready features)
- **Ratio/flag features:** `drop_call_ratio`, `complete_ratio`, `revenue_per_min`, `custcare_flag`, `eqp_years`, `mou_trend_ratio` (3-month vs 6-month usage), `subs_ratio`, `mou_ratio`, `overage_burden`
- **Encoding:** one-hot encoding of 9 categorical fields (area, credit class, account spending limit, new-cell flag, dualband, refurbished/new, handset web capability, home ownership, credit-card indicator)
- **Missing values:** numeric → column median; categorical → explicit `"missing"` category (non-response can itself be informative)

### 3. Modeling & Evaluation
Four classifiers compared with **3-fold stratified cross-validation** and out-of-fold predictions, so every customer contributes to the reported metrics.

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.594 | 0.585 | 0.630 |
| Random Forest | 0.611 | 0.622 | 0.659 |
| Histogram Gradient Boosting | 0.626 | 0.629 | 0.679 |
| **XGBoost (tuned)** | **0.638** | **0.637** | **0.693** |

ROC-AUC is the primary metric because the business uses the model to *rank* customers by risk. XGBoost uses shallow trees (depth 5), a low learning rate (0.03), subsampling, and L1/L2 regularization to control overfitting.

### 4. From Score to Action: Risk-Decile Targeting
Customers are ranked by predicted churn probability and split into deciles:

| Risk decile | Actual churn rate |
|---|---|
| 1 (highest risk) | 78.5% |
| 2 | 67.0% |
| 5 | 52.7% |
| 10 (lowest risk) | 19.8% |

The top decile also has above-average revenue ($59.85/mo vs $57.91 overall), so the highest-risk group is also high-value.

### 5. Business Case (illustrative, per 100,000 customers)

| Assumption | Value |
|---|---|
| Targeted group | Top decile (10,000 customers) |
| Churn rate in that group | 78.5% |
| Retention success rate | 20% of would-be churners saved |
| Outreach cost | $50 per customer |
| Avg. monthly revenue | $59.85 per customer |

| Output | Value |
|---|---|
| Customers saved annually | ~1,571 |
| Gross annual revenue preserved | ~$1.13M |
| Program cost | $500K |
| **Net annual benefit** | **~$628K** |
| **ROI** | **~125.6%** (~$2.26 revenue preserved per $1 spent) |

Figures scale linearly with customer base. They depend on the retention success-rate assumption, which the pilot is designed to validate.

## Recommended Rollout

1. **Pilot (weeks 1–6):** score the full base monthly; contact the top-risk decile in one region with an equipment-upgrade + loyalty offer; use a holdout control group; track realized save-rate vs the 20% assumption.
2. **Scale (weeks 7–14):** go national if pilot save-rate ≥ 15%; integrate scoring into the CRM; automate monthly score refresh.
3. **Optimize (ongoing):** retrain quarterly, A/B test offer types (device subsidy vs plan discount), expand beyond the top decile as ROI data accumulates.

## Risks & Limitations
- **Moderate discrimination (AUC ≈ 0.69):** good for ranking and prioritizing, not for automated decisions. Use as one input alongside agent judgment.
- **Balanced training sample:** rank-ordering and drivers transfer; absolute rates and dollar impact must be re-validated on live traffic.
- **Outreach fatigue:** cap contact frequency, rotate offers, exclude customers in active complaints.
- **Privacy & compliance:** handling of usage and demographic data must follow telecom privacy/consent policies; legal review before go-live.
- Feature importance from tree models is directional, not causal.

## Repository Structure

```
.
├── notebooks/
│   └── churn_analysis.ipynb        # full analysis: EDA → features → models → business simulation
├── presentation/
│   └── Churn_Retention_Proposal.pptx   # 15-slide executive deck
├── data/                           # NOT tracked: place Client.csv and Record.csv here
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

1. Clone the repo and install dependencies:
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Place the two source files in `data/` (`Client.csv`, `Record.csv`). They are not included for confidentiality.
3. Open the notebook and run all cells (the file paths in the first cell point to `data/`):
   ```bash
   jupyter notebook notebooks/churn_analysis.ipynb
   ```

## Tech Stack
Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · seaborn · Jupyter

## Author
**Paras** — built as part of the GCI World 2026 consulting program.
