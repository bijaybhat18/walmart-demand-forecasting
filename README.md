# Walmart Retail Demand Forecasting & Inventory Optimization

A demand forecasting project that predicts weekly sales at the store/department
level and translates forecast accuracy into estimated inventory cost savings.

**[Live dashboard demo →](https://bijaybhat18-walmart-demand-forecasting-app-etblaw.streamlit.app/")**

---

## Business Problem

Retailers like Walmart must decide how much inventory to stock per store and
department each week. Overstocking ties up capital and increases holding costs;
understocking causes lost sales and empty shelves — especially costly during
high-demand periods like Thanksgiving and Christmas.

**Goal:** build a forecasting model that improves on a naive baseline, and
quantify the inventory cost impact of that improvement.

## Data

- **Source:** [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) (Kaggle)
- **Scope:** 421,570 weekly sales records across 45 stores and 81 departments, Feb 2010 – Oct 2012
- **Files used:** `train.csv` (weekly sales), `stores.csv` (store type & size)

## Approach

1. **Data cleaning** — identified and handled 1,285 negative sales records (return-driven weeks), validated extreme values against known retail calendar events (Black Friday, Christmas Eve) rather than treating them as errors, and flagged 164 sparse store/department combinations with limited history.
2. **Exploratory analysis** — quantified the holiday sales lift (+7% on average, though this understates the two largest spikes), found Type C stores are the most space-efficient despite lowest total revenue, and confirmed strong yearly seasonality.
3. **Feature engineering** — built lag features (1-week, 52-week), rolling averages, and explicit holiday-proximity features (`DaysToChristmas`) to address the finding that the raw `IsHoliday` flag understates the importance of the two biggest sales weeks.
4. **Baseline models** — naive last-week and seasonal-naive (same-week-last-year) benchmarks, evaluated with **Weighted MAE** (holiday weeks weighted 5x), matching the metric used in the original Kaggle competition.
5. **Model** — XGBoost regressor, achieving a **33.8% reduction in WMAE** versus the naive baseline ($1,093.92 vs. $1,653.46).
6. **Explainability** — SHAP analysis showed the model relies primarily on recent sales momentum (`Sales_RollingMean4`), with holiday flags providing a secondary correction — a finding that also surfaces a limitation (see below).
7. **Inventory cost simulation** — translated forecast error into estimated dollar cost under assumed holding/stockout cost rates, finding a **~30% reduction in estimated inventory costs**, a result that held stable (30.0–30.3%) across a range of cost assumptions.
8. **Dashboard** — interactive Streamlit app for exploring trends, per-store/department forecasts, model performance, and inventory cost sensitivity.

## Key Results

| Metric | Naive Baseline | XGBoost Model | Improvement |
|---|---|---|---|
| WMAE | $1,653.46 | $1,093.92 | 33.8% |
| Estimated inventory cost (test period) | $9,190,915 | $6,410,163 | 30.3% |

## Limitations & Future Work

- The test window (Aug–Oct 2012) does not include a Thanksgiving/Christmas period; a stronger validation would specifically hold out a holiday season.
- SHAP analysis suggests the model may underweight holiday effects for departments without a strong pre-holiday sales ramp — an interaction feature (rolling mean × days-to-Christmas) could address this.
- Inventory cost estimates rely on assumed holding/stockout cost rates rather than real business figures.
- `features.csv` (economic indicators: CPI, fuel price, markdowns) was not available in this data pull; incorporating it would likely improve holiday-period accuracy.

## Project Structure
```
├── app.py                          # Streamlit dashboard
├── notebooks/
│   └── walmart_forecasting.ipynb   # Full analysis: EDA → modeling → SHAP → cost simulation
├── requirements.txt
└── README.md
```

## Running Locally

```bash
git clone <your-repo-url>
cd walmart-demand-forecasting
pip install -r requirements.txt

# Download train.csv and stores.csv from the Kaggle competition page and place in data/
# Run notebooks/walmart_forecasting.ipynb to regenerate model.pkl and the CSVs the app needs

streamlit run app.py
```

## Tech Stack
Python, pandas, XGBoost, SHAP, PostgreSQL, Streamlit, Plotly
