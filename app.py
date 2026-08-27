"""
Walmart Retail Demand Forecasting & Inventory Optimization
Streamlit Dashboard

Run with: streamlit run app.py
Expects these files in the same folder:
    - model.pkl
    - test_set_with_preds.csv
    - full_data_with_features.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Walmart Demand Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Load data (cached so it doesn't reload on every interaction)
# ----------------------------
@st.cache_data
def load_data():
    full_data = pd.read_csv('full_data_with_features.csv', parse_dates=['Date'])
    test_set = pd.read_csv('test_set_with_preds.csv', parse_dates=['Date'])
    return full_data, test_set

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

full_data, test_set = load_data()
model = load_model()

# Baseline WMAE from your notebook analysis (hardcode your actual numbers here)
NAIVE_WMAE = 1653.46
XGB_WMAE = test_set['weight'].pipe(
    lambda w: np.average(np.abs(test_set['Weekly_Sales'] - test_set['XGB_Pred']), weights=w)
)

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Store/Dept Explorer", "Model Performance", "Inventory Cost Impact"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project:** Walmart Retail Demand Forecasting  
    **Model:** XGBoost Regressor  
    **Data:** Walmart Store Sales (Kaggle)
    """
)

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "Overview":
    st.title("Walmart Retail Demand Forecasting & Inventory Optimization")
    st.markdown(
        """
        A demand forecasting model built to help optimize inventory decisions —
        balancing the cost of overstocking against the cost of stockouts.
        """
    )

    # Key stats row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Weekly Sales Records", f"{len(full_data):,}")
    col2.metric("Stores", f"{full_data['Store'].nunique()}")
    col3.metric("Departments", f"{full_data['Dept'].nunique()}")
    col4.metric("Date Range", f"{full_data['Date'].min().date()} – {full_data['Date'].max().date()}")

    st.markdown("---")

    # Overall trend chart
    st.subheader("Total Weekly Sales Over Time")
    weekly_total = full_data.groupby('Date')['Weekly_Sales'].sum().reset_index()
    fig = px.line(weekly_total, x='Date', y='Weekly_Sales',
                  title="Clear seasonal spikes around Thanksgiving and Christmas each year")
    fig.update_layout(yaxis_title="Total Weekly Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Sales by Store Type")
        type_avg = full_data.groupby('Type')['Weekly_Sales'].mean().reset_index() if 'Type' in full_data.columns else None
        if type_avg is not None:
            fig2 = px.bar(type_avg, x='Type', y='Weekly_Sales', color='Type')
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Average Sales by Month")
        monthly_avg = full_data.groupby('Month')['Weekly_Sales'].mean().reset_index()
        fig3 = px.bar(monthly_avg, x='Month', y='Weekly_Sales')
        st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# PAGE 2: STORE/DEPT EXPLORER
# ============================================================
elif page == "Store/Dept Explorer":
    st.title("Store / Department Sales Explorer")
    st.markdown("Pick a store and department to see its historical sales pattern, with forecasted values overlaid where available.")

    col1, col2 = st.columns(2)
    with col1:
        store_choice = st.selectbox("Select Store", sorted(full_data['Store'].unique()))
    with col2:
        available_depts = sorted(full_data[full_data['Store'] == store_choice]['Dept'].unique())
        dept_choice = st.selectbox("Select Department", available_depts)

    series = full_data[
        (full_data['Store'] == store_choice) & (full_data['Dept'] == dept_choice)
    ].sort_values('Date')

    forecast_series = test_set[
        (test_set['Store'] == store_choice) & (test_set['Dept'] == dept_choice)
    ].sort_values('Date')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series['Date'], y=series['Weekly_Sales'],
                              mode='lines', name='Actual Sales'))

    if not forecast_series.empty:
        fig.add_trace(go.Scatter(x=forecast_series['Date'], y=forecast_series['XGB_Pred'],
                                  mode='lines+markers', name='Model Forecast',
                                  line=dict(dash='dash')))

    fig.update_layout(title=f"Store {store_choice}, Dept {dept_choice} — Weekly Sales",
                       xaxis_title="Date", yaxis_title="Weekly Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    if not forecast_series.empty:
        mae = np.mean(np.abs(forecast_series['Weekly_Sales'] - forecast_series['XGB_Pred']))
        st.info(f"Mean Absolute Error for this series (test period): ${mae:,.2f}")
    else:
        st.warning("No forecast available for this store/dept in the test period (likely a sparse series).")

# ============================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.title("Model Performance vs. Baseline")

    col1, col2, col3 = st.columns(3)
    col1.metric("Naive Baseline WMAE", f"${NAIVE_WMAE:,.2f}")
    col2.metric("XGBoost WMAE", f"${XGB_WMAE:,.2f}",
                delta=f"-{(NAIVE_WMAE - XGB_WMAE)/NAIVE_WMAE*100:.1f}%")
    col3.metric("Improvement", f"{(NAIVE_WMAE - XGB_WMAE)/NAIVE_WMAE*100:.1f}%")

    st.markdown(
        """
        **WMAE (Weighted Mean Absolute Error)** weights holiday weeks 5x more heavily than
        regular weeks — matching the metric used in the original Kaggle competition, since
        forecast accuracy during holidays matters most for the business.
        """
    )

    st.markdown("---")
    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({
        'feature': model.feature_names_in_ if hasattr(model, 'feature_names_in_') else model.get_booster().feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                 title="XGBoost Feature Importance")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **Key finding:** the 4-week rolling average of sales dominates model predictions,
        with holiday flags (Thanksgiving, Christmas) providing a secondary correction.
        Store-level attributes (size, type) contribute minimally once recency and
        calendar effects are accounted for.
        """
    )

    st.markdown("---")
    st.subheader("Predicted vs. Actual (Test Period)")
    fig2 = px.scatter(test_set, x='Weekly_Sales', y='XGB_Pred',
                       opacity=0.3, labels={'Weekly_Sales': 'Actual Sales', 'XGB_Pred': 'Predicted Sales'})
    max_val = max(test_set['Weekly_Sales'].max(), test_set['XGB_Pred'].max())
    fig2.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                               name='Perfect Prediction', line=dict(dash='dash', color='red')))
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# PAGE 4: INVENTORY COST IMPACT
# ============================================================
elif page == "Inventory Cost Impact":
    st.title("Inventory Cost Impact")
    st.markdown(
        """
        Translating forecast accuracy into estimated dollar impact, using assumed
        holding and stockout cost rates. Adjust the sliders below to see how the
        savings estimate changes under different cost assumptions.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        holding_cost = st.slider("Holding cost per $ of overstock", 0.01, 0.50, 0.10, 0.01)
    with col2:
        stockout_cost = st.slider("Stockout cost per $ of understock", 0.01, 0.75, 0.25, 0.01)

    def inventory_cost(actual, predicted, h_cost, s_cost):
        error = predicted - actual
        over = np.where(error > 0, error, 0)
        under = np.where(error < 0, -error, 0)
        return over * h_cost + under * s_cost

    cost_xgb = inventory_cost(test_set['Weekly_Sales'], test_set['XGB_Pred'], holding_cost, stockout_cost).sum()
    cost_naive = inventory_cost(test_set['Weekly_Sales'], test_set['Baseline_NaiveLastWeek'], holding_cost, stockout_cost).sum()
    savings = cost_naive - cost_xgb
    savings_pct = savings / cost_naive * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated Cost (XGBoost Model)", f"${cost_xgb:,.0f}")
    col2.metric("Estimated Cost (Naive Baseline)", f"${cost_naive:,.0f}")
    col3.metric("Estimated Savings", f"${savings:,.0f}", delta=f"{savings_pct:.1f}%")

    st.markdown("---")
    st.subheader("Sensitivity Across Scenarios")

    scenarios = [
        (0.05, 0.15, "Low cost scenario"),
        (0.10, 0.25, "Base case"),
        (0.15, 0.40, "High cost scenario"),
        (0.10, 0.50, "High stockout penalty"),
    ]
    rows = []
    for h, s, label in scenarios:
        c_xgb = inventory_cost(test_set['Weekly_Sales'], test_set['XGB_Pred'], h, s).sum()
        c_naive = inventory_cost(test_set['Weekly_Sales'], test_set['Baseline_NaiveLastWeek'], h, s).sum()
        rows.append({
            'Scenario': label,
            'Holding Cost': h,
            'Stockout Cost': s,
            'XGB Cost': c_xgb,
            'Naive Cost': c_naive,
            'Savings %': (c_naive - c_xgb) / c_naive * 100
        })
    scenario_df = pd.DataFrame(rows)
    st.dataframe(scenario_df.style.format({
        'XGB Cost': '${:,.0f}', 'Naive Cost': '${:,.0f}', 'Savings %': '{:.1f}%'
    }), use_container_width=True)

    st.success(
        f"Savings percentage remains stable (~{scenario_df['Savings %'].min():.0f}–"
        f"{scenario_df['Savings %'].max():.0f}%) across a range of plausible cost "
        f"assumptions, indicating the improvement is driven by genuine forecast "
        f"accuracy gains rather than sensitive cost parameters."
    )
