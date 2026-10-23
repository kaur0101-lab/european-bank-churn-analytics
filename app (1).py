import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="European Banking Churn Analytics",
    layout="wide"
)

# Custom header layout
st.markdown("<h1 style='color: #1f77b4;'>Customer Segmentation & Churn Pattern Analytics</h1>", unsafe_allow_html=True)
st.markdown("**Unified Mentor | European Central Bank Decision Support Dashboard**")
st.markdown("---")

# 2. Load and Automatically Process 'European_Bank.csv'
@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv('European_Bank.csv')
    except FileNotFoundError:
        st.error("'European_Bank.csv' not found! Please make sure your CSV file is uploaded in the same folder.")
        return pd.DataFrame()
    
    # Clean unnecessary unique identifier columns if present
    if 'RowNumber' in df.columns:
        df = df.drop(columns=['RowNumber'])
    if 'Surname' in df.columns:
        df = df.drop(columns=['Surname'])
        
    # Feature Engineering / Required Segmentations
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 29, 45, 60, np.inf], labels=['<30', '30–45', '46–60', '60+'])
    df['CreditScoreBand'] = pd.cut(df['CreditScore'], bins=[0, 599, 750, np.inf], labels=['Low', 'Medium', 'High'])
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=[-1, 2, 7, np.inf], labels=['New', 'Mid-term', 'Long-term'])
    df['BalanceSegment'] = pd.cut(df['Balance'], bins=[-1, 0, 100000, np.inf], labels=['Zero-balance', 'Low-balance', 'High-balance'])
    df['ChurnStatus'] = df['Exited'].map({0: 'Retained', 1: 'Churned'})
    return df

df = load_and_process_data()

if not df.empty:
    # 3. Sidebar User Capabilities (Segment Filters)
    st.sidebar.header("Segment Filters")
    selected_geo = st.sidebar.multiselect("Geography", options=df['Geography'].unique(), default=df['Geography'].unique())
    selected_gender = st.sidebar.multiselect("Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
    selected_age = st.sidebar.multiselect("Age Group", options=df['AgeGroup'].dropna().unique(), default=df['AgeGroup'].dropna().unique())
    selected_balance = st.sidebar.multiselect("Balance Segment", options=df['BalanceSegment'].dropna().unique(), default=df['BalanceSegment'].dropna().unique())

    # Dynamic Filtering Application
    filtered_df = df[
        (df['Geography'].isin(selected_geo)) &
        (df['Gender'].isin(selected_gender)) &
        (df['AgeGroup'].isin(selected_age)) &
        (df['BalanceSegment'].isin(selected_balance))
    ]

    # 4. Dynamic KPI Updates Module
    st.subheader("📊 Dynamic Key Performance Indicators (KPIs)")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    total_cust = len(filtered_df)
    churned_cust = filtered_df['Exited'].sum()
    churn_rate = (churned_cust / total_cust * 100) if total_cust > 0 else 0
    high_val_lost = filtered_df[(filtered_df['Exited'] == 1) & (filtered_df['Balance'] > 100000)]['Balance'].sum()
    active_ratio = (filtered_df['IsActiveMember'].mean() * 100) if total_cust > 0 else 0

    kpi1.metric("Total Customers", f"{total_cust:,}")
    kpi2.metric("Total Churned", f"{churned_cust:,}")
    kpi3.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
    kpi4.metric("High-Value Capital Lost", f"€{high_val_lost/1e6:.2f}M")
    kpi5.metric("Active Member %", f"{active_ratio:.1f}%")

    st.markdown("---")

    # Consistent color palette for charts
    color_map = {'Retained': '#2b5c8f', 'Churned': '#d9534f'}

    # 5. Core Modules Organized via Professional Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Overall Churn Summary", 
        "🌍 Geography-wise Churn", 
        "📈 Age & Tenure Comparison", 
        "💎 High-Value Churn Explorer"
    ])

    with tab1:
        st.subheader("Overall Churn Summary & Product Distribution")
        c1, c2 = st.columns(2)
        with c1:
            fig_overall = px.pie(filtered_df, names="ChurnStatus", title="Proportion of Churned vs Retained Customers",
                                 color="ChurnStatus", color_discrete_map=color_map, hole=0.4)
            st.plotly_chart(fig_overall, use_container_width=True)
        with c2:
            fig_prod = px.histogram(filtered_df, x="NumOfProducts", color="ChurnStatus", barmode="group",
                                   title="Churn Distribution by Number of Products Held", color_discrete_map=color_map)
            st.plotly_chart(fig_prod, use_container_width=True)

    with tab2:
        st.subheader("Geography-wise Churn Visualization & Risk Index")
        c3, c4 = st.columns(2)
        with c3:
            fig_geo = px.histogram(filtered_df, x="Geography", color="ChurnStatus", barmode="group",
                                  title="Regional Churn Breakdown (France, Spain, Germany)", color_discrete_map=color_map)
            st.plotly_chart(fig_geo, use_container_width=True)
        with c4:
            geo_rate = filtered_df.groupby('Geography', observed=False)['Exited'].mean().reset_index()
            geo_rate['Churn Rate (%)'] = geo_rate['Exited'] * 100
            fig_geo_rate = px.bar(geo_rate, x="Geography", y="Churn Rate (%)", color="Geography",
                                  title="Geographic Risk Index (Churn Rate %)", text_auto='.1f')
            st.plotly_chart(fig_geo_rate, use_container_width=True)

    with tab3:
        st.subheader("Age & Tenure Churn Comparison")
        c5, c6 = st.columns(2)
        with c5:
            fig_age = px.histogram(filtered_df, x="AgeGroup", color="ChurnStatus", barmode="group",
                                  title="Churn Comparison Across Age Segments", color_discrete_map=color_map)
            st.plotly_chart(fig_age, use_container_width=True)
        with c6:
            fig_tenure = px.histogram(filtered_df, x="TenureGroup", color="ChurnStatus", barmode="group",
                                     title="Churn Comparison Across Tenure Groups", color_discrete_map=color_map)
            st.plotly_chart(fig_tenure, use_container_width=True)

    with tab4:
        st.subheader("High-Value Customer Churn Explorer")
        c7, c8 = st.columns(2)
        with c7:
            bal_rate = filtered_df.groupby('BalanceSegment', observed=False)['Exited'].mean().reset_index()
            bal_rate['Churn Rate (%)'] = bal_rate['Exited'] * 100
            fig_bal = px.bar(bal_rate, x="BalanceSegment", y="Churn Rate (%)", color="BalanceSegment",
                             title="Churn Rate by Balance Segment", text_auto='.1f')
            st.plotly_chart(fig_bal, use_container_width=True)
        with c8:
            fig_scatter = px.scatter(filtered_df, x="EstimatedSalary", y="Balance", color="ChurnStatus",
                                    opacity=0.6, title="Estimated Salary vs Balance (High-Value Risk)", color_discrete_map=color_map)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # 6. Drill-Down Views Capability
    st.markdown("---")
    with st.expander("🔍 Click to Expand: Detailed Customer Drill-Down View (Raw Filtered Dataset)"):
        st.markdown("Examine individual records instantly based on your active sidebar filter selections:")
        st.dataframe(filtered_df[['CustomerId', 'Geography', 'Gender', 'Age', 'CreditScore', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'ChurnStatus']], use_container_width=True)
