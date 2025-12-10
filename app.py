import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import summarize_data, plot_numeric_data, plot_categorical_data

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="HR Analytics Dashboard", page_icon="👥", layout="wide")

st.title("👥 HR Analytics Dashboard")
st.markdown("Analyze employee demographics, retention, and salary trends.")

# --- 1. DATA LOADING ---
uploaded_file = st.file_uploader("Upload Employee.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # --- 2. SIDEBAR FILTERS ---
    st.sidebar.header("Filter Data")
    
    # Check if 'Department' exists to filter by it
    if 'Department' in df.columns:
        depts = df['Department'].unique().tolist()
        selected_depts = st.sidebar.multiselect("Select Department", depts, default=depts)
        if selected_depts:
            df = df[df['Department'].isin(selected_depts)]
            
    # Check if 'Gender' exists to filter by it
    if 'Gender' in df.columns:
        genders = df['Gender'].unique().tolist()
        selected_gender = st.sidebar.multiselect("Select Gender", genders, default=genders)
        if selected_gender:
            df = df[df['Gender'].isin(selected_gender)]

    st.success(f"Showing data for {len(df)} employees")

    # --- 3. KEY METRICS (KPIs) ---
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    # Total Employees
    col1.metric("Total Employees", len(df))
    
    # Attrition Rate (if column exists)
    if 'Attrition' in df.columns:
        attrition_count = df[df['Attrition'] == 'Yes'].shape[0]
        attrition_rate = (attrition_count / len(df)) * 100
        col2.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    else:
        col2.metric("Attrition Rate", "N/A")

    # Average Age (if column exists)
    if 'Age' in df.columns:
        avg_age = df['Age'].mean()
        col3.metric("Avg. Age", f"{avg_age:.1f} yrs")
        
    # Average Income (if column exists)
    # Note: Adjust column name 'MonthlyIncome' if your CSV has a different name
    if 'MonthlyIncome' in df.columns:
        avg_income = df['MonthlyIncome'].mean()
        col4.metric("Avg. Income", f"${avg_income:,.0f}")
    
    st.markdown("---")

    # --- 4. DETAILED ANALYSIS TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "📉 Attrition Analysis", "🔢 Raw Data"])
    
    with tab1:
        st.header("Distribution of Data")
        col_num, col_cat = st.columns(2)
        
        with col_num:
            st.subheader("Numeric Features")
            # Using your existing utils function
            try:
                fig_num = plot_numeric_data(df)
                st.pyplot(fig_num)
            except:
                st.info("No numeric data to plot.")
                
        with col_cat:
            st.subheader("Categorical Features")
            # Using your existing utils function
            try:
                fig_cat = plot_categorical_data(df)
                st.pyplot(fig_cat)
            except:
                st.info("No categorical data to plot.")

    with tab2:
        st.header("Who is leaving?")
        if 'Attrition' in df.columns and 'Department' in df.columns:
            # Custom plot for Attrition by Department
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(data=df, x='Department', hue='Attrition', ax=ax)
            ax.set_title("Attrition by Department")
            st.pyplot(fig)
        elif 'Attrition' not in df.columns:
            st.warning("This dataset does not have an 'Attrition' column.")
            
        if 'Attrition' in df.columns and 'Age' in df.columns:
             # Custom plot for Age vs Attrition
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(data=df, x='Age', hue='Attrition', kde=True, ax=ax)
            ax.set_title("Age Distribution by Attrition Status")
            st.pyplot(fig)

    with tab3:
        st.header("Raw Data")
        st.dataframe(df)

else:
    st.info("Please upload your Employee.csv file to begin.")
