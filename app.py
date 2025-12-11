import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Import the necessary classes from your existing files
from stats_engine import DataProfiler
from plotter import DataPlotter
from report_gen import PDFReport # Not used in this snippet but kept for final step

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Pro Data Analyst Assistant", page_icon="⭐", layout="wide")

st.title("⭐ Pro Data Analyst Assistant")
st.markdown("Automated data preparation, advanced analysis, and basic predictive modeling.")

# --- 1. DATA LOADING ---
uploaded_file = st.file_uploader("Upload your Dataset (.csv)", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin the automated analysis.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)
st.success(f"Dataset '{uploaded_file.name}' loaded successfully!")
st.markdown("---")

# Initialize state
if 'current_df' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
    st.session_state.current_df = df.copy()
    st.session_state.uploaded_file_name = uploaded_file.name
    
# Initialize the profiler and run the analysis
profiler = DataProfiler(st.session_state.current_df)
summary_df = profiler.run_full_scan()


# --- 2. DATA OVERVIEW & ERROR IDENTIFICATION ---
st.header("1️⃣ Data Overview & Error Identification")

col_rows, col_cols = st.columns(2)
col_rows.metric("Total Rows (Observations)", st.session_state.current_df.shape[0])
col_cols.metric("Total Columns (Features)", st.session_state.current_df.shape[1])

st.markdown("---")

missing_data_rows = summary_df[summary_df['Missing (%)'] > 0]
skewed_cols = summary_df[(summary_df['Type'].isin(['int64', 'float64'])) & (abs(summary_df['Skewness']) > 1)]
outlier_cols = summary_df[summary_df['Outliers (IQR)'] > 0]

st.subheader("🚨 Detected Data Quality Issues")

if not missing_data_rows.empty:
    st.error(f"**Missing Values:** Found in **{len(missing_data_rows)}** columns.")
if not outlier_cols.empty:
    st.warning(f"**Outliers:** Found in **{len(outlier_cols)}** numeric columns.")
if not skewed_cols.empty:
    st.warning(f"**High Skewness:** Found in **{len(skewed_cols)}** numeric columns.")

if missing_data_rows.empty and outlier_cols.empty and skewed_cols.empty:
     st.success("✅ Initial Data Scan: Data is clean and ready for analysis!")
    
st.markdown("---")


# --- 3. ERROR RESOLUTION (PRO CLEANING) ---
st.header("2️⃣ Advanced Error Resolution (Cleaning & Preprocessing)")
col_nan, col_outlier = st.columns(2)

with col_nan:
    st.subheader("Missing Value Handling")
    nan_option = st.selectbox(
        "Select Missing Value Strategy:",
        ["Do Nothing", "Drop Rows (Simple Clean)", "Impute Mean/Mode", "Impute Median/Mode"]
    )
    if st.button("Apply NAN Strategy"):
        if nan_option == "Drop Rows (Simple Clean)":
            st.session_state.current_df.dropna(inplace=True)
            st.success("🧹 Dropped all rows with missing values.")
        elif nan_option in ["Impute Mean/Mode", "Impute Median/Mode"]:
            strategy = 'mean' if 'Mean' in nan_option else 'median'
            st.session_state.current_df = profiler.impute_data(imputation_strategy=strategy)
            st.success(f"🧹 Imputed missing values using **{strategy.capitalize()}** for numeric and **Mode** for categorical.")
        
        # Rerun profiler to update state
        profiler = DataProfiler(st.session_state.current_df)
        summary_df = profiler.run_full_scan()


with col_outlier:
    st.subheader("Outlier Handling")
    outlier_action = st.selectbox(
        "Select Outlier Strategy:",
        ["Do Nothing", "Cap Outliers (Winsorize)"]
    )
    if st.button("Apply Outlier Strategy"):
        if outlier_action == "Cap Outliers (Winsorize)":
            st.session_state.current_df = profiler.cap_outliers()
            st.success("📐 Capped outliers using the IQR method (Winsorization).")
        
        # Rerun profiler to update state
        profiler = DataProfiler(st.session_state.current_df)
        summary_df = profiler.run_full_scan()

st.markdown("---")

# --- 4. STATISTICAL PROPERTIES & FULL SUMMARY ---
st.header("3️⃣ Full Statistical Profile")

with st.expander("📊 Detailed Statistical Summary (Click to expand)", expanded=False):
    st.subheader("Statistical Properties of All Columns")
    st.dataframe(summary_df)
    
st.markdown("---")
    
# --- 5. VISUALIZATION AND INTERACTIVITY (ADDED BIVARIATE TAB) ---
st.header("4️⃣ Interactive Visualization")
data_plotter = DataPlotter(st.session_state.current_df)

tab_dist, tab_corr, tab_bivariate, tab_raw = st.tabs(["📊 Distributions", "🔥 Correlation Heatmap", "🔗 Relationship Analysis", "📝 Raw Data (Current)"])

with tab_dist:
    # Existing Distribution plot logic
    st.subheader("Feature Distribution Analysis")
    col_to_plot = st.selectbox("Select a Column to Visualize:", st.session_state.current_df.columns, key="dist_col")
    if col_to_plot:
        try:
            fig_dist = data_plotter.plot_distribution(col_to_plot)
            st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate plot for selected column. Error: {e}")

with tab_corr:
    # Existing Correlation Heatmap plot logic
    st.subheader("Numeric Feature Correlation")
    numeric_df = st.session_state.current_df.select_dtypes(include=['number'])
    if numeric_df.empty:
        st.warning("No numeric columns found to plot correlation.")
    else:
        fig_corr = data_plotter.plot_correlation_heatmap(numeric_df)
        st.plotly_chart(fig_corr, use_container_width=True)

with tab_bivariate:
    # New Bivariate Plot Logic
    st.subheader("Compare Two Variables")
    cols = st.session_state.current_df.columns.tolist()
    col_x = st.selectbox("Select X-axis Column:", cols, key="x_col")
    col_y = st.selectbox("Select Y-axis Column:", cols, key="y_col")

    if col_x and col_y:
        try:
            is_x_num = pd.api.types.is_numeric_dtype(st.session_state.current_df[col_x])
            is_y_num = pd.api.types.is_numeric_dtype(st.session_state.current_df[col_y])

            if is_x_num and is_y_num:
                # Numeric vs Numeric -> Scatter Plot
                fig = px.scatter(st.session_state.current_df, x=col_x, y=col_y, trendline="ols", title=f"Scatter Plot: {col_x} vs {col_y}")
                st.plotly_chart(fig, use_container_width=True)
            elif is_y_num and not is_x_num:
                # Categorical vs Numeric -> Box Plot
                fig = px.box(st.session_state.current_df, x=col_x, y=col_y, title=f"Box Plot: {col_y} by {col_x}")
                st.plotly_chart(fig, use_container_width=True)
            elif is_x_num and not is_y_num:
                # Numeric vs Categorical -> Swapped Box Plot (or a warning)
                st.warning("For better visualization, please select the numeric column as the Y-axis.")
                fig = px.box(st.session_state.current_df, x=col_y, y=col_x, title=f"Box Plot: {col_x} by {col_y}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                 # Categorical vs Categorical -> Bar/Count Plot
                df_counts = st.session_state.current_df.groupby([col_x, col_y]).size().reset_index(name='Count')
                fig = px.bar(df_counts, x=col_x, y='Count', color=col_y, title=f"Stacked Bar Chart: {col_x} vs {col_y}")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Could not generate relationship plot. Please check data types. Error: {e}")

with tab_raw:
    # Existing Raw Data tab
    st.subheader("Current Working Dataset")
    st.dataframe(st.session_state.current_df)


# --- 6. SIMPLE PREDICTIVE MODELING (NEW PRO FEATURE) ---
st.header("5️⃣ Simple Predictive Modeling (Binary Classification)")

target_col = st.selectbox(
    "Select Target Column (for Binary Classification):",
    st.session_state.current_df.columns,
    index=0
)

if st.button("Run Logistic Regression"):
    try:
        data = st.session_state.current_df.copy()
        
        # 1. Check if target is binary
        if data[target_col].nunique() != 2:
            st.error("Model requires a **binary (2 unique values)** target column (e.g., Yes/No, 0/1, True/False).")
        else:
            # 2. Preprocessing: Drop NaNs (if any left) and handle Categorical Features
            data.dropna(inplace=True)
            
            # Identify feature columns
            feature_cols = [col for col in data.columns if col != target_col]
            
            # Apply Label Encoding to the target column (for consistency)
            le = LabelEncoder()
            data[target_col] = le.fit_transform(data[target_col])
            
            # One-Hot Encode the categorical features
            data = pd.get_dummies(data, columns=data[feature_cols].select_dtypes(include=['object', 'category']).columns)
            
            # Final feature set (excluding the original categorical columns now)
            X = data.drop(columns=[target_col])
            y = data[target_col]

            # 3. Model Training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 4. Display Results
            st.subheader(f"Logistic Regression Results (Target: {target_col})")
            st.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
            st.text("Classification Report:")
            st.code(classification_report(y_test, y_pred, target_names=le.classes_))
            
    except Exception as e:
        st.error(f"Modeling Error: {e}. Check if data is clean and has enough numeric features.")


# --- 7. REPORT GENERATION ---
st.markdown("---")
st.header("6️⃣ Export Analysis")

# Use the latest summary_df after cleaning
try:
    pdf_report_generator = PDFReport(st.session_state.current_df, summary_df)
    pdf_output_bytes = pdf_report_generator.create_pdf()

    st.download_button(
        label="⬇️ Download Full Statistical Report (PDF)",
        data=pdf_output_bytes,
        file_name="Pro_Data_Analysis_Report.pdf",
        mime="application/pdf"
    )
except Exception as e:
    st.error(f"Could not generate PDF report. Error: {e}")

st.info("The Pro Data Analyst Assistant is ready for deep-dive analysis and modeling.")
