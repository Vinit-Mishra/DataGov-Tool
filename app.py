import streamlit as st 
import pandas as pd
import numpy as np
import io
import re
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Import the necessary classes from your existing files
from stats_engine import DataProfiler
from plotter import DataPlotter
from report_gen import PDFReport 

# --- HELPER FUNCTIONS ---
def validate_email(text):
    if not isinstance(text, str): return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, text))

def is_numeric_string(text):
    if isinstance(text, (int, float)): return True
    if not isinstance(text, str): return False
    return text.replace('.','',1).isdigit()

def detect_format_inconsistencies(df):
    inconsistent_indices = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]): continue
        valid_series = df[col].dropna().astype(str)
        if len(valid_series) == 0: continue
        total = len(valid_series)
        
        # Check Numeric
        numeric_matches = valid_series.apply(is_numeric_string)
        match_ratio = numeric_matches.sum() / total
        if 0.9 < match_ratio < 1.0:
            inconsistent_indices[f"{col} (Expected: Numeric)"] = df[~df[col].apply(is_numeric_string) & df[col].notna()].index.tolist()
            continue

        # Check Email
        email_matches = valid_series.apply(validate_email)
        match_ratio = email_matches.sum() / total
        if 'email' in str(col).lower() or match_ratio > 0.5:
             if match_ratio < 1.0:
                 inconsistent_indices[f"{col} (Expected: Email)"] = df[~df[col].apply(validate_email) & df[col].notna()].index.tolist()
    return inconsistent_indices


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Pro Data Analyst Assistant", page_icon="⭐", layout="wide")

st.title("Data Analyst Assistant")
st.markdown("Automated data preparation, advanced analysis, and basic predictive modeling.")

# --- 1. DATA LOADING ---
uploaded_file = st.file_uploader("Upload your Dataset (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("Please upload a CSV or Excel file to begin the automated analysis.")
    st.stop()

# Load Data
try:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success(f"Dataset '{uploaded_file.name}' loaded successfully!")
    st.markdown("---")
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Initialize State
if 'current_df' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
    st.session_state.current_df = df.copy()
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.cleaning_msg = "" # Store success messages here

# Run Profiler
profiler = DataProfiler(st.session_state.current_df)
summary_df = profiler.run_full_scan()


# --- SECTION 1: DIAGNOSIS (READ ONLY) ---
st.header("1️⃣ Data Diagnosis")

duplicates_count = st.session_state.current_df.duplicated().sum()
format_errors = detect_format_inconsistencies(st.session_state.current_df)
total_format_errors = sum([len(v) for v in format_errors.values()])

col_rows, col_cols, col_dupes, col_fmt = st.columns(4)
col_rows.metric("Rows", st.session_state.current_df.shape[0])
col_cols.metric("Columns", st.session_state.current_df.shape[1])
col_dupes.metric("Duplicates", duplicates_count)
col_fmt.metric("Format Errors", total_format_errors)

if duplicates_count > 0:
    st.warning(f"Found {duplicates_count} duplicate rows.")
    with st.expander("View Duplicates"):
        st.dataframe(st.session_state.current_df[st.session_state.current_df.duplicated()])

if total_format_errors > 0:
    st.error(f"Found {total_format_errors} format inconsistencies.")
    with st.expander("View Inconsistencies"):
        for col_msg, indices in format_errors.items():
            st.write(f"**{col_msg}**: {len(indices)} rows")

st.markdown("---")

# --- SECTION 2: STATISTICAL PROFILE ---
st.header("2️⃣ Statistical Profile")
with st.expander("📊 Detailed Summary", expanded=False):
    st.dataframe(summary_df)
st.markdown("---")


# --- SECTION 3: ACTION CENTER (CLEANING) ---
st.header("3️⃣ Fix & Clean Data")

# Display Last Action Message (e.g. "Dropped 5 rows. New Shape: ...")
if st.session_state.cleaning_msg:
    st.success(st.session_state.cleaning_msg)
    # Clear message so it doesn't stay forever (optional, keeping it for visibility)

# Identify Issues
cols_with_nan = summary_df[summary_df['Missing (%)'] > 0].index.tolist()
cols_with_outliers = summary_df[summary_df['Outliers (IQR)'] > 0].index.tolist()

# Layout: Use Tabs to keep it clean (Avoids "Two Options" clutter)
tab_nan, tab_outlier, tab_dupe, tab_fmt = st.tabs(["Missing Values", "Outliers", "Duplicates", "Format Errors"])

# 1. Missing Values Tab
with tab_nan:
    if cols_with_nan:
        col_names_str = [str(c) for c in cols_with_nan]
        st.error(f"⚠️ Columns with Missing Values: {', '.join(col_names_str)}")
        
        nan_option = st.selectbox("Choose Missing Value Action:", ["Select Action...", "Drop Rows", "Impute Mean", "Impute Median"], key="nan_sb")
        
        if st.button("Apply Missing Value Fix"):
            if nan_option == "Drop Rows":
                st.session_state.current_df.dropna(inplace=True)
                action_desc = "Dropped rows with missing values."
            elif "Impute" in nan_option:
                strategy = 'mean' if 'Mean' in nan_option else 'median'
                st.session_state.current_df = profiler.impute_data(imputation_strategy=strategy) 
                action_desc = f"Imputed missing values using {strategy}."
            else:
                action_desc = None

            if action_desc:
                # Update Message and Rerun
                new_shape = st.session_state.current_df.shape
                st.session_state.cleaning_msg = f"✅ {action_desc} | New Dataset Shape: {new_shape[0]} Rows, {new_shape[1]} Cols"
                st.rerun()
    else:
        st.success("✅ No missing values detected.")

# 2. Outliers Tab
with tab_outlier:
    if cols_with_outliers:
        col_names_str = [str(c) for c in cols_with_outliers]
        st.warning(f"⚠️ Columns with Outliers: {', '.join(col_names_str)}")
        
        outlier_action = st.selectbox("Choose Outlier Action:", ["Select Action...", "Cap Outliers (Winsorize)"], key="outlier_sb")
        
        if st.button("Apply Outlier Fix"):
            if outlier_action == "Cap Outliers (Winsorize)":
                st.session_state.current_df = profiler.cap_outliers() 
                new_shape = st.session_state.current_df.shape
                st.session_state.cleaning_msg = f"✅ Capped outliers. | New Dataset Shape: {new_shape[0]} Rows, {new_shape[1]} Cols"
                st.rerun()
    else:
        st.success("✅ No statistical outliers detected.")

# 3. Duplicates Tab
with tab_dupe:
    if duplicates_count > 0:
        st.warning(f"⚠️ {duplicates_count} Duplicate Rows Found")
        if st.button("Remove Duplicates"):
            st.session_state.current_df.drop_duplicates(inplace=True)
            new_shape = st.session_state.current_df.shape
            st.session_state.cleaning_msg = f"✅ Removed duplicates. | New Dataset Shape: {new_shape[0]} Rows, {new_shape[1]} Cols"
            st.rerun()
    else:
        st.success("✅ No duplicates detected.")

# 4. Format Errors Tab
with tab_fmt:
    if total_format_errors > 0:
        bad_fmt_cols = [str(c).split(" (Expected:")[0] for c in format_errors.keys()]
        st.error(f"⚠️ Columns with Format Issues: {', '.join(bad_fmt_cols)}")
        
        fmt_action = st.selectbox("Choose Format Action:", ["Select Action...", "Convert to NaN (Then Impute)", "Drop Rows"], key="fmt_sb")
        
        if st.button("Apply Format Fix"):
            action_desc = None
            if fmt_action == "Drop Rows":
                all_invalid_indices = []
                for idx_list in format_errors.values():
                    all_invalid_indices.extend(idx_list)
                st.session_state.current_df.drop(index=list(set(all_invalid_indices)), inplace=True)
                action_desc = "Dropped rows with invalid formats."
            elif "Convert" in fmt_action:
                for col_msg, indices in format_errors.items():
                    actual_col = col_msg.split(" (Expected:")[0]
                    st.session_state.current_df.loc[indices, actual_col] = np.nan
                action_desc = "Converted invalid formats to NaN."
            
            if action_desc:
                new_shape = st.session_state.current_df.shape
                st.session_state.cleaning_msg = f"✅ {action_desc} | New Dataset Shape: {new_shape[0]} Rows, {new_shape[1]} Cols"
                st.rerun()
    else:
        st.success("✅ No format inconsistencies detected.")

st.markdown("---")


# --- SECTION 4: PREVIEW ---
st.header("4️⃣ Data Preview (Current State)")
st.dataframe(st.session_state.current_df.head(10))
st.markdown("---")


# --- SECTION 5: VISUALIZATION ---
st.header("5️⃣ Visualization")
data_plotter = DataPlotter(st.session_state.current_df)

tab_dist, tab_corr, tab_bivariate = st.tabs(["Distributions", "Correlations", "Scatter/Box Plots"])

with tab_dist:
    col_to_plot = st.selectbox("Select Column:", st.session_state.current_df.columns, key="dist_col")
    if col_to_plot:
        try:
            st.plotly_chart(data_plotter.plot_distribution(col_to_plot), use_container_width=True)
        except: st.error("Cannot plot this column.")

with tab_corr:
    numeric_df = st.session_state.current_df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        st.plotly_chart(data_plotter.plot_correlation_heatmap(numeric_df), use_container_width=True)
    else: st.warning("No numeric columns.")

with tab_bivariate:
    c1, c2 = st.columns(2)
    col_x = c1.selectbox("X Axis:", st.session_state.current_df.columns, key="x_col")
    col_y = c2.selectbox("Y Axis:", st.session_state.current_df.columns, key="y_col")
    if col_x and col_y:
        try:
            plot_data = st.session_state.current_df.dropna(subset=[col_x, col_y])
            if pd.api.types.is_numeric_dtype(plot_data[col_x]) and pd.api.types.is_numeric_dtype(plot_data[col_y]):
                fig = px.scatter(plot_data, x=col_x, y=col_y, trendline="ols")
            else:
                fig = px.box(plot_data, x=col_x, y=col_y)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"Error: {e}")


# --- SECTION 6: MODELING ---
st.header("6️⃣ Predictive Modeling")
target_col = st.selectbox("Select Target (Binary):", st.session_state.current_df.columns)

if st.button("Run Logistic Regression"):
    try:
        data = st.session_state.current_df.copy().dropna()
        if data[target_col].nunique() != 2:
            st.error("Target must have exactly 2 unique values (Binary).")
        else:
            le = LabelEncoder()
            data[target_col] = le.fit_transform(data[target_col])
            data = pd.get_dummies(data, drop_first=True)
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
            st.code(classification_report(y_test, y_pred))
    except Exception as e:
        st.error(f"Error: {e}")

# --- SECTION 7: EXPORT ---
st.markdown("---")
if st.button("Generate Report"):
    try:
        clean_summary = summary_df.drop(columns=['Skewness_Numeric'], errors='ignore')
        pdf_gen = PDFReport(st.session_state.current_df, clean_summary)
        st.download_button("Download PDF", pdf_gen.create_pdf(), "report.pdf", "application/pdf")
    except Exception as e: st.error(f"PDF Error: {e}")
