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

# --- HELPER FUNCTIONS FOR PATTERN MATCHING ---
def validate_email(text):
    if not isinstance(text, str): return False
    # Standard robust Email Regex
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, text))

def validate_url(text):
    if not isinstance(text, str): return False
    pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return bool(re.match(pattern, text))

def is_numeric_string(text):
    if isinstance(text, (int, float)): return True
    if not isinstance(text, str): return False
    return text.replace('.','',1).isdigit()

def detect_format_inconsistencies(df):
    """
    Scans for columns that are mostly one format but have some outliers.
    Returns a dictionary of {column_name: [list_of_invalid_indices]}
    """
    inconsistent_indices = {}
    
    for col in df.columns:
        # Skip purely numeric columns (already handled by IQR)
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Get non-null values
        valid_series = df[col].dropna().astype(str)
        if len(valid_series) == 0: continue

        total = len(valid_series)
        
        # 1. Check for Mixed Numeric (e.g. ["100", "200", "Two"])
        numeric_matches = valid_series.apply(is_numeric_string)
        match_ratio = numeric_matches.sum() / total
        if 0.9 < match_ratio < 1.0: # If >90% are numbers, the rest are outliers
            inconsistent_indices[f"{col} (Expected: Numeric)"] = df[~df[col].apply(is_numeric_string) & df[col].notna()].index.tolist()
            continue

        # 2. Check for Emails (if column name suggests it or content looks like it)
        email_matches = valid_series.apply(validate_email)
        match_ratio = email_matches.sum() / total
        # Heuristic: If col name has 'email' or >50% content is email
        if 'email' in col.lower() or match_ratio > 0.5:
             if match_ratio < 1.0: # If not perfect, flag the rest
                 inconsistent_indices[f"{col} (Expected: Email)"] = df[~df[col].apply(validate_email) & df[col].notna()].index.tolist()
                 continue

        # 3. Check for Dates (heuristically via pd.to_datetime)
        # We try converting. If mostly success, flag failures.
        try:
            dates = pd.to_datetime(valid_series, errors='coerce')
            date_ratio = dates.notna().sum() / total
            if 0.9 < date_ratio < 1.0:
                 inconsistent_indices[f"{col} (Expected: Date)"] = df[dates.isna() & df[col].notna()].index.tolist()
        except:
            pass

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

# --- MULTI-FORMAT LOADING LOGIC ---
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

# Initialize state
if 'current_df' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
    st.session_state.current_df = df.copy()
    st.session_state.uploaded_file_name = uploaded_file.name
    
# Initialize profiler
profiler = DataProfiler(st.session_state.current_df)
summary_df = profiler.run_full_scan()


# --- SECTION 1: DATA OVERVIEW ---
st.header("1️⃣ Data Overview & Error Identification")

duplicates_count = st.session_state.current_df.duplicated().sum()

# Run the New Format Checker
format_errors = detect_format_inconsistencies(st.session_state.current_df)
total_format_errors = sum([len(v) for v in format_errors.values()])

col_rows, col_cols, col_dupes, col_fmt = st.columns(4)
col_rows.metric("Total Rows", st.session_state.current_df.shape[0])
col_cols.metric("Total Columns", st.session_state.current_df.shape[1])
col_dupes.metric("Duplicate Rows", duplicates_count)
col_fmt.metric("Format/Pattern Errors", total_format_errors)

# Show Duplicates
if duplicates_count > 0:
    st.warning(f"⚠️ Found {duplicates_count} duplicate rows.")
    with st.expander("👀 View Duplicate Rows", expanded=False):
        st.dataframe(st.session_state.current_df[st.session_state.current_df.duplicated()])

# Show Format Errors
if total_format_errors > 0:
    st.error(f"🚨 Found {total_format_errors} values that don't match the column format (e.g., Invalid Emails, Text in Number columns).")
    with st.expander("👀 View Pattern Mismatches (Format Outliers)", expanded=True):
        for col_msg, indices in format_errors.items():
            st.write(f"**Column: {col_msg}** - {len(indices)} issues found.")
            st.dataframe(st.session_state.current_df.loc[indices])

st.markdown("---")

# Existing Statistical Checks
summary_df['Skewness_Numeric'] = pd.to_numeric(summary_df['Skewness'], errors='coerce') 
missing_data_rows = summary_df[summary_df['Missing (%)'] > 0]
outlier_cols = summary_df[summary_df['Outliers (IQR)'] > 0]

if missing_data_rows.empty and outlier_cols.empty and duplicates_count == 0 and total_format_errors == 0:
     st.success("✅ Initial Data Scan: Data is clean and ready for analysis!")
    
st.markdown("---")


# --- SECTION 2: STATISTICAL PROFILE ---
st.header("2️⃣ Full Statistical Profile")
with st.expander("📊 Detailed Statistical Summary (Click to expand)", expanded=False):
    st.subheader("Statistical Properties of All Columns")
    display_df = summary_df.drop(columns=['Skewness_Numeric'], errors='ignore')
    st.dataframe(display_df)
st.markdown("---")


# --- SECTION 3: ERROR RESOLUTION ---
st.header("3️⃣ Advanced Error Resolution (Cleaning & Preprocessing)")
col_nan, col_outlier, col_clean_dupe = st.columns(3)

# 1. Missing Values
with col_nan:
    st.subheader("Missing Values")
    nan_option = st.selectbox("Strategy:", ["Do Nothing", "Drop Rows", "Impute Mean", "Impute Median"])
    if st.button("Apply Missing Strategy"):
        if nan_option == "Drop Rows":
            st.session_state.current_df.dropna(inplace=True)
            st.success("Dropped rows with missing values.")
        elif "Impute" in nan_option:
            strategy = 'mean' if 'Mean' in nan_option else 'median'
            st.session_state.current_df = profiler.impute_data(imputation_strategy=strategy) 
            st.success(f"Imputed using {strategy}.")
        st.rerun()

# 2. Outliers (Statistical)
with col_outlier:
    st.subheader("Statistical Outliers")
    outlier_action = st.selectbox("Strategy:", ["Do Nothing", "Cap (Winsorize)"])
    if st.button("Apply Outlier Strategy"):
        if outlier_action == "Cap (Winsorize)":
            st.session_state.current_df = profiler.cap_outliers() 
            st.success("Capped outliers.")
        st.rerun()

# 3. Duplicates
with col_clean_dupe:
    st.subheader("Duplicates")
    dupe_action = st.selectbox("Strategy:", ["Do Nothing", "Remove Duplicates"])
    if st.button("Apply Duplicate Strategy"):
        if dupe_action == "Remove Duplicates":
            st.session_state.current_df.drop_duplicates(inplace=True)
            st.success("Removed duplicate rows.")
        st.rerun()

# 4. NEW: Format/Pattern Cleaning
st.subheader("🔧 Format Inconsistency Cleaning")
if total_format_errors > 0:
    st.warning("Detected pattern mismatches (e.g., 'abc' in a numeric column, or invalid emails).")
    col_fmt_act, col_fmt_btn = st.columns([3, 1])
    with col_fmt_act:
        fmt_action = st.selectbox("Choose Action for Invalid Formats:", ["Do Nothing", "Convert Invalid to NaN (Then Impute)", "Drop Rows with Invalid Formats"])
    with col_fmt_btn:
        st.write("") # Spacer
        st.write("") 
        if st.button("Fix Formats"):
            if fmt_action == "Drop Rows with Invalid Formats":
                all_invalid_indices = []
                for idx_list in format_errors.values():
                    all_invalid_indices.extend(idx_list)
                initial_count = st.session_state.current_df.shape[0]
                st.session_state.current_df.drop(index=list(set(all_invalid_indices)), inplace=True)
                st.success(f"Dropped {initial_count - st.session_state.current_df.shape[0]} rows with invalid formats.")
                st.rerun()
            elif "Convert" in fmt_action:
                # We iterate and set bad values to NaN
                for col_msg, indices in format_errors.items():
                    # Extract original column name from the message string "ColName (Expected: Type)"
                    actual_col = col_msg.split(" (Expected:")[0]
                    st.session_state.current_df.loc[indices, actual_col] = np.nan
                st.success("Invalid values converted to NaN. You can now use the Missing Values tool to impute them.")
                st.rerun()
else:
    st.success("No format inconsistencies detected.")

st.markdown("---")


# --- SECTION 4: POST-CLEANING PREVIEW ---
st.header("4️⃣ Data Preview (Post-Cleaning)")
st.dataframe(st.session_state.current_df.head(10))
st.markdown("---")


# --- SECTION 5: VISUALIZATION ---
st.header("5️⃣ Interactive Visualization")
data_plotter = DataPlotter(st.session_state.current_df)

tab_dist, tab_corr, tab_bivariate, tab_raw = st.tabs(["📊 Distributions", "🔥 Correlation Heatmap", "🔗 Relationship Analysis", "📝 Raw Data"])

with tab_dist:
    st.subheader("Feature Distribution Analysis")
    col_to_plot = st.selectbox("Select a Column to Visualize:", st.session_state.current_df.columns, key="dist_col")
    if col_to_plot:
        try:
            fig_dist = data_plotter.plot_distribution(col_to_plot)
            st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

with tab_corr:
    st.subheader("Numeric Feature Correlation")
    numeric_df = st.session_state.current_df.select_dtypes(include=['number'])
    if numeric_df.empty:
        st.warning("No numeric columns found.")
    else:
        fig_corr = data_plotter.plot_correlation_heatmap(numeric_df)
        st.plotly_chart(fig_corr, use_container_width=True)

with tab_bivariate:
    st.subheader("Compare Two Variables")
    cols = st.session_state.current_df.columns.tolist()
    col_x = st.selectbox("Select X-axis Column:", cols, key="x_col")
    col_y = st.selectbox("Select Y-axis Column:", cols, key="y_col")

    if col_x and col_y:
        try:
            plot_data = st.session_state.current_df.dropna(subset=[col_x, col_y])
            if pd.api.types.is_numeric_dtype(plot_data[col_x]) and pd.api.types.is_numeric_dtype(plot_data[col_y]):
                fig = px.scatter(plot_data, x=col_x, y=col_y, trendline="ols", title=f"Scatter Plot: {col_x} vs {col_y}")
            else:
                fig = px.box(plot_data, x=col_x, y=col_y, title=f"Relationship: {col_y} by {col_x}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Plotting Error: {e}")

with tab_raw:
    st.subheader("Current Working Dataset")
    st.dataframe(st.session_state.current_df)


# --- SECTION 6: MODELING ---
st.header("6️⃣ Simple Predictive Modeling (Binary Classification)")

target_col = st.selectbox("Select Target Column:", st.session_state.current_df.columns)

if st.button("Run Logistic Regression"):
    try:
        data = st.session_state.current_df.copy()
        if data[target_col].nunique() != 2:
            st.error("Model requires a **binary** target column.")
        else:
            with st.spinner("Training Model..."):
                data.dropna(inplace=True)
                le = LabelEncoder()
                data[target_col] = le.fit_transform(data[target_col])
                
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
                
                X = data.drop(columns=[target_col])
                y = data[target_col]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
                st.text("Classification Report:")
                st.code(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))
    except Exception as e:
        st.error(f"Modeling Error: {e}")


# --- SECTION 7: REPORT ---
st.markdown("---")
st.header("7️⃣ Export Analysis")

try:
    clean_summary = summary_df.drop(columns=['Skewness_Numeric'], errors='ignore')
    pdf_
