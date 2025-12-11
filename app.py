# --- 2. DATA OVERVIEW & ERROR IDENTIFICATION ---
st.header("1️⃣ Data Overview & Error Identification")

# 1. Show rows and columns
col_rows, col_cols = st.columns(2)
col_rows.metric("Total Rows (Observations)", st.session_state.current_df.shape[0])
col_cols.metric("Total Columns (Features)", st.session_state.current_df.shape[1])

st.markdown("---")

# ----------------------------------------------------
# ⬇️ START OF FIX FOR TypeError ⬇️
# ----------------------------------------------------

# Convert Skewness to numeric, coercing the '-' to NaN for safe calculation
summary_df['Skewness_Numeric'] = pd.to_numeric(summary_df['Skewness'], errors='coerce') 

# 4. Identifies error in the dataset
missing_data_rows = summary_df[summary_df['Missing (%)'] > 0]
outlier_cols = summary_df[summary_df['Outliers (IQR)'] > 0]

# Use the new 'Skewness_Numeric' column for safe calculation of skewed columns
skewed_cols = summary_df[
    (summary_df['Type'].isin(['int64', 'float64'])) & 
    (abs(summary_df['Skewness_Numeric']) > 1)
]

# ----------------------------------------------------
# ⬆️ END OF FIX FOR TypeError ⬆️
# ----------------------------------------------------


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
# ... (rest of app.py continues) ...
