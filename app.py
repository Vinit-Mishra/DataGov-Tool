import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

# Import the necessary classes from your existing files
from stats_engine import DataProfiler
from plotter import DataPlotter
from report_gen import PDFReport

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Data Analyst Assistant", page_icon="📈", layout="wide")

st.title("📈 Data Analyst Assistant (Beta)")
st.markdown("Your personal Junior Data Analyst for rapid data profiling and cleanup.")

# --- 1. DATA LOADING ---
uploaded_file = st.file_uploader("Upload your Dataset (.csv)", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin the automated analysis.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)
st.success(f"Dataset '{uploaded_file.name}' loaded successfully! Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
st.markdown("---")

# Initialize state to hold the (potentially cleaned) DataFrame
if 'current_df' not in st.session_state:
    st.session_state.current_df = df.copy()

# Initialize the profiler and run the analysis on the current state of the data
profiler = DataProfiler(st.session_state.current_df)
summary_df = profiler.run_full_scan()

# --- 2. DATA DESCRIPTION & STATISTICAL PROPERTIES ---
st.header("1️⃣ Data Description & Statistical Profile")

# 3. Whole Description of Data & 6. Statistical Property
with st.expander("📊 Full Statistical Summary (Click to expand)", expanded=False):
    st.subheader("Statistical Properties of All Columns")
    st.dataframe(summary_df)

# --- 3. ERROR IDENTIFICATION & RESOLUTION ---
st.header("2️⃣ Error Identification & Cleaning")

# 4. Identifies error in the dataset
missing_data_rows = summary_df[summary_df['Missing (%)'] > 0]
skewed_cols = summary_df[(summary_df['Type'].isin(['int64', 'float64'])) & (abs(summary_df['Skewness']) > 1)]

# Display Warnings
if not missing_data_rows.empty:
    st.error(f"🚨 **Critical Error: Missing Values** found in {len(missing_data_rows)} columns. This can skew analysis.")
    with st.expander("Details on Missing Data"):
        st.dataframe(missing_data_rows[['Column', 'Missing (%)']])
        
if not skewed_cols.empty:
    st.warning(f"⚠️ **Warning: Highly Skewed Data** found in {len(skewed_cols)} numeric columns. Consider transformation.")
    with st.expander("Details on Skewed Data"):
        st.dataframe(skewed_cols[['Column', 'Skewness', 'Recommendations']])
else:
    st.success("✅ Initial Data Scan complete. No critical missing values or highly skewed columns detected.")
    
# 5. Resolve error - Interactive Cleaning
if st.button("🪄 Auto-Clean Dataset (Drop NaNs)"):
    # Simple fix: drop rows with any NaN values
    initial_rows = st.session_state.current_df.shape[0]
    st.session_state.current_df.dropna(inplace=True)
    rows_dropped = initial_rows - st.session_state.current_df.shape[0]
    st.success(f"🧹 Cleaned dataset! Dropped **{rows_dropped}** rows with missing values.")
    # Re-run profiler on the cleaned data
    profiler = DataProfiler(st.session_state.current_df)
    summary_df = profiler.run_full_scan()
    st.dataframe(summary_df)
    
st.markdown("---")

# --- 4. VISUALIZATION AND INTERACTIVITY ---
st.header("3️⃣ Interactive Visualization")
data_plotter = DataPlotter(st.session_state.current_df)

tab_dist, tab_corr, tab_raw = st.tabs(["📊 Distributions", "🔥 Correlation Heatmap", "📝 Raw Data (Current)"])

with tab_dist:
    # 2. Options to see the visualization
    st.subheader("Feature Distribution Analysis")
    
    col_to_plot = st.selectbox(
        "Select a Column to Visualize:",
        st.session_state.current_df.columns
    )
    
    if col_to_plot:
        try:
            # Use the plot_distribution from plotter.py (Plotly)
            fig_dist = data_plotter.plot_distribution(col_to_plot)
            st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate plot for selected column. Error: {e}")

with tab_corr:
    st.subheader("Numeric Feature Correlation")
    numeric_df = st.session_state.current_df.select_dtypes(include=['number'])
    if numeric_df.empty:
        st.warning("No numeric columns found to plot correlation.")
    else:
        # Use the plot_correlation_heatmap from plotter.py (Plotly)
        fig_corr = data_plotter.plot_correlation_heatmap(numeric_df)
        st.plotly_chart(fig_corr, use_container_width=True)

with tab_raw:
    st.subheader("Current Working Dataset")
    st.dataframe(st.session_state.current_df)

# --- 5. REPORT GENERATION (ATTRACTIVE FEATURE) ---
st.markdown("---")
st.header("4️⃣ Export Analysis")

# Generate the PDF report
pdf_report_generator = PDFReport(st.session_state.current_df, summary_df)
pdf_output_bytes = pdf_report_generator.create_pdf()

st.download_button(
    label="⬇️ Download Full Statistical Report (PDF)",
    data=pdf_output_bytes,
    file_name="Automated_Data_Analysis_Report.pdf",
    mime="application/pdf"
)

st.info("The Data Analyst Assistant has completed its initial analysis. Use the tabs above to deep-dive.")
