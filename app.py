import streamlit as st
import pandas as pd
from stats_engine import DataProfiler
from plotter import DataPlotter
from report_gen import PDFReport

st.set_page_config(page_title="DataGov Tool", layout="wide")

st.title("📊 DataGov: The Automated Data Scientist")
st.markdown("Upload any CSV to auto-generate a **Statistical Health Report**, **Charts**, and **PDF Documentation**.")

uploaded_file = st.file_uploader("Upload your CSV file here", type="csv")

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.success(f"File Uploaded Successfully! Shape: {df.shape}")

    # Tabs for organization
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "📈 Deep Stats", "🎨 Visualizations", "📄 PDF Report"])

    # Initialize Engines
    profiler = DataProfiler(df)
    plotter = DataPlotter(df)

    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        st.subheader("Column Types & Missing Values")
        buffer = pd.DataFrame({
            'Type': df.dtypes.astype(str),
            'Missing Count': df.isnull().sum(),
            'Missing (%)': (df.isnull().sum() / len(df)) * 100
        })
        st.dataframe(buffer)

    # --- TAB 2: STATISTICS ---
    with tab2:
        st.subheader("Statistical Deep Dive")
        with st.spinner("Running Normality & Skewness Tests..."):
            summary_df = profiler.run_full_scan()
            st.dataframe(summary_df, use_container_width=True)
            
            st.info("💡 Note: If 'Is Normal?' is NO, use non-parametric tests for analysis.")

    # --- TAB 3: VISUALIZATIONS ---
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Correlation Matrix")
            # Filter numeric columns for correlation
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                fig_corr = plotter.plot_correlation_heatmap(numeric_df)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Not enough numerical columns for correlation.")

        with col2:
            st.subheader("Distribution Checker")
            numeric_cols = df.select_dtypes(include=['number']).columns
            selected_col = st.selectbox("Select Column to Inspect", numeric_cols)
            if selected_col:
                fig_dist = plotter.plot_distribution(selected_col)
                st.plotly_chart(fig_dist, use_container_width=True)

    # --- TAB 4: REPORT GENERATION ---
    with tab4:
        st.subheader("Export Documentation")
        if st.button("Generate Professional PDF Report"):
            with st.spinner("Compiling PDF..."):
                pdf_gen = PDFReport(df, summary_df)
                pdf_bytes = pdf_gen.create_pdf()
                
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name="DataGov_Report.pdf",
                    mime="application/pdf"

                )
