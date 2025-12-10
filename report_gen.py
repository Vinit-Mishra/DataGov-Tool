from fpdf import FPDF
import pandas as pd

class PDFReport:
    def __init__(self, df, summary_df):
        self.df = df
        self.summary_df = summary_df

    def create_pdf(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="DataGov: Automated Dataset Report", ln=True, align='C')
        pdf.ln(10)

        # Basic Info
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Total Rows: {self.df.shape[0]} | Total Columns: {self.df.shape[1]}", ln=True)
        pdf.ln(10)

        # Statistical Summary Table
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Statistical Anomalies & Recommendations:", ln=True)
        pdf.set_font("Arial", size=10)
        
        # Iterate through summary and print important rows
        for index, row in self.summary_df.iterrows():
            # --- FIX: Remove Emojis for PDF Compatibility ---
            raw_normal = str(row.get('Is Normal?', '-'))
            
            # Convert "✅ Yes" to just "YES"
            if "Yes" in raw_normal:
                clean_normal = "YES"
            # Convert "❌ No" to just "NO"
            elif "No" in raw_normal:
                clean_normal = "NO"
            else:
                clean_normal = "-"
            
            # Create the line of text
            line = f"Col: {row['Column']} | Skew: {row.get('Skewness', '-') } | Normal: {clean_normal}"
            
            # This ensures no hidden weird characters crash the PDF
            clean_line = line.encode('latin-1', 'replace').decode('latin-1')
            
            pdf.cell(0, 10, txt=clean_line, ln=True)
        
        return pdf.output(dest='S').encode('latin-1')
