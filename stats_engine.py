import pandas as pd
import numpy as np
from scipy.stats import shapiro, kstest

class DataProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.stats_summary = []

    def _check_normality(self, data):
        clean_data = data.dropna()
        if len(clean_data) < 3: return "N/A", 0, False
        
        if len(clean_data) < 5000:
            stat, p_value = shapiro(clean_data)
            test_name = "Shapiro-Wilk"
        else:
            stat, p_value = kstest((clean_data - clean_data.mean())/clean_data.std(), 'norm')
            test_name = "Kolmogorov-Smirnov"
            
        return test_name, round(p_value, 4), p_value > 0.05

    def run_full_scan(self):
        for col in self.df.columns:
            col_data = self.df[col]
            dtype = str(col_data.dtype)
            
            col_stats = {
                "Column": col,
                "Type": dtype,
                "Missing (%)": round((col_data.isnull().sum() / len(self.df)) * 100, 2),
                "Unique Values": col_data.nunique()
            }

            if pd.api.types.is_numeric_dtype(col_data):
                col_stats["Mean"] = round(col_data.mean(), 2)
                col_stats["Skewness"] = round(col_data.skew(), 2)
                test_name, p_val, is_normal = self._check_normality(col_data)
                col_stats["Is Normal?"] = "✅ Yes" if is_normal else "❌ No"
                col_stats["Recommendations"] = "Log Transform" if abs(col_stats["Skewness"]) > 1 else "None"
            else:
                col_stats["Mean"] = "-"
                col_stats["Is Normal?"] = "-"
                col_stats["Recommendations"] = "Label Encode"

            self.stats_summary.append(col_stats)

        return pd.DataFrame(self.stats_summary)