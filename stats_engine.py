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

    def check_outliers(self, data):
        """Checks for outliers using the IQR method."""
        if not pd.api.types.is_numeric_dtype(data):
            return 0, False
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = data[(data < lower_bound) | (data > upper_bound)].count()
        return outliers_count, outliers_count > 0

    def run_full_scan(self):
        self.stats_summary = [] # Reset summary
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
                outliers_count, has_outliers = self.check_outliers(col_data)
                col_stats["Mean"] = round(col_data.mean(), 2)
                col_stats["Skewness"] = round(col_data.skew(), 2)
                col_stats["Outliers (IQR)"] = outliers_count
                
                test_name, p_val, is_normal = self._check_normality(col_data)
                col_stats["Is Normal?"] = "✅ Yes" if is_normal else "❌ No"
                
                # Recommendations based on errors
                if col_stats["Missing (%)"] > 0:
                    col_stats["Recommendations"] = "Impute Mean/Median"
                elif has_outliers:
                    col_stats["Recommendations"] = "Cap Outliers (Winsorize)"
                elif abs(col_stats["Skewness"]) > 1:
                    col_stats["Recommendations"] = "Log/Power Transform"
                else:
                    col_stats["Recommendations"] = "None"
            else:
                # Non-numeric stats
                col_stats["Mean"] = "-"
                col_stats["Skewness"] = "-"
                col_stats["Outliers (IQR)"] = 0
                col_stats["Is Normal?"] = "-"
                col_stats["Recommendations"] = "Label/OneHot Encode" if col_stats["Missing (%)"] == 0 else "Impute Mode"

            self.stats_summary.append(col_stats)

        return pd.DataFrame(self.stats_summary)
        
    def impute_data(self, imputation_strategy='mean'):
        """Performs simple imputation."""
        new_df = self.df.copy()
        for col in new_df.columns:
            if new_df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(new_df[col]):
                    if imputation_strategy == 'mean':
                        new_df[col].fillna(new_df[col].mean(), inplace=True)
                    elif imputation_strategy == 'median':
                        new_df[col].fillna(new_df[col].median(), inplace=True)
                else: # Categorical/Object
                    new_df[col].fillna(new_df[col].mode()[0], inplace=True)
        return new_df

    def cap_outliers(self):
        """Caps outliers using the IQR method (Winsorization)."""
        new_df = self.df.copy()
        for col in new_df.columns:
            if pd.api.types.is_numeric_dtype(new_df[col]):
                Q1 = new_df[col].quantile(0.25)
                Q3 = new_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                new_df[col] = np.where(new_df[col] > upper_bound, upper_bound, new_df[col])
                new_df[col] = np.where(new_df[col] < lower_bound, lower_bound, new_df[col])
        return new_df
