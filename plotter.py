import plotly.express as px
import plotly.figure_factory as ff

class DataPlotter:
    def __init__(self, df):
        self.df = df

    def plot_correlation_heatmap(self, numeric_df):
        corr = numeric_df.corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        return fig

    def plot_distribution(self, column_name):
        fig = px.histogram(self.df, x=column_name, marginal="box", nbins=30, title=f"Distribution of {column_name}")
        return fig