import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix


class DataAnalyzer:
    def __init__(self, data):
        self.data = data

    def run_analysis(self):
        descriptive_stats = self.descriptive_stats()
        correlation = self.correlation(visualize=True)
        covariance = self.covariance(visualize=True)
        self.pairplot_of_vars()
        return descriptive_stats, correlation, covariance

    def descriptive_stats(self):
        return self.data.describe()

    def correlation(self, visualize: bool = False):
        if visualize:
            sns.heatmap(self.data.corr(), annot=True, cmap="coolwarm")
        return self.data.corr()

    def pairplot_of_vars(self):
        sns.pairplot(self.data)

    def covariance(self, visualize: bool = False):
        if visualize:
            sns.heatmap(self.data.cov(), annot=True, cmap="coolwarm")
        return self.data.cov()
