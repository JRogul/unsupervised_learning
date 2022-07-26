from src import data_visualization
from src import data_proccesing
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
train_df = data_proccesing.train_df
data_visualization.plot_variance_treshold(train_df)
