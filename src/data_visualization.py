import pandas as pd

from src import data_proccesing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import umap
plt.matplotlib.use("Qt5Agg")
#load the data
train_scaled = data_proccesing.train_scaled
train_df = data_proccesing.train_df


#simple approach for feature selection
#separates features with high variance from same output features
def plot_variance_treshold(df):
    feature_selector = VarianceThreshold(threshold=1.5)
    feature_selector.fit_transform((train_df.drop('id', axis=1)))
    plt.figure(figsize=(15, 10))
    sns.barplot(x=feature_selector.variances_, y=train_df.drop('id', axis=1).columns, orient='h')
    plt.axvline(x=1.5, label='Treshold')
    plt.xlabel('variance')
    plt.legend()
    plt.show()


#plots distribution of every feature
def plot_distribution(df):
    fig = plt.figure(figsize=(16, 10))
    for count, f in enumerate(df.columns):
        plt.subplot(6,5,count+1)
        sns.histplot(x=df[f])
        plt.title('Feature {}'.format(f))
        plt.xlabel('')
    fig.tight_layout()
    plt.show()

#todo bigger comment about umap
def umapp(train_scaled,preds,show=True):
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(train_scaled)
    if show==True:
        df = pd.DataFrame({'x' : embedding[:,0],
                           'y' : embedding[:,1],
                           'clusters' : preds})
        sns.scatterplot(data=df, x='x', y='y', hue='clusters', palette='Spectral')
        plt.show()
