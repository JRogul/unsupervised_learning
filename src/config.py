#usefull paths
MODELS_PATH = '../models/'
TRAIN_DATA = '../data/data.csv'
TRAIN_PREPROCESSED_DATA = '../data/preprocessed_data'
SUBMISSIOM_PATH = '../data/sample_submission.csv'
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

#todo dictionary of simpler models

#metrics used for unsupervised scoring
def usnupervised_metrics(preds, df, models_name, show=True):

     s = (models_name,
          silhouette_score(df, preds),
          calinski_harabasz_score(df, preds),
          davies_bouldin_score(df, preds))
     if show == True:
         print('{} : Silhouette : {:.2f} Calinski : {:.2f} Davies : {:.2f}'.format(s[0], s[1], s[2], s[3]))

     return s
