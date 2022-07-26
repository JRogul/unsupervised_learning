import numpy as np
import sklearn
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer
from sklearn.mixture import BayesianGaussianMixture
from sklego.mixture import BayesianGMMClassifier
from src import data_proccesing
from src import config
import warnings
warnings.filterwarnings('ignore')

def train(model, df, models_name):
    model.fit(df)
    joblib.dump(model,config.MODELS_PATH + models_name)


def predict(model, df):
    predictions_proba = model.predict_proba(df)
    predictions = np.argmax(predictions_proba, axis=1)
    X = np.array(df)
    y = np.array(predictions)
    return X, y
def clasiffier(model,X,y, models_name):
    model.fit(X, y)
    predict = model.predict(X)
    joblib.dump(model, config.MODELS_PATH + models_name)
    return predict
def submission(predictions):
    sub_df = pd.read_csv(config.SUBMISSIOM_PATH)
    sub_df['Predicted'] = predictions
    sub_df.to_csv('submission', index=False)



bgm = BayesianGaussianMixture(
    n_components = 7,
    random_state = 0,
    covariance_type='full',
    )

bgmm_classifier = BayesianGMMClassifier(
    n_components=7,
    random_state=0,
    tol=1e-3,
    covariance_type='full',
    max_iter=200,
    n_init=3,
    init_params='k-means++'
    )

train(bgm, data_proccesing.train_scaled, 'BayesianGaussianMixture')
X, y = predict(bgm, data_proccesing.train_scaled)
