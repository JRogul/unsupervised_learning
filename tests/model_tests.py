from src import train
from src import data_proccesing
import joblib
train_scaled = data_proccesing.train_scaled
model_bgm = joblib.load('../models/BayesianGaussianMixture')

X, y = train.predict(model_bgm,train_scaled)
train.submission(y)