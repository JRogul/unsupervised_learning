from src import train
from src import data_proccesing
import joblib
from src import config
from src import data_visualization
train_scaled = data_proccesing.train_scaled

model_bgm = joblib.load('../models/BayesianGaussianMixture')

X, y = train.predict(model_bgm,train_scaled)
data_visualization.umapp(train_scaled,y)
config.usnupervised_metrics(y, train_scaled, '../models/BayesianGaussianMixture')
