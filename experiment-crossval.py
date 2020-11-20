from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt 
import numpy as np

num_samples =  1000

import mlflow
#mlflow.set_tracking_uri("http://training.itu.dk:5000/")
#mlflow.set_experiment("Hermes Demo")
mlflow.sklearn.autolog()

def get_ys(xs):
    signal = -0.1*xs**3 + xs**2 - 5*xs - 5
    noise = np.random.normal(0,200,(len(xs),1))
    return signal + noise

X = np.random.uniform(-20,20,num_samples).reshape((num_samples,1))
y = get_ys(X)

poly_params = {
    'Poly__degree': range(1,8),
}
mlflow.end_run()
with mlflow.start_run():

    mlflow.log_param("num_samples", num_samples)

    model = Pipeline([
        ("Poly", PolynomialFeatures()),
        ("LinReg", LinearRegression())
    ])

    gridsearch = GridSearchCV(model, poly_params, scoring="r2")
    gridsearch.fit(X,y)


