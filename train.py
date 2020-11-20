from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import mlflow
import numpy as np

num_samples =  100

def get_ys(xs):
    signal = -0.1*xs**3 + xs**2 - 5*xs - 5
    noise = np.random.normal(0,200,(len(xs),1))
    return signal + noise

X = np.random.uniform(-20,20,num_samples).reshape((num_samples,1))
y = get_ys(X)


model = Pipeline([
    ("Poly", PolynomialFeatures(degree=3)),
    ("LenReg", LinearRegression())
])
model.fit(X,y)

mlflow.sklearn.save_model(model, "model")
