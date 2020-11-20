from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt 
import numpy as np

num_samples =  100

import mlflow
#mlflow.set_tracking_uri("http://training.itu.dk:5000/")
#mlflow.set_experiment("Demo Hermes")

def get_ys(xs):
    signal = -0.1*xs**3 + xs**2 - 5*xs - 5
    noise = np.random.normal(0,200,(len(xs),1))
    return signal + noise

X = np.random.uniform(-20,20,num_samples).reshape((num_samples,1))
y = get_ys(X)

for degree in range(1,6):
    # Starting run
    with mlflow.start_run():
        # Logging params
        mlflow.log_params({
            "num_samples": num_samples,
            "poly_degree": degree
        })

        model = Pipeline([
            ("Poly", PolynomialFeatures(degree=degree)),
            ("LenReg", LinearRegression())
        ])

        model.fit(X,y)

        # Logging metrics
        r2 = r2_score(y,  model.predict(X))
        mlflow.log_metric("r2", r2)

        # Logging artifacts
        plt.clf()
        plotting_x = np.linspace(-20,20,num=50).reshape((50,1))
        plotting_preds = model.predict(plotting_x)
        plt.scatter(X,y,label="data")
        plt.plot(plotting_x, plotting_preds, label=f"degree={degree}")
        plt.legend()
        filename = "plots/PolyReg_%d_%d.png" % (degree, num_samples)
        plt.savefig(filename)
        mlflow.log_artifact(filename)

        # Finally, log the model
        mlflow.sklearn.log_model(model, "model")


