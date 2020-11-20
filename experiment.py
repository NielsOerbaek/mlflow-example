from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt 
import numpy as np

num_samples =  100

def get_ys(xs):
    signal = -0.1*xs**3 + xs**2 - 5*xs - 5
    noise = np.random.normal(0,200,(len(xs),1))
    return signal + noise

X = np.random.uniform(-20,20,num_samples).reshape((num_samples,1))
y = get_ys(X)

plt.scatter(X,y,label="data")
plt.show()
exit()

for degree in range(1,6):
    model = Pipeline([
        ("Poly", PolynomialFeatures(degree=degree)),
        ("LenReg", LinearRegression())
    ])
    model.fit(X,y)
    plotting_x = np.linspace(-20,20,num=50).reshape((50,1))
    r2 = r2_score(y,  model.predict(X))
    print("degree: %d, r2: %.2f" % (degree, r2))
    plotting_preds = model.predict(plotting_x)
    plt.plot(plotting_x, plotting_preds, label=f"degree={degree}")

plt.legend()
plt.show()

