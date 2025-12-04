

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


def linear_reg(X, Y, X1):

  clf = LinearRegression(fit_intercept=False).fit(X, Y)

  return clf.predict(X1)

def lasso_reg(X, Y, X1, lam=0.0334*1.374):
    model = Lasso(alpha=lam, fit_intercept=True)
    model.fit(X, Y)
    return model.predict(X1)
