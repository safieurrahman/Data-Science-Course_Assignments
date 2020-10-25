from sklearn.metrics import r2_score
from scipy.optimize import minimize
import numpy as np


class LinReg():
    """
    This class trains linear model with intercept (bias)
    """
    def __init__(self, C = None):
        self.p = None # parameters field of model
        self.C = C

    def predict(self, X):
        if self.p is None:
            raise("Please train the model first!")

        return self._predict(X, self.p)

    def _predict(self, X, p):
        # this is used for optimization of objective
        # returns a vector where i-th elemnt corresponds to
        # the output of the model with i-th input row

        # <<< INSERT CODE HERE >>>
        

    def fit(self, X,Y):
        # objective: minimize squared deviation
        def obj(p):
            Yp = self._predict(X, p)

            # <<< INSERT CODE HERE >>>

        p0 = np.zeros(X.shape[1] + 1)
        sol = minimize(obj, p0, method="L-BFGS-B", tol=1e-6)
        self.p = sol.x

    def score(self, Xv, Yv):
        # this function gets predictions on Xv and compares to Yv
        # using r2_score function
        Yp = self.predict(Xv)
        return r2_score(Yv, Yp)
