import numpy as np
import pandas as pd
import copy
from typing import Union
from sklearn.base import BaseEstimator, RegressorMixin
import rpy2.robjects as r_objects
from rpy2.robjects import pandas2ri


class EarthRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, nk: int, degree, pmethod, thresh):
        self.nk = nk
        self.degree = degree
        self.pmethod = pmethod
        self.thresh = thresh
        self.model = None

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: np.ndarray):
        # assert isinstance(X, pd.DataFrame)
        if isinstance(X, pd.DataFrame):
            X = X.values

        pandas2ri.activate()

        X_fit = copy.deepcopy(X)

        X_fit = pd.DataFrame(X_fit)
        X_fit.columns = ['c' + str(i) for i in range(X_fit.shape[1])]
        X_fit["target"] = y

        data_set = pandas2ri.py2rpy_pandasdataframe(X_fit)

        r_objects.r('''
                   train_mars <- function(train, nk, degree,thresh,pmethod) {
                            library(earth)
                            model <- earth(target ~.,
                                          train,
                                          nk = nk,
                                          degree = degree,
                                          thresh = thresh,
                                          pmethod=pmethod)

                            model

                    }
                    ''')

        fit_model = r_objects.globalenv['train_mars']
        self.model = fit_model(data_set, self.nk, self.degree, self.thresh, self.pmethod)
        pandas2ri.deactivate()

    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            X = X.values

        pandas2ri.activate()
        r_objects.r('''
                   predict_earth <- function(model,test) {
                            library(earth)

                            pred <- as.numeric(predict(model, test))

                    }
                    ''')

        predict_method = r_objects.globalenv['predict_earth']

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            X.columns = ['c' + str(i) for i in range(X.shape[1])]

        y_hat = predict_method(self.model, X)
        pandas2ri.deactivate()

        return y_hat
