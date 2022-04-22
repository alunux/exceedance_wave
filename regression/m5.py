import numpy as np
import pandas as pd
import copy
from typing import Union
from sklearn.base import BaseEstimator, RegressorMixin
import rpy2.robjects as r_objects
from rpy2.robjects import pandas2ri


class CubistR(BaseEstimator, RegressorMixin):

    def __init__(self, n_committees: int = 1):
        self.n_committees = n_committees
        self.model = None

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: np.ndarray):
        # assert isinstance(X, pd.DataFrame)
        if isinstance(X, pd.DataFrame):
            X = X.values

        pandas2ri.activate()

        X_fit = copy.deepcopy(X)

        if isinstance(X, np.ndarray):
            X_fit = np.c_[X_fit, y]
        else:
            X_fit["target"] = y

        X_fit = pd.DataFrame(X_fit)
        X_fit.columns = ['c' + str(i) for i in range(X_fit.shape[1])]

        data_set = pandas2ri.py2rpy_pandasdataframe(X_fit)

        r_objects.r('''
                   train_cubist <- function(train, committees) {
                            x = as.matrix(train[,-ncol(train)])
                            y = as.numeric(train[,ncol(train)])

                            library(Cubist)
                            model <- cubist(x=x, y=y, committees = committees)
                            model

                    }
                    ''')

        fit_model = r_objects.globalenv['train_cubist']
        self.model = fit_model(data_set, self.n_committees)
        pandas2ri.deactivate()

    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            X = X.values

        pandas2ri.activate()
        r_objects.r('''
                   predict_cubist <- function(model,test) {
                            library(Cubist)

                            pred <- as.numeric(predict(model, test))

                    }
                    ''')

        predict_method = r_objects.globalenv['predict_cubist']

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            X.columns = ['c' + str(i) for i in range(X.shape[1])]

        y_hat = predict_method(self.model, X)
        pandas2ri.deactivate()

        return y_hat
