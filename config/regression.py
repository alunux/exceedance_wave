from sklearn.ensemble \
    import (RandomForestRegressor,
            ExtraTreesRegressor,
            AdaBoostRegressor,
            BaggingRegressor)
from sklearn.linear_model \
    import (Lasso,
            Ridge,
            OrthogonalMatchingPursuit,
            ElasticNet)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression, PLSCanonical

from regression.lgbm import LightGBMRegressor
from regression.m5 import CubistR
from regression.mars import EarthRegressor

from common.utils import expand_grid_from_dict

METHODS = \
    dict(
        EarthRegressor=EarthRegressor,
        CubistR=CubistR,
        LightGBMRegressor=LightGBMRegressor,
        RandomForestRegressor=RandomForestRegressor,
        PLSRegression=PLSRegression,
        PLSCanonical=PLSCanonical,
        ExtraTreesRegressor=ExtraTreesRegressor,
        OrthogonalMatchingPursuit=OrthogonalMatchingPursuit,
        AdaBoostRegressor=AdaBoostRegressor,
        Lasso=Lasso,
        KNeighborsRegressor=KNeighborsRegressor,
        Ridge=Ridge,
        ElasticNet=ElasticNet,
        BaggingRegressor=BaggingRegressor,
    )

METHODS_PARAMETERS = \
    dict(
        RandomForestRegressor={
            'n_estimators': [50, 100],
            'max_depth': [None, 3],
        },
        ExtraTreesRegressor={
            'n_estimators': [50, 100],
            'max_depth': [None, 3],
        },
        OrthogonalMatchingPursuit={},
        AdaBoostRegressor={
            'base_estimator': [DecisionTreeRegressor(max_depth=3),
                               DecisionTreeRegressor(max_depth=1)],
            'loss': ['linear'],
            'learning_rate': [0.3, 1],
        },
        Lasso={
            'alpha': [1, .5, .25, .75]
        },
        KNeighborsRegressor={
            'n_neighbors': [1, 5, 10],
            'weights': ['uniform', 'distance'],
        },
        CubistR={
            'n_committees': [1, 2, 3]
        },
        Ridge={
            'alpha': [1, .5, .25, .75]
        },
        ElasticNet={
        },
        PLSRegression={
            'n_components': [2, 5]
        },
        PLSCanonical={
            'n_components': [2, 5]
        },
        BaggingRegressor={
            'base_estimator': [DecisionTreeRegressor(max_depth=3),
                               #DecisionTreeRegressor(max_depth=5),
                               DecisionTreeRegressor(max_depth=1)
                               ],
            'n_estimators': [25, 50]
        },
        LightGBMRegressor={
            'boosting_type': ['dart', 'goss', 'gbdt']
        },
        EarthRegressor={
            'degree': [1, 3],
            'nk': [10, 20],
            'pmethod': ['forward'],
            'thresh': [0.01, 0.001]
        }
    )

MODELS_ON_SUBSET = ['GaussianProcessRegressor', 'SVR', 'LinearSVR', 'NuSVR', 'CubistR', 'ProjectionPursuitRegressor']
#
# n_models = 0
# for k in METHODS_PARAMETERS:
#     if len(METHODS_PARAMETERS[k]) > 0:
#         n_models += expand_grid_from_dict(METHODS_PARAMETERS[k]).shape[0]
#     else:
#         n_models += 1
#
# print(n_models)
