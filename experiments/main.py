from pprint import pprint
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from common.cv import MonteCarloCV
from common.utils import save_data

from config.wave import (DATA_DIR,
                         EMBED_DIM,
                         MAX_HORIZON,
                         THRESHOLD_PERCENTILE,
                         CV_N_FOLDS,
                         TRAIN_SIZE,
                         TEST_SIZE,
                         TARGET,
                         UNUSED_COLUMNS,
                         BUOY_ID)

from common.embed import MultivariateTDE
from exceedance.cdf import CDFEngine
from exceedance.rf import ExceedanceRandomForest
from exceedance.classification import VanillaClassifier, ResampledClassifier
from exceedance.ensemble import HeterogeneousEnsemble

pd.set_option('display.max_columns', None)


if __name__ == '__main__':

    path = Path(DATA_DIR)

    wave = pd.read_csv(path.absolute())
    # wave = pd.read_csv(DATA_DIR)

    # Preprocessing data
    # df = df.drop(UNUSED_COLUMNS, axis=1)
    # df['DATE'] = pd.to_datetime(df['DATE'])
    # df = df.set_index('DATE')
    # df = df.resample('H').mean()
    #
    wave = wave[1:]
    # filtering target buoy
    buoy_wave = wave.loc[wave['station_id'] == BUOY_ID, :]
    # buoy_wave.head(40000).to_csv('data/wave_data.csv', index=False)
    # casting time to appropriate format
    buoy_wave.loc[:, 'time'] = pd.to_datetime(buoy_wave['time'])
    # setting time as index
    buoy_wave.set_index('time', inplace=True)
    buoy_wave = buoy_wave.sort_index()
    # removing useless/unused columns
    buoy_wave = buoy_wave.loc[:, ~buoy_wave.columns.str.endswith('_qc')]
    buoy_wave = buoy_wave.drop(UNUSED_COLUMNS, axis=1)
    # casting columns as floats
    buoy_wave = buoy_wave.astype(float)
    buoy_wave = buoy_wave.resample('H').mean()
    buoy_wave[TARGET] = buoy_wave[TARGET] / 100

    HORIZON_LIST = list(range(1, MAX_HORIZON + 1))

    for horizon_ in HORIZON_LIST:
        print(f'Horizon: {horizon_}')
        #
        cv = MonteCarloCV(n_splits=CV_N_FOLDS,
                          train_size=TRAIN_SIZE,
                          test_size=TEST_SIZE)
        #
        data_set = MultivariateTDE(data=buoy_wave,
                                   horizon=horizon_,
                                   k=EMBED_DIM,
                                   target_col=TARGET)
        #
        LAG_COLUMNS = [f'{TARGET}-{i}' for i in range(1, EMBED_DIM + 1)]
        TARGET_COLUMNS = [f'{TARGET}+{i}' for i in range(1, horizon_ + 1)]
        #
        data_set = data_set.dropna()
        #
        X = data_set.drop(TARGET_COLUMNS, axis=1)
        Y = data_set[TARGET_COLUMNS]
        y = Y[TARGET_COLUMNS[-1]]
        #
        results = []
        for train_index, test_index in cv.split(X):
            print('.')
            print('Subsetting iter data')
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            #
            print('Getting pars')
            y_std = y_train.std()
            thr = np.quantile(y_train, THRESHOLD_PERCENTILE)
            #
            print('Removing invalid data')
            X_train, y_train = \
                ExceedanceRandomForest.remove_invalid_observations(X=X_train, y=y_train,
                                                                   lag_columns=LAG_COLUMNS,
                                                                   decision_thr=thr)
            X_test, y_test = \
                ExceedanceRandomForest.remove_invalid_observations(X=X_test, y=y_test,
                                                                   lag_columns=LAG_COLUMNS,
                                                                   decision_thr=thr)
            #
            y_train_clf = (y_train >= thr).astype(int)
            y_test_clf = (y_test >= thr).astype(int)
            #
            print('Training')
            all_methods = {
                'RFC': VanillaClassifier(model=RandomForestClassifier()),
                'RFC+SMOTE': ResampledClassifier(model=RandomForestClassifier(), resampling_model=SMOTE()),
                'LR': VanillaClassifier(model=LogisticRegression()),
                'RFR': ExceedanceRandomForest(),
                'LASSO': Lasso(),
                'HRE': HeterogeneousEnsemble(),
            }
            #
            print('Fitting models')
            all_methods['RFC'].fit(X_train, y_train_clf)
            all_methods['RFC+SMOTE'].fit(X_train, y_train_clf)
            all_methods['LR'].fit(X_train, y_train_clf)
            all_methods['RFR'].fit(X_train, y_train)
            all_methods['LASSO'].fit(X_train, y_train)
            all_methods['HRE'].fit_and_trim(X_train, y_train, select_percentile=.5)
            #
            print('Predicting...')
            y_hat_num_rf = all_methods['RFR'].predict(X_test)
            y_hat_num_he = all_methods['HRE'].predict(X_test)
            y_hat_num_ls = all_methods['LASSO'].predict(X_test)
            #
            print('Exceedance probability')
            print('MC rf')
            y_prob_rf_ae = CDFEngine.get_probs(
                y_hat=y_hat_num_rf,
                distribution='norm',
                scale=y_std,
                threshold=thr,
            )
            print('MC he')
            y_prob_he_ae = CDFEngine.get_probs(
                y_hat=y_hat_num_he,
                distribution='norm',
                scale=y_std,
                threshold=thr,
            )
            print('MC ls')
            y_prob_ls_ae = CDFEngine.get_probs(
                y_hat=y_hat_num_ls,
                distribution='norm',
                scale=y_std,
                threshold=thr,
            )
            print('Exceed prob')
            exceedance_prob = {
                'RFC': all_methods['RFC'].predict_proba(X_test),
                'RFC+SMOTE': all_methods['RFC+SMOTE'].predict_proba(X_test),
                'LR': all_methods['LR'].predict_proba(X_test),
                'RFR': all_methods['RFR'].predict_exceedance_proba(X_test, thr),
                'HRE': all_methods['HRE'].predict_proba(X_test, thr),
                'RFR+CDF': y_prob_rf_ae,
                'LASSO+CDF': y_prob_ls_ae,
                'HRE+CDF': y_prob_he_ae,
            }
            #
            forecasts = {
                'RFR': y_hat_num_rf,
                'HRE': y_hat_num_he,
                'LASSO': y_hat_num_ls,
            }
            #
            auc_scores = {}
            for method_ in exceedance_prob:
                auc_scores[method_] = \
                    roc_auc_score(y_true=y_test_clf, y_score=exceedance_prob[method_])
            #
            auc_scores = {k + '_auc': auc_scores[k] for k in auc_scores}
            #
            r2_scores = {}
            for method_ in forecasts:
                r2_scores[method_] = \
                    r2_score(y_true=y_test, y_pred=forecasts[method_])
            #
            r2_scores = {k + '_r2': r2_scores[k] for k in r2_scores}
            #
            results_iter = {**auc_scores, **r2_scores}
            pprint(results_iter)
            #
            results.append(results_iter)
            #
            pprint(results)
        #
        save_data(results, f'results/scores_h{horizon_}.pkl')
