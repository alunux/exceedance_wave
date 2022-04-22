from pprint import pprint
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
from common.cv import MonteCarloCV
from exceedance.rf import ExceedanceRandomForest
from exceedance.cdf import CDFEngine

from common.utils import save_data

pd.set_option('display.max_columns', None)

if __name__ == '__main__':

    # wave = pd.read_csv(DATA_DIR)
    path = Path(DATA_DIR)

    wave = pd.read_csv(path.absolute())

    # Preprocessing data
    # df = df.drop(UNUSED_COLUMNS, axis=1)
    # df['DATE'] = pd.to_datetime(df['DATE'])
    # df = df.set_index('DATE')
    # df = df.resample('H').mean()
    #
    wave = wave[1:]
    # filtering target buoy
    buoy_wave = wave.loc[wave['station_id'] == BUOY_ID, :]
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

    HORIZON_LIST = list(range(1, MAX_HORIZON))

    for horizon_ in HORIZON_LIST:
        #
        cv = MonteCarloCV(n_splits=CV_N_FOLDS, train_size=TRAIN_SIZE, test_size=TEST_SIZE)
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
            #
            print('Removing invalid data')
            X_train, y_train = ExceedanceRandomForest.remove_invalid_observations(X=X_train, y=y_train,
                                                                                  lag_columns=LAG_COLUMNS,
                                                                                  decision_thr=thr)
            X_test, y_test = ExceedanceRandomForest.remove_invalid_observations(X=X_test, y=y_test,
                                                                                lag_columns=LAG_COLUMNS,
                                                                                decision_thr=thr)
            #
            y_train_clf = (y_train >= thr).astype(int)
            y_test_clf = (y_test >= thr).astype(int)
            #
            model = ExceedanceRandomForest()
            #
            model.fit(X_train, y_train)
            # preds_all = {}
            #
            y_hat = model.predict(X_test)
            print('Exceedance probability')
            print('MC rf')
            # y_prob_mc = MonteCarloEngine.rng_ensemble(
            #     y_hat=y_hat,
            #     scale=y_std,
            #     decision_thr=thr,
            #     n_trials=MC_N_TRIALS
            # )
            print('Exceedance probability - Timalytical')
            print('MC rf')
            y_prob_ae = CDFEngine.get_all_probs(
                y_hat=y_hat,
                scale=y_std,
                threshold=thr,
            )
            # y_prob_mc = pd.DataFrame({k+'_mc':v for k,v in y_prob_mc.items()})
            # y_prob_ae = pd.DataFrame({k+'_ae':v for k,v in y_prob_ae.items()})
            # y_prob = pd.concat([y_prob_mc,y_prob_ae], axis=1)
            auc_scores = {
                k: roc_auc_score(y_true=y_test_clf, y_score=y_prob_ae[k]) for k in y_prob_ae
            }
            #
            results.append(auc_scores)
            #
            pprint(results)
        #
        save_data(results, f'results/scores_sensitivity_h{horizon_}.pkl')
