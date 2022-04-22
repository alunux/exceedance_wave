import numpy as np
import pandas as pd

from scipy.stats import norm, laplace, logistic, gumbel_r, lognorm, cauchy

DISTRIBUTIONS = {
    'norm': norm,
    'laplace': laplace,
    'logistic': logistic,
    'gumbel': gumbel_r,
    'lognorm': lognorm,
    'cauchy': cauchy,
}


class CDFEngine:

    @staticmethod
    def cdf_by_dist(distribution: str, location: float, scale: float):
        DIST = {
            'norm': dict(loc=location, scale=scale),
            'laplace': dict(loc=location, scale=scale),
            'logistic': dict(loc=location, scale=scale),
            'gumbel': dict(loc=location, scale=scale),
            'lognorm': dict(loc=location, s=scale),
            'cauchy': dict(loc=location, scale=scale),
        }

        return DIST[distribution]

    @classmethod
    def point_exceedance_by_dist(cls, distribution: str, location: float, scale: float, threshold: float):
        PROBS = {
            'norm': 1 - norm.cdf(threshold, **cls.cdf_by_dist('norm', location, scale)),
            'laplace': 1 - laplace.cdf(threshold, **cls.cdf_by_dist('laplace', location, scale)),
            'logistic': 1 - logistic.cdf(threshold, **cls.cdf_by_dist('logistic', location, scale)),
            'gumbel': 1 - gumbel_r.cdf(threshold, **cls.cdf_by_dist('gumbel', location, scale)),
            'lognorm': 1 - lognorm.cdf(threshold, **cls.cdf_by_dist('lognorm', location, scale)),
            'cauchy': 1 - cauchy.cdf(threshold, **cls.cdf_by_dist('cauchy', location, scale)),
        }

        return PROBS[distribution]

    @classmethod
    def get_probs(cls, distribution: str, y_hat: np.ndarray, scale: float, threshold: float):
        p_exc = [cls.point_exceedance_by_dist(distribution=distribution,
                                              location=x_,
                                              scale=scale,
                                              threshold=threshold)
                 for x_ in y_hat]

        p_exc = np.asarray(p_exc)

        return p_exc

    @classmethod
    def get_all_probs(cls, y_hat: np.ndarray, scale: float, threshold: float):
        dists = [*DISTRIBUTIONS]

        dist_pe = {}
        for d_ in dists:
            p_exc = [cls.point_exceedance_by_dist(distribution=d_,
                                                  location=x_,
                                                  scale=scale,
                                                  threshold=threshold)
                     for x_ in y_hat]

            p_exc = np.asarray(p_exc)
            dist_pe[d_] = p_exc

        dist_pe = pd.DataFrame(dist_pe)

        return dist_pe
