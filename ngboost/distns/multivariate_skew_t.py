"""
"""
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore

from scipy.special import gamma
from scipy.stats import t, multivariate_normal, chi2
import numpy as np

class MVStLogScore(LogScore):
    def score(self, Y):
        return -self.logpdf(Y)

    def d_score(self, Y):
        """
       

        Args:
            Y: The response data

        Returns:
            self.N, self.n_params shaped array containing the gradient.

        """
    
        pass

    def metric(self):

        """

        Returns:
            self.N, self.n_params, self.n_params shaped array containing the fisher information for
             the ith observation in the last two indices.

        """
        pass


def MultivariateSkewT(d):
    """
    #  Factory function that generates classes for
    #  d-dimensional multivariate skew-t distributions for NGBoost

    # This distribution has LogScore implemented for it.

    # Currently only for a regression implementation.
    """
    class MVSt(RegressionDistn):
        """

        """

        n_params = int(1 + d * (d + 5) / 2)
        scores = [MVStLogScore]
        multi_output = True

        def __init__(self, params):
            super().__init__(params)

            self.d = d

            self.nu0 = None
            self.nu_tilde = None
            self.rho = None
            self.loc = None
            self.eta = None

        def logpdf(self, Y):

            term1 = np.log( 
                gamma( 
                    (self.df + d) / 2
                ) / (
                gamma(
                    self.df / 2
                ) * (np.pi * self.df) ** (d / 2)
                )
            )

            term2 = np.sum(self.rho) / 4

            term3 = - (self.df / 2) * (1 + d / self.df) * np.log(1 + self.Q(Y) / self.df)

            term4 = np.log(2 * t.cdf(
                np.sqrt(self.df + d / 
                        self.df + self.Q(Y)) * np.dot(
                            self.eta, Y - self.loc
                        )
            ), df = self.df + d)

            return term1 + term2 + term3 + term4

        def fit(Y):
            pass

        def rv(self):
            u_star = multivariate_normal(0, self.omega_star).rvs(size=self.d + 1)
            v = chi2(df = self.df).rvs() / self.df
            z = np.matmul(self.stds, u_star[1:]) * np.sign(u_star[0])
            return self.loc + z / np.sqrt(v)


        def rvs(self, n):
            return [self.rv() for _ in range(n)]

        def sample(self, n):
            return self.rvs(n)

        @property
        def disp(self):
            pass

        @property
        def stds(self):
            pass

        @property
        def corr(self):
            pass

        @property
        def omega_star(self):
            pass

        @property
        def df(self):
            return self.nu0 + np.exp(self.nu_tilde)

        @property
        def skew(self):
            pass

        def Q(self, Y):
            return np.matmul(
                np.matmul(Y - self.loc , self.disp),
                np.transpose(Y - self.loc)
            )

        @property
        def delta(self):
            np.matmul(
                self.corr, self.skew
                ) / (
                    1 + np.sqrt(
                        np.matmul(
                            np.matmul(
                                np.transpose(self.skew),
                                self.corr),
                        self.skew)
                    )
                )

        @property
        def params(self):
            return {
                "location": self.loc, 
                "dispersion": self.disp,
                "skew": self.skew,
                "df": self.df
                }

        def scipy_distribution(self):
            """

            """
            pass

        def mean(self):
            pass

        def cov(self):
            pass

    return MVSt
