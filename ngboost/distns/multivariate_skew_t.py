"""
"""
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore

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


def MultivariateSkewT(k):
    """
    #  Factory function that generates classes for
    #  k-dimensional multivariate skew-t distributions for NGBoost

    # This distribution has LogScore implemented for it.

    # Currently only for a regression implementation.
    """
    class MVSt(RegressionDistn):
        """

        """

        n_params = None
        scores = [MVStLogScore]
        multi_output = True

        def __init__(self, params):
            super().__init__(params)
            pass

        def logpdf(self, Y):
            pass

        def fit(Y):
            pass

        def rv(self):
            pass

        def rvs(self, n):
            return [self.rv() for _ in range(n)]

        def sample(self, n):
            return self.rvs(n)

        @property
        def disp(self):
            pass

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
