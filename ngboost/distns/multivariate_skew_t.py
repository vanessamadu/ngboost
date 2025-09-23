"""The NGBoost multivariate skew-t distribution and scores"""

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


def MultivariateSkewt(k):
    
    class K_VariateSkewt(RegressionDistn):

        n_params = None # NEEDS DEFINING AS A FUNCTION OF K
        score = [MultivariateSkewtLogScore]
        multi_output = True
        
        def __init__(self, params):
            super().__init__(params)
            self._params = params
            self.dist = None # NEEDS DEFINING

            # --- parameter-related attributed --- #

        @property
        def param(self):
            return {} # NEEDS DEFINING

        # ====== DISTRIBUTION IMPLEMENTATION ====== #
        def logpdf(self,Y):
            pass

        def rv(self):
            pass

        # ========================================= #
        def fit(Y):
            pass

        def sample(self,m):
            pass

        def mean(self):
            pass

    return K_VariateSkewt

class MultivariateSkewtLogScore(LogScore):
    def score(self,Y):
       return -self.logpdf(Y)
    
    def d_score(self,Y):
        pass
    
    def metric(self):
        pass