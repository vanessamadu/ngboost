"""The NGBoost multivariate skew-t distribution and scores"""

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore

def MultivariateSkewt(k):
    
    class K_VariateSkewt(RegressionDistn):

        n_params = None # NEEDS DEFINING AS A FUNCTION OF K
        
        def __init__(self, params):
            self.score = [MultivariateSkewtLogScore]
            self.n_params = None # NEEDS DEFINING
            self._params = params
            self.dist = None # NEEDS DEFINING

        @property
        def param(self):
            return {} # NEEDS DEFINING
        
        def fit(self):
            pass

        def sample(self):
            pass

        def mean(self):
            pass
    return K_VariateSkewt

class MultivariateSkewtLogScore(LogScore):
    def score(self,Y):
       return -self.dist.logpdf(Y)
    
    def d_score(self,Y):
        pass
    
    def metric(self):
        pass