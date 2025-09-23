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
            # --- parameter-related attributed --- #
            self.loc = params[0:k]
            self.skew = params[k:2*k]
            self.df = params[2*k+1]
            self.disp = params[2*k+2:]

        @property
        def params(self):
            return {'loc':self.loc,
                    'skew':self.skew,
                    'df':self.df,
                    'disp':self.disp}

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