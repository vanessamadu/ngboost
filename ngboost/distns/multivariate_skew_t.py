"""The NGBoost multivariate skew-t distribution and scores"""

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore
import numpy as np
from scipy import stats, special


def MultivariateSkewt(k):
    
    class K_VariateSkewt(RegressionDistn):

        n_params = None # NEEDS DEFINING AS A FUNCTION OF K
        score = [MultivariateSkewtLogScore]
        multi_output = True
        
        def __init__(self, params):
            super().__init__(params)
            self.dim = k
            # ------ parameter-related attributed ------ #
            self.loc = params[0:k]
            self.skew = params[k:2*k]
            self.df = params[2*k+1]
            self.disp = params[2*k+2:]
            self.A = None # NEEDS DEFINING
            # ------------------------------------------ #

        @property
        def params(self):
            return {'loc':self.loc,
                    'skew':self.skew,
                    'df':self.df,
                    'disp':self.disp}

        @property
        def disp_inv(self):
            return np.matmul(np.transpose(self.A),self.A) # uses Sigma^{-1} = A^T A
        
        # ====== DISTRIBUTION IMPLEMENTATION ====== #
        def logpdf(self,Y):

            # logpdf terms
            c_d = np.log(special.gamma(
                (self.df+self.dim)/2)
                ) 
            - np.log(special.gamma(self.df/2)) 
            - (self.dim/2)*np.log(np.pi * self.df)

            Q = np.dot(Y-self.loc,
                       np.matmul(self.disp_inv),Y-self.loc)
            
            ## intermediate terms
            det_disp = 1/(np.prod(np.diag(self.A)))**2
            T_input = (np.dot(self.skew,Y-self.loc)*np.sqrt(self.df + self.dim)
                              )/(np.sqrt(Q + self.df)
            )
            
            T = 0.5 + T_input*special.gamma((self.df+self.dim+1)/2)*special.hyp2f1(
                0.5,(self.df+self.dim+1)/2,1.5,-T_input**2/(self.df+self.dim)
            )/(np.sqrt(np.pi*(self.df+self.dim))*special.gamma((self.df+self.dim)/2))

            return c_d - 0.5*special.gamma(det_disp) - (self.df/2)*(1+self.dim/self.df)*np.log(1+Q/self.df) + np.log(2*T)

        def rv(self):
            pass

        # ========================================= #
        def fit(Y):
            pass

        def sample(self,m):
            return [self.rv() for _ in range(m)]

        def mean(self):
            
            if self.df > 2:
                C = np.sqrt(2*self.df/np.pi)
                num = np.matmul(self.disp,self.skew)
                denom = (self.df-2)*np.sqrt(1+np.dot(self.skew,np.matmul(self.disp,self.skew)))
                return C*num/denom + self.loc
            else:
                raise ValueError("Mean is undefined for df <= 2")
                

    return K_VariateSkewt

class MultivariateSkewtLogScore(LogScore):
    def score(self,Y):
       return -self.logpdf(Y)
    
    def d_score(self,Y):
        pass
    
    def metric(self):
        pass