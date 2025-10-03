"""The NGBoost multivariate skew-t distribution and scores"""

from ngboost.distns.distn import RegressionDistn
from ngboost.distns.utils import cholesky_factor
from ngboost.scores import LogScore
import numpy as np
from scipy import special


def MultivariateSkewt(k):
    
    class K_VariateSkewt(RegressionDistn):

        n_params = int((k + 4) * (k + 1) / 2 - 1)
        score = [MultivariateSkewtLogScore]
        multi_output = True
        
        def __init__(self, params):
            super().__init__(params)
            self.dim = int(k)
            self.n_data = int(params.shape[1])

            # ------ parameter attributes ------ #
            self.loc = params[0:k,:]
            self.skew = params[k:2*k,:]
            self.df = params[2*k+1,:]
            self.modified_A = params[2*k+2:,:]

            # === related attributes === #
            self.A = cholesky_factor(self.modified_A)

            # ---------------------------------- #

        @property
        def params(self):
            return {'loc':self.loc,
                    'skew':self.skew,
                    'df':self.df,
                    'Exponentiated diagonal lower triangle of A':self.modified_A}

        @property
        def disp_inv(self):
            return self.A @ self.A.transpose(0, 2, 1)
        
        @property
        def disp(self):
            A_inv = np.linalg.inv(self.A)
            return A_inv.transpose(0,2,1) @ A_inv 
        
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
                dispersion = self.disp
                num = np.matmul(dispersion,self.skew)
                denom = (self.df-2)*np.sqrt(1+np.dot(self.skew,np.matmul(dispersion,self.skew)))
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