"""The NGBoost multivariate skew-t distribution and scores"""

from ngboost.distns.distn import RegressionDistn
from ngboost.distns.utils import cholesky_factor
from ngboost.scores import LogScore
import numpy as np
from scipy import special


def MultivariateSkewt(p):
    
    class P_VariateSkewt(RegressionDistn):

        n_params = int((p + 4) * (p + 1) / 2 - 1)
        score = [MultivariateSkewtLogScore]
        multi_output = True
        
        def __init__(self, params):
            super().__init__(params) # n_params x n_data
            self.dim = int(p)
            self.n_data = int(params.shape[1]) 

            # ------ parameter attributes ------ #
            self.loc = params[0:p,:] # dim x n_data
            self.skew = params[p:2*p,:] # dim x n_data
            self.df = params[2*p+1,:] # 1 x n_data
            self.modified_A = params[2*p+2:,:] # p(p+1)/2 x n_data

            # === related attributes === #
            self.A = cholesky_factor(self.modified_A,self.dim) # p x p x n_data

            # ---------------------------------- #

        @property
        def params(self):
            return {'loc':self.loc,
                    'skew':self.skew,
                    'df':self.df,
                    'Log diagonal lower triangle of A':self.modified_A}

        @property
        def disp_inv(self):
            return self.A @ self.A.transpose(0, 2, 1) # p x p x n_data
        
        @property
        def disp(self):
            A_inv = np.linalg.inv(self.A)
            return A_inv.transpose(0,2,1) @ A_inv # p x p x n_data
        
        # ====== DISTRIBUTION IMPLEMENTATION ====== #
        def logpdf(self,Y):
            # should return something 1 x n_data

            # logpdf terms
            c_d = np.log(special.gamma(
                (self.df+self.dim)/2)
                ) 
            - np.log(special.gamma(self.df/2)) 
            - (self.dim/2)*np.log(np.pi * self.df) 

            Q = np.einsum('j...,jk...,k...',Y-self.loc,self.disp_inv,Y-self.loc) # 1 x n_data
            #np.dot(Y-self.loc,np.matmul(self.disp_inv),Y-self.loc)
            
            ## intermediate terms
            det_disp = 1/(np.prod(np.diag(self.A)))**2

            T_input = np.einsum('i...,i...',self.skew,Y-self.loc)*np.sqrt(self.df + self.dim)/(np.sqrt(Q + self.df))
            
            T = 0.5 + T_input*special.gamma((self.df+self.dim+1)/2)*special.hyp2f1(
                0.5,(self.df+self.dim+1)/2,1.5,(-T_input**2)/(self.df+self.dim)
            )/(np.sqrt(np.pi*(self.df+self.dim))*special.gamma((self.df+self.dim)/2))

            return c_d - 0.5*np.log(det_disp) - (self.df/2)*(1+self.dim/self.df)*np.log(1+Q/self.df) + np.log(2*T)

        def rv(self):
            pass

        # ========================================= #
        def fit(Y):
            pass

        def sample(self,m):
            return [self.rv() for _ in range(m)]

        def mean(self):
            # return 1 x n object

            if self.df > 2:
                C = np.sqrt(2*self.df/np.pi)
                dispersion = self.disp
                num = np.einsum('jk...,j...',dispersion,self.skew)
                denom = (self.df-2)*np.sqrt(1+np.einsum('j...,jk...,k...',self.skew,dispersion,self.skew))
                return C*num/denom + self.loc
            else:
                raise ValueError("Mean is undefined for df <= 2")
                

    return P_VariateSkewt

class MultivariateSkewtLogScore(LogScore):
    def score(self,Y):
       return -self.logpdf(Y)
    
    def d_score(self,Y):
        pass
    
    def metric(self):
        pass