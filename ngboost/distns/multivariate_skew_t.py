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
        
        def Q(self,Y):
            return np.einsum('j...,jk...,k...',Y-self.loc,self.disp_inv,Y-self.loc) # 1 x n_data
            
        
        def T(self,Y):
            T_input = np.einsum('i...,i...',self.skew,Y-self.loc)*np.sqrt(self.df + self.dim)/(np.sqrt(self.Q + self.df))

            T_val = 0.5 + T_input*special.gamma((self.df+self.dim+1)/2)*special.hyp2f1(
                0.5,(self.df+self.dim+1)/2,1.5,(-T_input**2)/(self.df+self.dim)
            )/(np.sqrt(np.pi*(self.df+self.dim))*special.gamma((self.df+self.dim)/2))
            return T_val
        
        def t(self,Y):
            # logpdf terms
            c = special.gamma(
                (self.df+self.dim)/2)/(
                    special.gamma(self.df/2)*(self.dim/2)*np.pi * self.df)
            det_disp = 1/(np.prod(np.diag(self.A)))**2

            return c / (np.sqrt(det_disp)*(1+self.Q(Y)/self.df)**(self.df/2)*(1+self.dim/self.df))
        
        def logpdf(self,Y):
            # should return something 1 x n_dat

            return np.log(self.t(Y)) + np.log(2*self.T(Y))

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