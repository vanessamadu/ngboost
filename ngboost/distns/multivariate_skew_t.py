"""
"""
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore

from scipy.special import gammaln, digamma, gamma
from scipy.stats import t, multivariate_normal, chi2
import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import conversion, default_converter

# import R packages
base = importr('base')
sn = importr('sn')

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
        VQ_val = self.VQ(Y)
        precision_val = self.precision

        grad_loc = np.matmul( precision_val, (1 + ( self.q(Y) * self.r(Y) ) / (self.df + self.d)) * VQ_val * (Y - self.loc) ) - \
                    np.sqrt(VQ_val) * self.r(Y) * self.eta
        grad_v_disp = 0.5 * np.matmul(np.matmul(self.duplication, np.kron(precision_val, precision_val)),
                                ((1 + self.q(Y) * self.r(Y) / (self.df + self.d) ) * VQ_val * np.outer( Y - self.loc, Y - self.loc) - self.disp).flatten('F'))
        grad_eta = np.sqrt(VQ_val) * self.r(Y) * (Y - self.loc)
        grad_df = 0.5 * (digamma( (self.df + self.d + 1) / 2 ) - digamma( self.df / 2 ) + 1 - \
                         (VQ_val * self.T2bar(Y) + self.Bbar(Y) + np.log( 1 + self.Q(Y) / self.df))
                        )
        return np.concatenate([grad_loc, grad_v_disp, grad_eta, [grad_df]])

    def metric(self):

        """

        Returns:
            self.N, self.n_params, self.n_params shaped array containing the fisher information for
             the ith observation in the last two indices.

        """
        precision_val = self.precision
        eta_bar_val = self.eta_bar

        F_loc_loc = ( (self.df + self.d) / (self.df + self.d + 2) ) * precision_val + \
                    ( 2 / (self.df + self.d + 1)) * ( (self.df + self.d) / (self.df + self.d - 1)) * self.M(self.r, self.r, 2, 4) * \
                        (np.dot(eta_bar_val, eta_bar_val)* precision_val - np.outer(self.eta, self.eta)) + \
                    (2 * (self.df + self.d) / (self.df + self.d -1 )) * self.M(self.r, self.r, 0, 6) * np.outer(self.eta, self.eta)

    ## Aux functions 

    def VQ(self, y):
        return (self.df + self.d) / (self.df + self.Q(y))

    def q(self, y):
        return np.sqrt(self.VQ(y)) * np.dot( self.eta, y - self.loc)

    def q2(self, y):
        return self.q(y) * np.sqrt( (self.df + self.d + 2) / (self.df + 2))

    def T(self, y):
        return t.cdf(self.q(y), loc = 0, scale = 1, df = self.df + self.d)

    def r(self,y):
        return t.pdf(self.q(y), loc = 0, scale = 1, df = self.df + self.d) / self.T(y)

    def B(self,y):
        pass

    def T2bar(self,y):
        return t.cdf(self.q2(y), loc = 0, scale = 1, df = self.df + self.d + 2) / self.T(y)

    def Bbar(self,y):
        return self.B(y)/self.T(y)

    def duplication(self):
        pass

    ## Fisher aux functions

    def M(self, g, h, i , j , k = 0):
        pass

    def eta_bar(self):
        A_inv = np.linalg.inv(self.A)
        return np.matmul(np.matmul(np.diag(np.exp(-self.rho)), A_inv), self.eta)

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
        global nu0
        nu0 = 4
        n_params = int(1 + d * (d + 5) / 2)
        scores = [MVStLogScore]
        multi_output = True

        def __init__(self, params):
            super().__init__(params)

            self.d = d

            self.loc = np.array(params[:d])
            self.rho = np.array(params[d:2*d])
            self.v_star_A = np.array(params[2*d: int(d*(d+3)/2)])
            self.eta = np.array(params[int(d*(d+3)/2):int(d*(d+5)/2)])
            self.nu_tilde = params[-1]
            
        def logpdf(self, Y):
            """_summary_

            Args:
                Y (_type_): _description_

            Returns:
                _type_: _description_
            """

            Q_val = self.Q(Y)

            term1 = - (self.d / 2) * np.log(np.pi * self.df) \
                    + gammaln((self.df + self.d) / 2) \
                    - gammaln(self.df / 2)
            
            term2 = np.sum(self.rho)

            term3 = - (self.df / 2) * (1 + self.d / self.df) * np.log(1 + Q_val / self.df)

            term4 = np.log(2 * t.cdf(
                np.sqrt(
                    (self.df + self.d) / (self.df + Q_val) 
                    ) * np.dot(
                        self.eta, Y - self.loc
                    )
            , df = self.df + self.d))

            return term1 + term2 + term3 + term4

        def fit(Y):
            Y_np = np.ascontiguousarray(Y, dtype=np.float64)
            n_rows, n_cols = Y_np.shape

            #                 # Explicitly construct an R matrix (column-major order)
            r_matrix = robjects.r['matrix'](
                robjects.FloatVector(Y_np.ravel(order='F')), 
                nrow=n_rows, 
                ncol=n_cols
                )

            robjects.r.assign("Y_mat", r_matrix)
            fit = robjects.r('sn::selm(Y_mat ~ 1, family = "ST")')
            dp = robjects.r['slot'](fit,"param").rx2('dp')


            # Extract parameter list (dp: direct parameters xi, Omega, alpha, nu)
            xi = np.array(dp.rx2('beta')) if 'beta' in dp.names else np.array(dp.rx2('xi'))
            disp = np.array(dp.rx2('Omega'))
            skew = np.array(dp.rx2('alpha'))
            df = float(np.array(dp.rx2('nu'))[0])

            #             # find A and rho from disp
            L = np.linalg.cholesky(disp)
            rho = -np.log(np.diagonal(L) ** 2)
            B_tril = np.multiply(np.tril(L, k=-1), np.sqrt(rho))
            np.fill_diagonal(B_tril, 1)
            A = np.linalg.inv(B_tril)

            stds = np.sqrt(np.diag(disp))
            eta = skew / stds
            nu_tilde = np.log(df - nu0)

            mask = A != 0
            v_star_A = A[mask]
            return np.array([xi, rho, v_star_A, eta, nu_tilde])
        
        def rv(self):
            """_summary_

            Returns:
                _type_: _description_
            """
            u_star = multivariate_normal(mean = np.zeros(self.d + 1), cov = self.omega_star).rvs()
            v = chi2(df = self.df).rvs() / self.df
            z = self.stds * u_star[1:] * np.sign(u_star[0])
            return self.loc + z / np.sqrt(v)

        def rvs(self, n):
            """_summary_

            Args:
                n (_type_): _description_

            Returns:
                _type_: _description_
            """
            return [self.rv() for _ in range(n)]

        def sample(self, n):
            """_summary_

            Args:
                n (_type_): _description_

            Returns:
                _type_: _description_
            """
            return self.rvs(n)

        @property
        def A(self):
            lt = np.eye(self.d)
            rows, cols = np.tril_indices(d, k=-1) 
            lt[rows,cols] = self.v_star_A
            return lt
        
        @property
        def disp(self):
            """_summary_

            Returns:
                _type_: _description_
            """
            A_inv = np.linalg.inv(self.A)
            return np.matmul(
                        np.matmul(
                            np.transpose(A_inv), np.diag(np.exp(-2 * self.rho ))),
                            A_inv)

        @property
        def precision(self):
            A_val = self.A
            return np.matmul(np.matmul(A_val, np.diag(np.exp(2 * self.rho))), np.transpose(A_val))

        @property
        def stds(self):
            """_summary_

            Returns:
                _type_: _description_
            """
            return np.sqrt(np.diag(self.disp))

        @property
        def corr(self):
            """_summary_

            Returns:
                _type_: _description_
            """
            return self.disp / np.outer(self.stds, self.stds)

        @property
        def omega_star(self):
            """_summary_

            Returns:
                _type_: _description_
            """
            delta_col = self.delta.reshape(-1, 1)   
            
            return np.block(
                [[1, np.transpose(delta_col)],
                 [delta_col, self.corr]]
            )

        @property
        def df(self):
            """_summary_

            Returns:
                _type_: _description_
            """
            return nu0 + np.exp(self.nu_tilde)

        @property
        def skew(self):
            return self.stds * self.eta

        def Q(self, Y):
            """_summary_

            Args:
                Y (_type_): _description_

            Returns:
                _type_: _description_
            """
            scaled_y0 = np.matmul(np.transpose(self.A), Y - self.loc)

            return np.matmul(
                np.matmul(np.transpose(scaled_y0), np.diag(np.exp(2 * self.rho))),
                scaled_y0
            )

        @property
        def delta(self):
            """_summary_

            Returns:
                _type_: _description_
            """
            return np.matmul(
                self.corr, self.skew
                ) / (
                     np.sqrt( 1 + 
                        np.matmul(
                            np.matmul(
                                np.transpose(self.skew),
                                self.corr),
                        self.skew)
                    )
                )
            
        @property
        def mu(self):
            return self.delta * np.sqrt(self.df / np.pi) * gamma( (self.df - 1) / 2 ) / gamma(self.df / 2)

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
            """_summary_

            Returns:
                _type_: _description_
            """
            return self.loc + self.stds * self.mu
            

        def cov(self):
            """_summary_
            currently incorrect!
            Returns:
                _type_: _description_ 
            """
            outer_product_term = self.stds * self.mu
        
            return ( self.df / (self.df - 2) ) * self.disp - np.outer(outer_product_term, outer_product_term)

    return MVSt

