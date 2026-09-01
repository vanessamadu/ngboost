"""
"""
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore

from scipy.special import gammaln, digamma
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
        VQ_val = self.VQ(Y)

        grad_loc = np.matmul( self.precision, (1 + ( self.q(Y) * self.r(Y) ) / (self.df + self.d)) * VQ_val * (Y - self.loc) ) - \
                    np.sqrt(VQ_val) * self.r(Y) * self.eta
        grad_v_disp = 0.5 * np.matmul(np.matmul(self.duplication, np.kron(self.precision, self.precision)),
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
        pass

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
            self.A = None

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
            pass

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
            return np.matmul(
                np.matmul(
                    np.transpose(self.A), np.diag(np.exp(2 * self.rho ))
                ), self.A
            )

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
            return self.nu0 + np.exp(self.nu_tilde)

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
            disp_val = self.disp
            return self.loc + np.sqrt( (2 * self.df) / np.pi) * np.matmul(disp_val, self.eta) / \
                ( (self.df - 2) * np.sqrt(1 + np.matmul( np.matmul( np.transpose(self.eta), disp_val ), self.eta) ) )
            

        def cov(self):
            """_summary_

            Returns:
                _type_: _description_
            """
            disp_value = self.disp
            const = self.df/( (self.df - 2) * (self.df - 4))
            outer_product_term = ( 2 * (self.df - 4) * np.matmul( np.outer(self.eta, self.eta) , disp_value ) ) / \
                (np.pi * (self.df - 2) * (1 + np.matmul( np.matmul( np.transpose(self.eta), disp_value ), self.eta ) ) )

            return const * np.matmul( disp_value , np.eye(self.d) - outer_product_term)

    return MVSt
