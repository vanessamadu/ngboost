"""
"""
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore

from scipy.special import gammaln, digamma, gamma
from scipy.stats import t, multivariate_normal, chi2
import scipy.integrate as integrate
import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

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
        precision_val = self.precision()
        r_val = self.r(Y)

        grad_loc = np.matmul( precision_val, (1 + ( self.q(Y) * r_val ) / (self.df + self.d)) * VQ_val * (Y - self.loc) ) - \
                    np.sqrt(VQ_val) * r_val  * self.eta
        grad_v_disp = 0.5 * np.matmul(np.matmul(self.duplication(), np.kron(precision_val, precision_val)),
                                ((1 + self.q(Y) * r_val  / (self.df + self.d) ) * VQ_val * np.outer( Y - self.loc, Y - self.loc) - self.disp).flatten('F'))
        grad_eta = np.sqrt(VQ_val) * r_val  * (Y - self.loc)
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
        precision_val = self.precision()
        eta_bar_val = self.eta_bar()
        duplication_val = self.duplication()
        Upsilonbar_val = self.Upsilonbar()
        Upsilon_val = self.Upsilon()
        df_d = self.df + self.d
        disp_val = self.disp()

        F_xi_xi = ( (df_d) / (df_d + 2) ) * precision_val + \
                    ( 2 / (df_d + 1)) * ( (df_d) / (df_d - 1)) * self.M(self.r, self.r, 2, 4) * \
                        ((np.linalg.norm(eta_bar_val)**2) * precision_val - np.outer(self.eta, self.eta)) + \
                    (2 * (df_d) / (df_d -1 )) * self.M(self.r, self.r, 0, 6) * np.outer(self.eta, self.eta)

        F_xi_v_omega = np.sqrt( (df_d) / (df_d - 1) ) * ( self.b(self.df) / (self.b(df_d - 1)) ) * \
            ( 
                ( 2 * df_d * self.M(self.r,1,4,1) - df_d * self.M(self.r,1,2,1) + \
                 np.sqrt(df_d) * (self.M(self.r,self.r,5,1) - self.M(self.r,self.r,3,1)) * np.linalg.norm(eta_bar_val)) * \
                 np.matmul(np.matmul(self.eta, np.transpose(Upsilon_val.flatten('F'))), duplication_val) + \
                (2 * self.M(self.r,1,2,3) + (1 / np.sqrt(df_d)) * self.M(self.r, self.r, 3,3) * np.linalg.norm(eta_bar_val)) * \
                np.matmul((np.kron(Upsilonbar_val, np.transpose(self.eta)) + np.kron(np.transpose(self.eta), Upsilonbar_val)+ \
                 np.matmul(self.eta, np.transpose(Upsilonbar_val.flatten("F")))), duplication_val) - (self.M(self.r,1,0,3) + (1 / np.sqrt(df_d)) * \
                np.linalg.norm(eta_bar_val) * self.M(self.r, self.r,1,3))* \
                    np.matmul(np.matmul(self.eta, np.transpose(Upsilonbar_val.flatten('F'))),duplication_val)
            )
        F_xi_eta = 2 * np.sqrt( (df_d) / (df_d - 1) ) * ( self.b(self.df) / (self.b(df_d - 1)) ) * (
            (df_d * self.M(self.r,1,2,1) - np.sqrt(df_d) * np.linalg.norm(self.eta) * self.M(self.r,1,1,3)**2) * np.matmul(Upsilon_val,disp_val) + \
            (self.M(self.r,1,0,3) + (1 / np.sqrt(df_d)) * self.M(self.r,self.r,1,3) * np.linalg.norm(eta_bar_val)) * np.matmul(Upsilonbar_val, disp_val)
        )

        F_xi_nu = np.sqrt( (df_d) / (df_d - 1) ) * ( self.b(self.df) / (self.b(df_d - 1)) ) * ( 
            ((self.df + 1) / self.df) * self.M(self.r, self.T2bar,0,5) + self.M(self.r, self.Bbar,0,3) + self.M(self.r,1,0,3,1) - \
            self.M(self.r,1,0,3)* self.psi_diff( df_d / 2, (self.df + 1) / 2)
        ) * self.eta

        F_v_omega_v_omega = 0.5 * np.matmul(np.transpose(duplication_val) ,
                                             np.matmul(
                                                 (df_d / (df_d + 2)) * np.kron(precision_val,precision_val)) - \
                                                    (1 / (df_d + 2)) * np.outer(precision_val.flatten('F'), precision_val.flatten('F'))
                                                      , duplication_val) + \
                            0.5 * (np.linalg.norm(eta_bar_val) ** 2) * np.matmul(np.transpose(duplication_val),
                                np.matmul(df_d * self.M(self.r, self.r, 6, 0) * np.outer(Upsilon_val.flatten('F'), Upsilon_val.flatten('F')) + \
                                    (df_d / (df_d - 1)) * self.M(self.r, self.r,4,2) * (2 * np.kron(Upsilonbar_val,Upsilon_val) + \
                                    2 * np.kron(Upsilon_val,Upsilonbar_val) + \
                                    np.matmul(Upsilon_val.flatten('F'), np.transpose(Upsilonbar_val.flatten('F'))) + \
                                    np.matmul(Upsilonbar_val.flatten('F'), np.transpose(Upsilon_val.flatten('F')))) + \
                                    (df_d / ((df_d + 1) * (df_d - 1))) * self.M(self.r, self.r,2,4) * \
                                    (2 * np.kron(Upsilonbar_val,Upsilonbar_val) + np.outer(Upsilonbar_val.flatten('F'), Upsilonbar_val.flatten('F')))
                                , duplication_val)
                            )

        F_v_omega_eta = df_d * self.M(self.r,self.r,4,0)* np.matmul(np.transpose(duplication_val), np.outer(Upsilon_val.flatten('F'), self.eta), disp_val) + \
                        (df_d / (df_d - 1)) * self.M(self.r,self.r,2,2) * np.matmul(np.matmul(np.transpose(duplication_val),
                                                                                    np.kron(Upsilonbar_val, self.eta) + np.kron(self.eta,Upsilonbar_val) + \
                                                                                        np.outer(Upsilonbar_val.flatten('F'), self.eta)) , disp_val)

        F_v_omega_nu = -0.25 * (df_d / (df_d + 2) + self.psi_diff(self.df / 2 , (df_d + 2) / 2) - self.psi_diff( (df_d + 1) / 2, self.df / 2)) * \
                        np.matmul(np.transpose(duplication_val , precision_val.flatten('F'))) - \
                        0.5 * np.sqrt(df_d) * np.linalg.norm(eta_bar_val) * (
                            ( (df_d / (df_d - 1)) * self.M(self.r, self.T2bar, 3,2) + self.M(self.r, self.Bbar, 3, 0)) * \
                                np.matmul(np.transpose(duplication_val),Upsilon_val.flatten('F')) + \
                            ( (df_d / ((df_d + 1) * (df_d - 1))) * self.M(self.r, self.T2bar,1,4) + (1 / (df_d - 1)) * self.M(self.r, self.Bbar,1,2) ) * \
                            np.matmul(np.transpose(duplication_val , Upsilonbar_val.flatten('F')))   
                        )        

        F_eta_eta = 2 * df_d * self.M(self.r, self.r,2,0) * np.matmul(disp_val, np.matmul(Upsilon_val, disp_val)) + \
                    2 * (df_d / (df_d - 1)) * self.M(self.r, self.r,0,2) * np.matmul(disp_val, np.matmul(Upsilonbar_val, disp_val))

        F_eta_nu = - np.sqrt(df_d) * ( (df_d / (df_d - 1)) * self.M(self.r,self.T2bar, 1,2) + self.M(self.r, self.Bbar,1,0)) * \
                    np.matmul(disp_val, self.eta) / np.linalg.norm(eta_bar_val)

        F_nu_nu = 0.5 * (((self.df + 2) / self.df) * ((df_d**2)/( (df_d + 1) * (df_d - 1))) * self.M(self.T2bar, self.T2bar, 0 ,4) + self.M(self.Bbar,self.Bbar,0,0) + \
                         (2 * df_d / (df_d - 1)) * self.M(self.T2bar, self.Bbar,0,2) ) + \
                    0.5 * (self.psi_diff((df_d - 1) / 2, (df_d) / 2) ** 2 + self.psi_diff((df_d - 1) / 2, (self.df) / 2) ** 2 + \
                           self.psi_diff((self.df) / 2, (df_d) / 2) - self.psi_diff((df_d - 1) / 2, (self.df) / 2) * self.psi_diff((df_d - 1) / 2, (df_d) / 2)) - \
                    0.25 * (self.psi_diff( (df_d + 1) / 2 , self.df / 2) + 1) ** 2
                       
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
        return integrate.quad(lambda x: t.pdf(x, df = self.df + self.d) *  np.log(1 + x**2 / (self.df + self.d)), -np.inf, self.q(y))[0]

    def T2bar(self,y):
        return t.cdf(self.q2(y), loc = 0, scale = 1, df = self.df + self.d + 2) / self.T(y)

    def Bbar(self,y):
        return self.B(y)/self.T(y)

    def duplication(self):
        output = np.zeros([int(self.d * (self.d + 1) / 2), self.d ** 2])
        for jj in range(self.d):
            for ii in range (jj, self.d):
                u = np.zeros(int(self.d * (self.d + 1) / 2))
                u[int(jj * self.d + ii - jj * (jj + 1) / 2)] = 1
                T = np.zeros([self.d, self.d])
                T[ii,jj] = 1
                T[jj,ii] = 1
                output += np.outer(u, T.flatten('F'))
        return output

    ## Fisher aux functions

    def M(self, g, h, i , j , k = 0):
        pass

    def eta_bar(self):
        A_inv = np.linalg.inv(self.A)
        return np.matmul(np.matmul(np.diag(np.exp(-self.rho)), A_inv), self.eta)

    def Upsilon(self):
        return np.outer(self.eta, self.eta)/np.linalg.norm(self.eta_bar())

    def Upsilonbar(self):
        return self.precision() - self.Upsilon()

    def b(self, k):
        return 2 * t.pdf(0, df = self.df + k)

    @staticmethod
    def psi_diff(a,b):
        return digamma(a) - digamma(b)

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
        nu0 = 2
        n_params = int(1 + d * (d + 5) / 2)
        scores = [MVStLogScore]
        multi_output = True

        def __init__(self, params):
            super().__init__(params)

            self.d = d

            self.loc = np.array(params[:d])
            self.rho = np.array(params[d:2*d])
            self.v_star_A = np.array(params[2*d: int(2*d + d*(d-1)/2)])
            self.eta = np.array(params[int(2*d + d*(d-1)/2):int(d + 2*d + d*(d-1)/2)])
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

        @staticmethod
        def fit(Y:np.ndarray):
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
            xi = np.squeeze(np.array(dp.rx2('beta')) if 'beta' in dp.names else np.array(dp.rx2('xi')))
            disp = np.array(dp.rx2('Omega'))
            skew = np.array(dp.rx2('alpha'))
            df = float(np.array(dp.rx2('nu'))[0])

            #             # find A and rho from disp
            Omega_inv = np.linalg.inv(disp)
            L = np.linalg.cholesky(Omega_inv)
            diagL = np.diag(L)
            A = L / diagL[np.newaxis, :]
            rho = np.log(diagL)

            eta = skew 
            if df <= nu0:
                nu_tilde = 1e-5  # set to small value if fit degrees of freedom is less than nu0
            else:
                nu_tilde = np.log(df - nu0) 

            mask = np.tril(A, k=-1) != 0
            v_star_A = A[mask]

            return np.concatenate([xi, rho, v_star_A, eta, [nu_tilde]])
        
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

