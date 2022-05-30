from __future__ import annotations
import math
<<<<<<< HEAD
import numpy as np
from scipy.stats import norm
=======
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from numpy.linalg import inv, det, slogdet
import plotly.express as px
>>>>>>> bd052ab5d836942da6c931c90eca40228aa7c07a


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X)
        self.var_ = np.var(X, ddof=1)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        def normpdf(x, mean, var):
            denom = (2 * math.pi * var) ** .5
            num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
            return (num / denom)

        pdf_list = []
<<<<<<< HEAD
        for i in X:
            pdf_list.append(normpdf(i, self.mu_, self.var_))
=======
        for i in range(1000):
            pdf_list.append(normpdf(X[i], self.mu_, self.var_))
        print(pdf_list)
>>>>>>> bd052ab5d836942da6c931c90eca40228aa7c07a
        return pdf_list

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
<<<<<<< HEAD
        def calc_loglikelihood(residuals):
            return -0.5 * (np.log(np.linalg.det(sigma)) + residuals.T.dot(np.linalg.inv(sigma)).dot(
                residuals) + 2 * np.log(2 * np.pi))

        sigma = np.array(sigma)
        residuals = np.subtract(X, mu)

        loglikelihood = np.apply_along_axis(calc_loglikelihood, 1, residuals)
        loglikelihoodsum = loglikelihood.sum()
=======
        return -0.5 * (np.log(np.linalg.det(sigma)) +
                       (X - mu).T.dot(np.linalg.inv(sigma)).dot(X - mu) +
                       2 * np.log(2 * np.pi))
>>>>>>> bd052ab5d836942da6c931c90eca40228aa7c07a

        return loglikelihoodsum

class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X, rowvar=False)
        print(self.mu_)
        print(self.cov_)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
<<<<<<< HEAD

        pdf_list = []
        for i in range(1000):
            pdf_list.append(norm.pdf(X[i], self.mu_, self.cov_))
=======
        def normpdf(x, mean, cov):
            denom = (2 * math.pi * cov) ** .5
            num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * cov))
            return num/denom

        pdf_list = []
        for i in range(1000):
            pdf_list.append(normpdf(X[i], self.mu_, self.cov_))
        print(pdf_list)
>>>>>>> bd052ab5d836942da6c931c90eca40228aa7c07a
        return pdf_list

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
<<<<<<< HEAD
        def calc_loglikelihood(residuals):
            return -0.5 * (np.log(np.linalg.det(cov)) + residuals.T.dot(np.linalg.inv(cov)).dot(
                residuals) + 2 * np.log(2 * np.pi))

        cov = np.array(cov)
        residuals = np.subtract(X, mu)

        loglikelihood = np.apply_along_axis(calc_loglikelihood, 1, residuals)
        loglikelihoodsum = loglikelihood.sum()

        return loglikelihoodsum
=======
        X = np.array([x - mu for x in X])
        return -0.5 * (np.log(np.linalg.cond(cov)) +
                       X.dot(np.linalg.inv(cov)).dot(X.T) +
                       2 * np.log(2 * np.pi))
>>>>>>> bd052ab5d836942da6c931c90eca40228aa7c07a
