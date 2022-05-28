import numpy as np
from ..metrics import misclassification_error
from ..base import BaseEstimator
from typing import Callable, NoReturn


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        self.D_ = np.full(n_samples, (1 / n_samples))

        self.models_ = []

        # Iterate through classifiers
        for _ in range(self.iterations_):
            clf = self.wl_()
            clf.fit(X, y)

            # calculate alpha
            EPS = 1e-10
            error = clf.loss(X, y)
            self.weights_ = 0.5 * np.log((1.0 - error + EPS) / (error + EPS))

            # calculate predictions and update weights
            predictions = clf.predict(X)

            self.D_ *= np.exp(-self.weights_ * y * predictions)
            # Normalize to one
            self.D_ /= np.sum(self.D_)

            # Save classifier
            self.models_.append(clf)



        # err = np.zeros(self.iterations_)
        # self.D_ = np.full_like(y, 1 / (X.shape[1]))
        # for i in range(self.iterations_):
        #     wl = self.wl()
        #     self.models_[i] = np.multiply(wl._fit(X, y), self.D_)
        #     y_pred = wl._predict(X)
        #     err[i] = self.compute_error(y, y_pred, self.D_[i])
        #     self.weights_ = self.compute_alpha(err[i])
        #     self.D_[:, i] = self.normalize(self.update_weights(self.D_[i], alpha, y, y_pred))
        # return np.sign(sum(self.D_ * self.models_))

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self.partial_loss(X, y, len(self.models_))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        clf = self.wl_()
        T_models = self.models_[:T]
        clf_preds = [clf.predict(X) * self.weights_ for clf in T_models]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred


    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.partial_predict(X, T), True)

    def compute_error(y, y_pred, w_i):
        '''
        Calculate the error rate of a weak classifier m. Arguments:
        y: actual target value
        y_pred: predicted value by weak classifier
        w_i: individual weights for each observation

        Note that all arrays should be the same length
        '''
        return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))

    def compute_alpha(error):
        '''
        Calculate the weight of a weak classifier m in the majority vote of the final classifier.
        error: error rate from weak classifier.
        '''
        return 0.5 * (np.log((1 - error) / error))

    def update_weights(w_i, alpha, y, y_pred):
        '''
        Update individual weights w_i after a boosting iteration. Arguments:
        w_i: individual weights for each observation
        y: actual target value
        y_pred: predicted value by weak classifier
        alpha: weight of weak classifier used to estimate y_pred
        '''
        return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))

    def normalize(w_i):
        '''
        : Normalize weights.
        '''
        return w_i / sum(w_i)