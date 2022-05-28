from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_indx, best_thr, best_thr_err, best_sign = np.inf, None, np.inf, None
        for feature_indx in range (X.shape[1]):
            thr, thr_err = feature_indx._find_threshold(feature_indx[:, feature_indx], y, 1)
            if thr_err < best_thr_err:
                best_thr_err = thr_err
                best_thr = thr
                best_indx = feature_indx
                best_sign = 1

            thr, thr_err = feature_indx._find_threshold(feature_indx[:, feature_indx], y, -1)
            if thr_err < best_thr_err:
                best_thr_err = thr_err
                best_thr = thr
                best_indx = feature_indx
                best_sign = -1

            self.threshold_, self.j_, self.sign_ = best_thr, best_indx, best_sign


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y_pred = np.full(shape=X.shape[0], fill_value=self.sign_)
        y_pred[X[:, self.j_] < self.threshold_] = -1 * self.sign_
        return y_pred

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        best_thr, best_thr_loss = None, np.inf
        for candidate_thr in np.unique(values):
            y_pred = np.full_like(values, sign)
            y_pred[values < candidate_thr] = -1 * sign
            y_pred_loss = misclassification_error(y,y_pred,True)
            if y_pred_loss < best_thr_loss:
                best_thr_loss = y_pred_loss
                best_thr = candidate_thr
        return best_thr, best_thr_loss

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
        return misclassification_error(y, self._predict(X), True)
