from sklearn.base import TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np


class MCCABlock(TransformerMixin):

    def __init__(self, block_scores, block_loadings, centerer):

        self.block_scores_ = block_scores
        self.block_loadings_ = block_loadings
        self.centerer_ = centerer

    def transform(self, X):
        """
        Projects a new data matrix onto the block loadings.

        Parameters
        ----------
        X: array-like, shape (n_new_samples, n_features)
            The data to project.

        Output
        ------
        scores: array-like, shape (n_new_samples, n_components)
            The projections of the new data.
        """
        return self.centerer_.transform(X).dot(self.block_loadings_)

    def inverse_transform(self, scores):
        """
        Transforms scores back to the original space.

        Parameters
        ----------
        scores: array-like, shape (n_samples, n_components)
            The CCA scores.

        Output
        ------
        X_hat: array-like, shape (n_samples, n_features)
            The predictions.
        """

        reconst = scores.dot(self.loadings_.T)

        m = self.centerer_.mean_
        if m is not None:
            reconst += m.reshape(1, -1)
        return reconst


class KMCCABlock(TransformerMixin):
    def __init__(self, kernel='linear', kernel_params=None,
                 filter_params=False, n_jobs=None):
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.filter_params = filter_params
        self.n_jobs = n_jobs

    def _get_kernel(self, X, Y=None):

        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, n_jobs=self.n_jobs,
                                **self.kernel_params)

    def set_fit_values(self, dual_vars, block_scores, centerer, X_fit):
        self.dual_vars_ = dual_vars
        self.block_scores_ = block_scores
        self.centerer_ = centerer
        self.X_fit_ = X_fit
        return self

    def transform(self, X):
        K = self._get_kernel(X, self.X_fit_)
        if self.centerer_ is not None:
            K = self.centerer_.transform(K)

        # TODO: is this right? Do we need additional scaling?
        return np.dot(K, self.dual_vars_)
