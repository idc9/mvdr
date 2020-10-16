from numbers import Number
import numpy as np
from scipy.linalg import block_diag
from itertools import combinations
from warnings import warn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import ledoit_wolf, oas
from textwrap import dedent

from mvlearn.utils import check_Xs

from mvdr.mcca.view_processing import center_views, \
    split, initial_svds
from mvdr.linalg_utils import eigh_wrapper, svd_wrapper, normalize_cols


class MCCA(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=1, center=True,
                 signal_ranks=None, regs=None,
                 i_mcca_method='auto'):

        self.n_components = n_components
        self.center = center
        self.signal_ranks = signal_ranks
        self.regs = regs
        self.i_mcca_method = i_mcca_method

    def fit(self, Xs, precomp_svds=None):

        Xs = check_Xs(Xs, multiview=True, return_dimensions=False)

        if self.signal_ranks is not None:
            # perhaps give option to force
            out = i_mcca(Xs,
                         signal_ranks=self.signal_ranks,
                         n_components=self.n_components,
                         center=self.center,
                         regs=self.regs,
                         precomp_svds=precomp_svds,
                         method=self.i_mcca_method)
        else:
            out = mcca_gevp(Xs,
                            n_components=self.n_components,
                            center=self.center,
                            regs=self.regs)

        self.common_norm_scores_ = out['common_norm_scores']
        self.cs_col_norms_ = out['cs_col_norms']
        self.evals_ = out['evals']

        n_views = len(Xs)
        self.views_ = [None for b in range(n_views)]
        for b in range(n_views):
            bs = out['view_scores'][b]
            bl = out['view_loadings'][b]
            cent = out['centerers'][b]
            self.views_[b] = MCCAView(view_scores=bs,
                                      view_loadings=bl,
                                      centerer=cent)

        return self

    @property
    def n_views_(self):
        if hasattr(self, 'views_'):
            return len(self.views_)

    @property
    def view_loadings_(self):
        if hasattr(self, 'views_'):
            return [blck.view_loadings_ for blck in self.views_]

    @property
    def view_scores_(self):
        if hasattr(self, 'views_'):
            return [blck.view_scores_ for blck in self.views_]

    @property
    def n_components_(self):
        if hasattr(self, 'common_norm_scores_'):
            return self.common_norm_scores_.shape[1]

    def fit_transform(self, Xs, precomp_svds=None):
        self.fit(Xs, precomp_svds=precomp_svds)
        return self.common_norm_scores_

    def transform(self, Xs):
        common_proj = sum(self.views_[b].transform(Xs[b])
                          for b in range(self.n_views_))
        return common_proj * (1 / self.cs_col_norms_)

    def inverse_transform(self, Xs):
        return [self.views_[b].inverse_transform(Xs[b])
                for b in range(self.n_views_)]


class MCCAView(TransformerMixin):

    def __init__(self, view_scores, view_loadings, centerer):

        self.view_scores_ = view_scores
        self.view_loadings_ = view_loadings
        self.centerer_ = centerer

    def transform(self, X):
        """
        Projects a new data matrix onto the view loadings.

        Parameters
        ----------
        X: array-like, shape (n_new_samples, n_features)
            The data to project.

        Output
        ------
        scores: array-like, shape (n_new_samples, n_components)
            The projections of the new data.
        """
        return self.centerer_.transform(X).dot(self.view_loadings_)

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


_mcca_docs = dict(
    Xs=dedent("""
        Xs : list of array-likes or numpy.ndarray
            The list of data matrices each shaped (n_samples, n_features_b).
        """),

    basic=dedent("""
        n_components: int, None, str
            Number of components to compute. If None, will compute as many as possible. If str must be one of ['min', 'max']. If 'min' will compute the min of the input feature sizes number of components. If 'max' will compute the min of the input feature sizes number of components.

        center: bool, list
            Whether or not to initially mean center the data as in PCA.
            Different options for each data view can be provided by inputting a list of bools.

        regs: None, float, str, list
            MCCA regularization for each data view, which can be important for high dimensional data. A value of 0 or None for all views corresponds to SUMCORR-AVGVAR MCCA. A value of 1 corresponds to partial least squares SVD in the case of 2 views and a natural generalization of this method for more than two views. Simple default regularization values can be obtained by passing in one of ['lw', 'oas'], which will use sklearn.covariance.ledoit_wolf or sklearn.covariance.oas. If a single value (None, float or str) is passed in that value will be used for every view. Different options for each data view can be provided by inputting a list.
        """),

    score_out=dedent("""
        evals: array-like, (n_components, )
            The MCCA eigenvalues.

        view_scores: list of array-like
            Projections of each data view onto its view scores.

        common_norm_scores: array-like, (n_samples, n_components)
            Normalized sum of the view scores.

        cs_col_norms: array-like, (n_components, )
            Column nomrs of the sum of the view scores.
            Useful for projecting new data
        """),

    other_out=dedent("""
        centerers: list of sklearn.preprocessing.StandardScaler
            The mean centering object for each view.

        view_loadings: list of array-like
            The loadings for each view used to project new data.
            Each entry of the list is shaped (n_features_b, n_components).
        """)

)


MCCA.__doc__ = dedent("""
    Multi-view canonical correlation analysis. Includes options for regularized MCCA and informative MCCA (i.e. where we first compute a low rank PCA).

    Parameters
    -----------
    {basic}

    signal_ranks: None, int, list
        The initial signal rank to compute i.e. rank of the SVD.
        If None, will compute the full SVD.
        Different values can be provided for each view by inputting a list.

    i_mcca_method: str
        Whether or not to use the SVD based method (only works with no regularization) or the gevp based method for informative MCCA. Must be one of ['auto', 'svd', 'gevp'].

    Attributes
    ----------
    views_: list of mvdr.mcca.MCCABlock.MCCABlock
        Containts the view level data for each data view.

    evals_: array-like, (n_components, )
            The MCCA eigenvalues.

    common_norm_scores_: array-like, (n_samples, n_components)
        Normalized sum of the view scores.

    cs_col_norms_: array-like, (n_components, )
        Column nomrs of the sum of the view scores.
        Useful for projecting new data.
    """.format(**_mcca_docs)
)


def mcca_gevp(Xs, n_components=None, center=True, regs=None):

    Xs, n_views, n_samples, n_features = check_Xs(Xs, multiview=True,
                                                  return_dimensions=True)

    if n_components is None:
        n_components = sum(n_features)
        # n_components = min(n_featues)
    elif n_components == 'min':
        n_components = min(n_features)
    elif n_components == 'max':
        n_components = max(n_features)

    if n_components > sum(n_features):
        warn("Requested too many components!")
    n_components = min(n_components, sum(n_features))

    # checking:
    # if no regularization
    # check all views are low dimensional
    # proably want sum(n_featues) <= n_samples or at least issue a warning if this is violated

    # center data views
    Xs, centerers = center_views(Xs, center=center)

    #########################################
    # solve generalized eigenvector problem #
    #########################################
    LHS, RHS = get_mcca_gevp_data(Xs=Xs, regs=regs)

    gevals, view_loadings = eigh_wrapper(A=LHS, B=RHS, rank=n_components)

    #################
    # Format output #
    #################

    # set view scores and loadings
    view_loadings = split(view_loadings, dims=n_features, axis=0)
    view_scores = [Xs[b].dot(view_loadings[b]) for b in range(n_views)]

    # common scores are the average of the view scores and are unit norm
    # this is also the flag mean of the subspaces spanned by the columns
    # of the views e.g. see (Draper et al., 2014)
    common_scores = sum(bs for bs in view_scores)

    # TODO: is this the behavior we want when regularization is used?
    common_norm_scores, col_norms = normalize_cols(common_scores)

    # enforce deterministic output due to possible sign flips
    common_norm_scores, view_scores, view_loadings = \
        mcca_det_output(common_norm_scores, view_scores, view_loadings)

    return {'view_scores': view_scores,
            'view_loadings': view_loadings,
            'common_norm_scores': common_norm_scores,
            'cs_col_norms': col_norms,
            'evals': gevals,
            'centerers': centerers}


mcca_gevp.__doc__ = dedent("""
    Computes multi-view canonical correlation analysis via the generalized eigenvector formulation of SUMCORR-AVGVAR.

    Parameters
    ----------
    {Xs}

    {basic}

    Output
    ------
    {score_out}

    {other_out}
    """.format(**_mcca_docs))


def i_mcca(Xs, signal_ranks=None,
           n_components=None, center=True, regs=None, precomp_svds=None,
           method='auto'):

    Xs, n_views, n_samples, n_features = check_Xs(Xs, multiview=True,
                                                  return_dimensions=True)

    if method == 'auto':
        if regs is not None:
            method = 'gevp'
        else:
            # TODO: decide which method to use more intelligently based
            # on the shape of the input matrices
            method = 'svd'

    if method == 'svd':
        # SVD won't method does not work with regularization
        assert regs is None or all(r is None for r in regs), \
            "SVD method cannot handle regularization."

    #######################
    # Compute initial SVD #
    #######################
    use_norm_scores = regs is None
    reduced, init_svds, centerers = \
        initial_svds(Xs=Xs, signal_ranks=signal_ranks,
                     normalized_scores=use_norm_scores,
                     center=center,
                     precomp_svds=precomp_svds)

    # set n_components
    n_features_reduced = [r.shape[1] for r in reduced]
    if n_components is None:
        n_components = sum(n_features_reduced)
    elif n_components == 'min':
        n_components = min(n_features_reduced)
    elif n_components == 'max':
        n_components = max(n_features_reduced)

    if n_components > sum(n_features_reduced):
        warn("Requested too many components!")
    n_components = min(n_components, sum(n_features_reduced))

    ################################
    # Compute MCCA on reduced data #
    ################################
    if method == 'svd':

        # left singluar vectors for each view
        bases = [reduced[b] for b in range(n_views)]
        results = flag_mean(bases, n_components=n_components)

        # rename flag mean output to correspond to MCCA conventions
        results['common_norm_scores'] = results.pop('flag_mean')
        results['evals'] = results.pop('sqsvals')
        results['cs_col_norms'] = np.sqrt(results['evals'])

        # map the view loadings back into the original feature space
        view_loadings = [None for _ in range(n_views)]
        for b in range(n_views):
            D_b = init_svds[b][1]
            V_b = init_svds[b][2]
            W_b = V_b * (1.0 / D_b)
            view_loadings[b] = W_b.dot(results['view_loadings'][b])
        results['view_loadings'] = view_loadings

    elif method == 'gevp':

        # compute MCCA gevp problem on reduced data
        results = mcca_gevp(Xs=reduced, n_components=n_components,
                            center=False,  # we already took care of this!
                            regs=regs)

        # map the view loadings back into the original feature space
        view_loadings = [None for _ in range(n_views)]
        for b in range(n_views):

            V_b = init_svds[b][2]

            if use_norm_scores:
                D_b = init_svds[b][1]
                W_b = V_b * (1.0 / D_b)
            else:
                W_b = V_b

            view_loadings[b] = W_b.dot(results['view_loadings'][b])
        results['view_loadings'] = view_loadings

    results['init_svds'] = init_svds
    results['centerers'] = centerers

    return results


i_mcca.__doc__ = dedent("""
    Computes informative multi-view canonical correlation analysis e.g. PCA-CCA.

    Parameters
    ----------
    {Xs}

    {basic}

    signal_ranks: None, int, list
        The initial signal rank to compute i.e. rank of the SVD.
        If None, will compute the full SVD.
        Different values can be provided for each view by inputting a list.

    precomp_svds: list of tuples
        Precomputed SVDs for each view. The tuples should be in the form (U, D, V) where U and V are the matrices whose columns are the left/right singluar vectors and D is the array of singular values.

    method: str
        Whether or not to use the SVD based method (only works with no regularization) or the gevp based method. Must be one of ['auto', 'svd', 'gevp'].

    Output
    ------
    {score_out}

    {other_out}

    init_svds: list of tuples
        The initial SVDs of each view.

    """.format(**_mcca_docs))


def get_mcca_gevp_data(Xs, regs=None, ret_lists=False):
    """
    Constructs the matrices for the MCCA generalized eigenvector problem.

    Parameters
    ----------
    Xs : list of array-likes or numpy.ndarray
        The list of data matrices each shaped (n_samples, n_features_b).

    regs: None, float, list of floats
        MCCA regularization for each data view, which can be important for high dimensional data. A value of 0 or None for all views corresponds to SUMCORR-AVGVAR MCCA. A value of 1 corresponds to partial least squares SVD in the case of 2 views and a natural generalization of this method for more than two views. Simple default regularization values can be obtained by passing in one of ['lw', 'oas'], which will use sklearn.covariance.ledoit_wolf or sklearn.covariance.oas. If a single value (None, float or str) is passed in that value will be used for every view. Different options for each data view can be provided by inputting a list.

    ret_lists: bool
        Return returns the views of the view matrices instead of the two matrices.

    Output
    ------
    LHS, RHS: array-like, (sum_b n_features_b, sum_b n_features_b)

    Left and right hand sides of GEVP

    LHS v = lambda RHS v

    """
    Xs, n_views, n_samples, n_features = check_Xs(Xs, multiview=True,
                                                  return_dimensions=True)

    regs = check_regs(regs=regs, n_views=n_views)

    LHS = [[None for b in range(n_views)]
           for b in range(n_views)]
    RHS = [None for b in range(n_views)]

    # cross covariance matrices
    for (a, b) in combinations(range(n_views), 2):
        LHS[a][b] = Xs[a].T @ Xs[b]
        LHS[b][a] = LHS[a][b].T

    # view covariance matrices, possibly regularized
    for b in range(n_views):
        if regs is None or regs[b] is None:
            RHS[b] = Xs[b].T @ Xs[b]

        elif isinstance(regs[b], Number):
            RHS[b] = (1 - regs[b]) * Xs[b].T @ Xs[b] + \
                regs[b] * np.eye(n_features[b])

        elif isinstance(regs[b], str):
            assert regs[b] in ['lw', 'oas']

            if regs[b] == 'lw':
                RHS[b] = ledoit_wolf(Xs[b])[0]

            elif regs[b] == 'oas':
                RHS[b] = oas(Xs[b])[0]
            RHS[b] *= n_samples  # put back on scale of X^TX as oppose to proper cov est returned by these functions

        LHS[b][b] = RHS[b]

    if not ret_lists:
        LHS = np.block(LHS)
        RHS = block_diag(*RHS)

    return LHS, RHS


def mcca_det_output(common_scores, view_scores=None, view_loadings=None):
    """
    Enforces determinsitic MCCA output. Makes largest absolute value entry
    of common scores positive.

    Parameters
    ----------
    common_scores: array-like

    view_scores: list of array-like or None

    view_loadings: list of array-like or None

    Output
    ------
    common_scores, view_scores, view_loadings

    """

    max_abs_cols = np.argmax(np.abs(common_scores), axis=0)
    signs = np.sign(common_scores[max_abs_cols, range(common_scores.shape[1])])
    common_scores = common_scores * signs
    for b in range(len(view_scores)):

        if view_scores is not None:
            view_scores[b] = view_scores[b] * signs

        if view_loadings is not None:
            view_loadings[b] = view_loadings[b] * signs

    return common_scores, view_scores, view_loadings


def check_regs(regs, n_views):
    """
    Checks the regularization paramters for each view.
    If the regulaization is not None, it must be a float between 0 and 1

    Parameters
    ----------
    regs: None, float, list of floats
        Process the regs argument i.e. if a single value is passed in will return a list.

    n_views: int
        Number of views.

    Output
    ------
    regs: None, list of floats
    """
    if isinstance(regs, (Number, str)):
        regs = [regs] * n_views

    if regs is not None:
        assert len(regs) == n_views, \
            "regs should be None or len(regs) == n_views"

        for b in range(n_views):
            if regs[b] is not None:
                if isinstance(regs[b], Number):
                    regs[b] = float(regs[b])
                    assert (regs[b] >= 0) and (regs[b] <= 1),\
                        "regs should be between 0 and 1, not {}".\
                        format(regs[b])

                elif isinstance(regs[b], str):
                    assert regs[b] in ['oas', 'lw']

    return regs


def flag_mean(bases, n_components=None, weights=None):
    """
    Computes the subspae flag mean (Draper et al, 2014). Given a colletion of orthonormal matrices, X_1, ..., X_B we compute the the low rank SVD of X := [X_1, ..., X_B]. The left singular vectors are the flag mean. We refer to the right singular vectors as the "view loadings". We further refer to the projetion of each view onto its corresponding entries of the view loadings as the view scores.

    Parameters
    ----------
    bases: list
        List of orthonormal basis matrices for each subspace.

    n_components: None, int
        Number of components to compute.

    weights: None, list of ints
        Weights to put on the subspaces

    Output
    ------
    dict with entries: flag_mean, sqsvals, view_scores, view_loadings

    flag_mean: array-like, (ambient_dim, n_components)
        Flag mean orthonormal basis matrix.

    sqsvals: array-like, (n_components, )
        The squared singular values
    """
    Xs, n_views, ambient_dim, subspace_dims = \
        check_Xs(bases, multiview=True, return_dimensions=True)

    # optionally add weights to each subspace
    if weights is not None:
        assert len(weights) == n_views
        for b in range(n_views):
            assert weights[b] > 0
            bases[b] = bases[b] * weights[b]

    # compte SVD of concatenated basis matrix
    flag_mean, svals, view_loadings = svd_wrapper(np.block(bases),
                                                  rank=n_components)

    # get the view loadings and scores
    view_loadings = split(view_loadings, dims=subspace_dims, axis=0)
    view_scores = [bases[b] @ view_loadings[b]
                   for b in range(n_views)]

    flag_mean, view_scores, view_loadings = \
        mcca_det_output(flag_mean, view_scores, view_loadings)

    return {'flag_mean': flag_mean,
            'sqsvals': svals ** 2,
            'view_scores': view_scores,
            'view_loadings': view_loadings}
