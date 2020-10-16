import numpy as np
from scipy.sparse import issparse
from copy import deepcopy
import warnings
import numbers
from sklearn.linear_model import LinearRegression
from textwrap import dedent

from mvlearn.utils import check_Xs

from mvdr.linalg_utils import svd_wrapper
from mvdr.mcca.view_processing import center_views, \
    split
from mvdr.ajive.wedin_bound import get_wedin_samples
from mvdr.ajive.random_direction import sample_randdir


def ajive(Xs, init_signal_ranks, joint_rank=None, indiv_ranks=None,
          center=True, common_loading_method='map_back',
          check_joint_identif=True,
          wedin_percentile=5, n_wedin_samples=1000, wedin_seed=None,
          rand_percentile=95, n_rand_samples=1000, rand_seed=None,
          final_decomp=True,
          store_full=True, n_jobs=None):

    Xs, n_views, n_samples, n_features = check_Xs(Xs, multiview=True,
                                                  return_dimensions=True)

    Xs, init_signal_ranks, indiv_ranks = \
        arg_checker(Xs, init_signal_ranks, joint_rank,
                    indiv_ranks, common_loading_method,
                    n_wedin_samples, n_rand_samples)

    Xs, centerers = center_views(Xs, center=center)

    ################################################################
    # step 1: initial signal space extraction by SVD on each view #
    ################################################################

    init_signal_svd = [None for _ in range(n_views)]
    sv_thresholds = [None for _ in range(n_views)]
    for b in range(n_views):

        # signal rank + 1 to get individual rank sv threshold
        U, D, V = svd_wrapper(Xs[b], init_signal_ranks[b] + 1)

        # The SV threshold is halfway between the init_signal_ranks[bn]th
        # and init_signal_ranks[bn] + 1 st singular value. Recall that
        # python is zero indexed.
        t = (D[init_signal_ranks[b] - 1] + D[init_signal_ranks[b]]) / 2

        # store data for later
        sv_thresholds[b] = t
        init_signal_svd[b] = {'scores': U[:, 0:init_signal_ranks[b]],
                              'svals': D[0:init_signal_ranks[b]],
                              'loadings': V[:, 0:init_signal_ranks[b]]}

    ##################################
    # step 2: joint space estimation #
    ##################################
    # this step estimates the joint rank and computes the common
    # joint space basis (the flag mean)

    M = np.bmat([init_signal_svd[b]['scores'] for b in range(n_views)])
    rank = min(n_samples, *init_signal_ranks)
    common_scores, common_svals, common_loadings = svd_wrapper(M, rank=rank)

    if joint_rank is None:  # estimate the joint rank
        joint_rank, rank_est_data = \
            estimate_joint_rank(Xs=Xs,
                                init_signal_svd=init_signal_svd,
                                common_svals=common_svals,
                                n_rand_samples=n_rand_samples,
                                n_wedin_samples=n_wedin_samples,
                                rand_seed=rand_seed,
                                wedin_seed=wedin_seed,
                                n_jobs=n_jobs)
    else:
        rank_est_data = {}

    rank_est_data['all_common_svals'] = deepcopy(common_svals)
    rank_est_data['identif_dropped'] = None

    if joint_rank == 0:
        common_scores = None
        common_svals = None
        common_loadings = None

        view_loadings = [None for b in range(n_views)]
        view_scores = [None for b in range(n_views)]
    else:
        # truncated common normalized scores
        common_scores = common_scores[:, 0:joint_rank]
        common_svals = common_svals[0:joint_rank]
        common_loadings = common_loadings[:, 0:joint_rank]

        # check identifiability constraint
        if check_joint_identif:
            common_scores, common_svals, common_loadings, \
                joint_rank, dropped = \
                check_identifiability(Xs, sv_thresholds,
                                      common_scores, common_svals,
                                      common_loadings)

            rank_est_data['identif_dropped'] = dropped

        ############################################
        # compute view common loadings and scores #
        ############################################
        common_loadings = split(common_loadings,
                                dims=init_signal_ranks, axis=0)
        view_loadings = [None for b in range(n_views)]
        if common_loading_method == 'map_back':

            # compute V_b L_b where
            # V_b are PCA loadings from the initial SVD and
            # L_b are the view_common_loadings
            # this is equivalent to pricipal components regression.
            for b in range(n_views):

                # V D^{-1} W
                view_loadings[b] = \
                    np.multiply(init_signal_svd[b]['loadings'],
                                1.0 / init_signal_svd[b]['svals']).\
                    dot(common_loadings[b])

        else:
            for b in range(n_views):
                view_loadings[b] = np.zeros((n_features[b], joint_rank))
                for r in range(joint_rank):

                    if view_loadings == 'regress':
                        # regress the common normalized scores
                        # onto each data view
                        get_coef = LinRefCoef()
                    else:
                        get_coef = view_loadings

                    view_loadings[b][:, r] = \
                        get_coef(X=Xs[b], y=common_scores[r])

        view_scores = [Xs[b].dot(view_loadings[b])
                       for b in range(n_views)]

    #######################################
    # step 3: compute final decomposition #
    #######################################
    # this step computes the view specific estimates

    if final_decomp:
        decomps = view_decomposition(Xs, common_scores, sv_thresholds,
                                     indiv_ranks=indiv_ranks,
                                     store_full=store_full)
    else:
        decomps = None

    if common_svals is not None:
        sqsvals = common_svals ** 2
    else:
        sqsvals = None

    return {'common': {'view_loadings': view_loadings,
                       'view_scores': view_scores,
                       'common_loadings': common_loadings,
                       'sqsvals': sqsvals,
                       'common_scores': common_scores,
                       'rank': joint_rank},

            'decomps': decomps,

            'rank_est': rank_est_data,

            'sv_thresholds': sv_thresholds,

            'init_signal_svd': init_signal_svd,

            'centerers': centerers,

            'center': center,
            'init_signal_ranks': init_signal_ranks
            }


_ajive_docs = dict(Xs=dedent("""
    Xs : list of array-likes or numpy.ndarray
        The list of data matrices each shaped (n_samples, n_features_b).
    """), basic_args=dedent("""
    init_signal_ranks: list of ints
        The initial signal ranks.
        These must be at most n_features_b - 1.

    center: bool or list
        Whether or not to center data views.
        Can pass in either one bool that applies to all views or a list of bools to specify different options for each view.

    common_loading_method: str
        How the common loadings are obtaind; must be one of
        ['map_back', 'regress'].

    check_joint_identif: bool
        Whether or not to check the joint identifiability condition.

    wedin_percentile: int (default=5)
        Percentile for wedin (lower) bound cutoff for squared singular values
        used to estimate joint rank.

    n_wedin_samples: int, None
        Number of wedin bound samples to draw. If None, will not use the
        wedin bound.

    randdir_percentile: int (default=95)
        Percentile for random direction (lower) bound cutoff for squared
        singular values used to estimate joint rank..

    n_randdir_samples: int, None
        Number of random directions samples to draw. If None, will
        not use the random direction bound.

    usr_joint_rank: {None, int}
        User supplied joint rank; if None, will estimate the joint rank.

    usr_iniv_ranks: {list, dict, None}
        User supplied individual ranks; if None, will estimate the
        individual ranks.

    store_full: bool
        Whether or not to store the full J, I, E matrices.
        These can be memory intensive if the data matrices are large.

    final_decomp: bool
        Whether or not to compure the final AJIVE decomposition after
        estimating the joint space.

    n_jobs: int, None
        Number of jobs for parallel processing wedin samples and random
        direction samples using sklearn.externals.joblib.Parallel.
        If None, will not use parallel processing.
        """))

ajive.__doc__ = dedent("""
    Angle-based joint and individual variation explained.

    Parameters
    ----------
    {Xs}

    {basic_args}

    Output
    ------
    dict
    """.format(**_ajive_docs))


def arg_checker(Xs, init_signal_ranks, joint_rank, indiv_ranks,
                common_loading_method,
                n_wedin_samples, n_rand_samples):
    """
    Checks arguments for AJIVE().fit()
    """
    ##########
    # views #
    ##########

    Xs, n_views, n_samples, n_features = check_Xs(Xs, multiview=True,
                                                  return_dimensions=True)

    if not n_views >= 2:
        raise ValueError('At least two views must be provided.')

    # check views have the same number of observations
    if len(set(Xs[b].shape[0] for b in range(n_views))) != 1:
        raise ValueError('Blocks must have same number'
                         ' of observations (rows).')

    # format views
    # make sure views are either csr or np.array
    for b in range(n_views):
        if issparse(Xs[b]):  # TODO: allow for general linear operators
            raise NotImplementedError
        else:
            Xs[b] = np.array(Xs[b])

    #####################
    # init_signal_ranks #
    #####################

    if len(init_signal_ranks) != n_views:
        raise ValueError('Each view must have an initial signal rank.')

    # initial signal rank must be at least one lower than the shape of the view
    for b in range(n_views):
        r = init_signal_ranks[b]
        if not (1 <= r) or not (r <= min(n_samples, n_features[b]) - 1):
            raise ValueError('initial signal rank for view {} must be '
                             'between 1 and min(n,d_b) - 1.'.format(b))

    ##############
    # joint_rank #
    ##############
    if joint_rank is not None and joint_rank > sum(init_signal_ranks):
        raise ValueError('joint_rank must be smaller than the sum'
                         'of the initial signal ranks')
    if joint_rank is not None:
        if sum([n_wedin_samples is None, n_rand_samples is None]) == 2:
            raise ValueError('If joint_rank is not provided, at least one'
                             'of the random direction and wedin bounds'
                             ' must be in play.')

    ###############
    # indiv_ranks #
    ###############
    if indiv_ranks is None:
        indiv_ranks = [None for b in range(n_views)]

    if len(indiv_ranks) != n_views:
        raise ValueError('If individual signal ranks are provided,'
                         ' they must be provided for each view.')

    for b in range(n_views):
        r = indiv_ranks[b]
        if not (r is None or isinstance(r, numbers.Number)):
            raise ValueError('Individual signal rank provided for view {} '
                             'must be either None or a number'.format(b))

    ##########################
    # common loadings method #
    ##########################

    if not callable(common_loading_method):
        if common_loading_method not in ['map_back', 'regress']:
            raise ValueError("common_loading_method must be one of"
                             "['map_back', 'regress'], not {}".
                             format(common_loading_method))

        if common_loading_method == 'regress':
            if max(n_features) >= n_samples:
                raise ValueError('common_loading_method = regress is only'
                                 'available in low dimensional setting i.e.'
                                 'the largest view dimension must be smaller'
                                 ' than the number of samples')

    return Xs, init_signal_ranks, indiv_ranks


def estimate_joint_rank(Xs, init_signal_svd, common_svals,
                        n_rand_samples=1000, n_wedin_samples=1000,
                        rand_percentile=95, wedin_percentile=5,
                        rand_seed=None, wedin_seed=None,
                        n_jobs=-1):
    """
    Estimates the joint rank of a collection of data views using the
    random direction bound and wedin bound.

    Parameters
    ----------
    Xs: list of array-like
        The original data matrices. These are only needed for the wedin bound.

    init_signal_svd: dict
        The initial SVD/PCA of each data view.

    common_svals: list of floats
        The singular values from the SVD of the concatonated signal basis matrix. Note these are inversely releated to the principal angles.

    n_rand_samples: int, None

    n_wedin_samples: int, None

    rand_percentile: int

    wedin_percentile: int

    n_jobs: -1, int, None

    Output
    ------
    TODO: document

    """
    # At least one of the bounds has to be in play
    assert sum([n_wedin_samples is None, n_rand_samples is None]) != 2

    n_views = len(Xs)
    n_obs = Xs[0].shape[0]

    # dimensions of the initial signal subspaces
    score_dims = [init_signal_svd[b]['scores'].shape[1]
                  for b in range(n_views)]

    #############################################
    # estimate random direction bound threshold #
    #############################################

    if n_rand_samples is not None:
        rand_sv_samples = sample_randdir(n=n_obs, dims=score_dims,
                                         n_samples=n_rand_samples,
                                         random_state=rand_seed,
                                         n_jobs=n_jobs)

        rand_threshold = np.percentile(rand_sv_samples, rand_percentile)

    else:
        rand_sv_samples = None
        rand_threshold = None

    ##################################
    # estimate wedin bound threshold #
    ##################################
    view_wedin_samples = [None for _ in range(n_views)]
    if n_wedin_samples is not None:

        for b in range(n_views):
            view_wedin_samples[b] = \
                get_wedin_samples(X=Xs[b],
                                  U=init_signal_svd[b]['scores'],
                                  D=init_signal_svd[b]['svals'],
                                  V=init_signal_svd[b]['loadings'],
                                  rank=score_dims[b],
                                  n_samples=n_wedin_samples,
                                  random_state=wedin_seed,
                                  n_jobs=n_jobs)

        wedin_sv_samples = len(Xs) - \
            np.array([sum(view_wedin_samples[b][r] ** 2
                      for b in range(n_views))
                     for r in range(n_wedin_samples)])
        wedin_threshold = np.percentile(wedin_sv_samples, wedin_percentile)

    else:
        wedin_sv_samples = None
        wedin_threshold = None

    #######################
    # estimate joint rank #
    #######################
    if rand_threshold is None:
        svalsq_threshold = wedin_threshold
    elif wedin_threshold is None:
        svalsq_threshold = rand_threshold
    else:
        svalsq_threshold = max(rand_threshold, wedin_threshold)

    joint_rank = sum(common_svals ** 2 > svalsq_threshold)

    return joint_rank, {'svalsq_threshold': svalsq_threshold,

                        'rand': {'samples': rand_sv_samples,
                                 'threshold': rand_threshold},

                        'wedin': {'view_samples': view_wedin_samples,
                                  'samples': wedin_sv_samples,
                                  'threshold': wedin_threshold}}


def check_identifiability(Xs, sv_thresholds,
                          common_scores, common_svals, common_loadings):
    """
    Checks the identifiability constraint on the joint singular values

    See page 15 on https://arxiv.org/pdf/1704.02060.pdf  (4th paragraph from the bottom)
    TODO: document
    """
    n_views = len(Xs)
    joint_rank = common_scores.shape[1]

    dropped = []
    # check identifiability constraint
    to_keep = set(range(joint_rank))
    for j in range(joint_rank):
        for b in range(n_views):

            score = Xs[b].T.dot(common_scores[:, j])
            sv = np.linalg.norm(score)

            # if sv is below the threshold for any data view remove j
            if sv < sv_thresholds[b]:
                warnings.warn('removing flag mean component ' + str(j))
                to_keep.remove(j)
                dropped.append(j)
                break

    # remove columns of joint_scores that don't satisfy the constraint
    joint_rank = len(to_keep)
    common_scores = common_scores[:, list(to_keep)]
    common_svals = common_svals[list(to_keep)]
    common_loadings = common_loadings[:, list(to_keep)]

    return common_scores, common_svals, common_loadings, joint_rank, dropped


def view_decomposition(views, common_scores, sv_thresholds=None,
                       indiv_ranks=None,
                       store_full=True,
                       joint_decomp=True, indiv_decomp=True):
    """
    Computes the AJIVE view decomposition and view PCAs for each data view.

    Parameters
    ----------
    views: list of array-like
        The original data views.

    common_scores: array-like, shape (n_obs, joint_rank)
        Estimate of common normalized scores.

    sv_thresholds: float
        Thresholds for singular vaules.

    indiv_ranks: None, list of int/None
        The user may specify the rank of each view's individual rank.
        If None, individual rank will be estimated.

    store_full: bool
        Whether or not to store the full matrices.

    joint_decomp: bool
        Whether or not to compute the joint decomposition.

    indiv_decomp: bool
        Whether or not to compute the individual decompositions.

    Output
    ------
    decomps: list of dicts

        decomps[b] has keys ['joint', 'indiv', 'noise']
        decomps[b]['joint'] contains the estimated joint matrix, J_b,
        as well as the PCA of J_b and has keys
        ['full', 'scores', 'loadings', 'svals', 'rank']

        Similarly for decomps[b]['indiv']
        decomps[b]['noise'] only has key 'full'
    """

    n_views = len(views)
    if common_scores is not None:
        joint_rank = common_scores.shape[1]
    else:
        joint_rank = 0

    if indiv_ranks is None:
        indiv_ranks = [None for _ in range(n_views)]

    if joint_rank == 0:
        joint_decomp = False

    decomps = [{} for b in range(n_views)]
    for b in range(n_views):
        X = views[b]

        ########################################
        # step 3.1: view specific joint space #
        ########################################
        # project X onto the joint space then compute SVD
        if joint_decomp:

            J = np.array(np.dot(common_scores, common_scores.T.dot(X)))
            U, D, V = svd_wrapper(J, joint_rank)
            if not store_full:
                J = None  # kill J matrix to save memory

        else:
            J, U, D, V = None, None, None, None
            # if store_full:
            #     J = np.zeros(shape=views[bn].shape)
            # else:
            #     J = None

        decomps[b]['joint'] = {'full': J,
                               'scores': U,
                               'svals': D,
                               'loadings': V,
                               'rank': joint_rank}

        #############################################
        # step 3.2: view specific individual space #
        #############################################
        # project X onto the orthogonal complement of the joint space,
        # estimate the individual rank, then compute SVD

        if indiv_decomp:

            # project X columns onto orthogonal complement of joint_scores
            if joint_rank == 0:
                X_orthog = X
            else:
                X_orthog = X - common_scores.dot(common_scores.T.dot(X))

            # estimate individual rank using sv threshold, then compute SVD
            if indiv_ranks[b] is None:
                max_rank = min(X.shape) - joint_rank  # saves computation
                U, D, V = svd_wrapper(X_orthog, max_rank)
                rank = sum(D > sv_thresholds[b])

                if rank == 0:
                    U, D, V = None, None, None
                else:
                    U = U[:, 0:rank]
                    D = D[0:rank]
                    V = V[:, 0:rank]

                indiv_ranks[b] = rank

            else:  # indiv_rank has been provided by the user
                rank = indiv_ranks[b]
                if rank == 0:
                    U, D, V = None, None, None
                else:
                    U, D, V = svd_wrapper(X_orthog, rank)

            if store_full:
                if rank == 0:
                    I = np.zeros(shape=views[b].shape)
                else:
                    I = np.array(U.dot(np.diag(D).dot(V.T)))
            else:
                I = None  # Kill I matrix to save memory
        else:
            I, U, D, V, rank = None, None, None, None, None

        decomps[b]['individual'] = {'full': I,
                                    'scores': U,
                                    'svals': D,
                                    'loadings': V,
                                    'rank': rank}

        ###################################
        # step 3.3: estimate noise matrix #
        ###################################

        if store_full and not issparse(X) and indiv_decomp and joint_decomp:
            E = X - (J + I)
        else:
            E = None

        decomps[b]['noise'] = E

    return decomps


class LinRefCoef:
    """
    LinRefCoef() is a callable object. We make it an object so that it can
    be pickled (functions don't play nice with pickling in python.)
    """

    def __call__(X, y):
        """
        Returns the linear regression coefficinet
        """
        lm = LinearRegression(fit_intercept=False)
        lm.fit(X=X, y=y)
        return lm.coef_
