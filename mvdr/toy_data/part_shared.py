import numpy as np
from sklearn.utils import check_random_state
from numbers import Number
from scipy.sparse import diags

from mvdr.linalg_utils import rand_orthog
from mvdr.utils import powerset


def sample_part_shared_fact_model(ranks, svals,
                                  n_samples=200, n_features=[10, 20, 30],
                                  noise_std=1.0,
                                  random_state=None):
    """
    Samples from a multi-view factor model with partially shared structures.

    Parameters
    ----------
    ranks:

    svals:

    n_samples: int
        Number of samples.

    dims: list of ints
        Number of features in each data view.

    noise_std: float, list of floats
        The noise standard deviation for each view.
        If a float, uses the same noise STD for each view.

    random_state: None, int
        Seed.

    Output
    ------
    views, info

    views: list of array-like
        The data views.

    info: dict
        Other infomation including the true scores/loadings, etc.
    """

    # bureaucracy
    rng = check_random_state(random_state)
    n_views = len(n_features)

    # setup noise std
    if isinstance(noise_std, Number):
        noise_std = [noise_std] * n_views
    assert len(noise_std) == n_views

    # make sure all views are low ranks
    K_tot, K_shared, view_ranks, view_indiv_ranks = \
        get_rank_info(n_views=n_views, ranks=ranks)
    assert n_samples > K_tot
    for b in range(n_views):
        assert view_ranks[b] <= n_features[b]

    # setup scores
    scores = rng.normal(size=(n_samples, K_tot), scale=1)
    scores = np.linalg.qr(scores)[0]
    # TODO: give option for U to be iid N(0,1). Need to figure out scaling in this case
    signal_scores = {}
    L = 0
    R = 0
    for S in ranks.keys():
        R += ranks[S]
        signal_scores[S] = scores[:, L:R]
        L += ranks[S]

    # sample view errors
    errors = [noise_std[b] * rng.normal(size=(n_samples, n_features[b]))
              for b in range(n_views)]

    # setup view loadings by sampling an orthonormal matrix
    view_loadings = [{} for b in range(n_views)]
    for S in ranks.keys():
        K = ranks[S]
        for b in S:
            view_loadings[b][S] = rand_orthog(n=n_features[b], K=K,
                                              random_state=rng)

    # setup view signal matices
    view_signal_mats = [np.zeros((n_samples, n_features[b]))
                        for b in range(n_views)]
    for S in ranks.keys():
        for b in S:

            W = view_loadings[b][S]
            sv = svals[b][S]
            U = signal_scores[S]

            view_signal_mats[b] += U @ diags(sv) @ W.T

    # set data views
    views = [None for b in range(n_views)]
    for b in range(n_views):
        views[b] = view_signal_mats[b] + errors[b]

    return views, {'ranks': ranks,
                   'svals': svals,
                   'signal_scores': signal_scores,
                   'view_signal_mats': view_signal_mats,
                   'view_loadings': view_loadings}


def get_part_shared_struct_ranks(n_views=3, rank=2,
                                 min_size=1, max_size=None):
    """
    Sets the signal rank for all possible partially shared structures.

    Parameters
    ----------
    n_views: int
        Number of views.

    rank: int
        The rank for all partially shared signal ranks.

    min_size: int
        Minimum size for partially shared sets; ignores sets smaller than this.
        For example, if min_size=2 then there will be no view individual signals.

    min_size: int
        Maximum size for partially shared sets; ignores sets larget than this.
        For example, if max_size=1 then there will be only view individual signals.

    Output
    ------
    ranks: dict
        The keys of this dict are frozenset(S) where S in 2^{n_views}.
        The value is the rank corresponding to S.
    """
    ranks = {}
    for S in powerset(x=range(n_views), min_size=min_size, max_size=max_size):
        ranks[frozenset(S)] = rank
    return ranks


def get_rank_info(n_views, ranks):
    """
    Gets rank information from the partially shared ranks.

    Parameters
    ----------
    n_views: int
        Number of views.

    rank: dict
        The ranks for each partially shared structure.
        The keys of this dict are frozenset(S) where S in 2^{n_views}.
        The value is the rank corresponding to S.

    Output
    ------
    rank_info: dict with keys: ['tot', 'shared', 'view', 'view_indiv']

    rank_info['tot']: int
        Total signal rank i.e. the PCA rank of the concatenated data.

    rank_info['shared']: int
        Total signal rank of all shared signals meaning signals the correspond to at leasat two views (i.e. excludes view individual signals).

    view: list of int
        The total PCA rank of each view marinally.

    view_indiv: list of int
        The view individual rank for each view.
    """

    view_ranks = [0 for b in range(n_views)]
    view_indiv_ranks = [0 for b in range(n_views)]
    K_tot = 0
    K_shared = 0

    for S in ranks.keys():
        r = ranks[S]

        for b in S:
            view_ranks[b] += r

        K_tot += r

        if len(S) == 1:
            view_indiv_ranks[b] = r

        elif len(S) >= 2:
            K_shared += r

    # return K_tot, K_shared, view_ranks, view_indiv_ranks
    return {'tot': K_tot,
            'shared': K_shared,
            'view': view_ranks,
            'view_indiv': view_indiv_ranks}


def scale_svals(svals, n_samples, n_features, noise_std, m=1.5):
    """
    Scales singular values according to the formula in (Choi et al. 2017)

    Parameters
    ----------
    svals: list of dicts
        svals[b] is a dict whose keys are all the partially shared structures containing b

    n_samples: int
        Number of samples.

    n_features: list of ints
        Number of features in each view.

    noise_std: list of floats
        Noise standard deviation for each view.

    m: float
        Signal strength.
    """

    n_views = len(n_features)
    assert len(svals) == n_views

    # setup noise std
    if isinstance(noise_std, Number):
        noise_std = [noise_std] * n_views
    assert len(noise_std) == n_views

    sval_bases = []
    for b in range(n_views):

        # from equation (2.15) of (Choi et al, 2017).
        base = m * noise_std[b] * (n_samples * n_features[b]) ** (.25)
        sval_bases.append(base)

    for b in range(n_views):
        for S in svals[b].keys():
            svals[b][S] = svals[b][S] * sval_bases[b]

    return svals


def get_ps_sval_spikes(n_views, ranks, random_state=None):
    """
    Gets (random) singular value spikes for partially shared structures.

    Parameters
    ----------
    n_views: int
        Number of views.

    ranks: dict
        The ranks for each partially shared structure.
        The keys of this dict are frozenset(S) where S in 2^{n_views}.
        The value is the rank corresponding to S.

    random_state: None, int
        Random state

    Output
    ------
    svals: list of dicts

    """
    # TODO: may want to think a bit more about how we want to do this

    rng = check_random_state(random_state)

    rank_info = get_rank_info(n_views=n_views, ranks=ranks)
    # tot = 0
    # for S in ranks.keys():
    #     tot += len(S) * ranks[S]
    low = 1
    # high = 1 + K_tot

    svals = [{} for b in range(n_views)]
    for S in ranks.keys():
        K = ranks[S]
        for b in S:
            high = 1 + rank_info['view'][b]

            svals[b][S] = rng.uniform(low=low, high=high, size=K)

    return svals


def get_scaled_ps_sval_spikes(ranks, n_samples, n_features,
                              noise_std, m=1.5,
                              random_state=None):
    """
    Gets the scaled singular value spikes for the parially shared signals.
    Parameters
    ----------
    ranks: dict
        The ranks for each partially shared structure.
        The keys of this dict are frozenset(S) where S in 2^{n_views}.
        The value is the rank corresponding to S.

    n_samples: int
        Number of samples.

    n_features: list of ints
        Number of features in each view.

    noise_std: list of floats
        Noise standard deviation for each view.

    m: float
        Signal strength.

    random_state: None, int
        Random state

    """

    svals = get_ps_sval_spikes(n_views=len(n_features), ranks=ranks,
                               random_state=random_state)

    return scale_svals(svals=svals, n_samples=n_samples,
                       n_features=n_features, noise_std=noise_std, m=m)
