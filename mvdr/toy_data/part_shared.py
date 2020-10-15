import numpy as np
from sklearn.utils import check_random_state
from numbers import Number
from scipy.sparse import diags

from mvdr.linalg_utils import rand_orthog
from mvdr.utils import powerset


def sample_part_shared_fact_model(ranks, svals,
                                  n_samples=200, n_features=[10, 20, 30],
                                  noise_std=1.0, m=1.5,
                                  random_state=None):
    """
    Samples from a multi-block factor model with partially shared structures.

    Parameters
    ----------
    ranks:

    svals:

    n_samples: int
        Number of samples.

    dims: list of ints
        Number of features in each data block.

    noise_std: float, list of floats
        The noise standard deviation for each block.
        If a float, uses the same noise STD for each block.

    m: float
        The signal strength.

    random_state: None, int
        Seed.

    Output
    ------
    blocks, info

    block: list of array-like
        The data blocks.

    info: dict
        Other infomation including the true scores/loadings, etc.
    """

    # bureaucracy
    rng = check_random_state(random_state)
    n_blocks = len(n_features)

    # setup noise std
    if isinstance(noise_std, Number):
        noise_std = [noise_std] * n_blocks
    assert len(noise_std) == n_blocks

    # make sure all blocks are low ranks
    K_tot, K_shared, block_ranks, block_indiv_ranks = \
        get_rank_info(n_blocks=n_blocks, ranks=ranks)
    assert n_samples > K_tot
    for b in range(n_blocks):
        assert block_ranks[b] <= n_features[b]

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

    # sample block errors
    errors = [noise_std[b] * rng.normal(size=(n_samples, n_features[b]))
              for b in range(n_blocks)]

    # setup block loadings by sampling an orthonormal matrix
    block_loadings = [{} for b in range(n_blocks)]
    for S in ranks.keys():
        K = ranks[S]
        for b in S:
            block_loadings[b][S] = rand_orthog(n=n_features[b], K=K,
                                               random_state=rng)

    # setup block signal matices
    block_signal_mats = [np.zeros((n_samples, n_features[b]))
                         for b in range(n_blocks)]
    for S in ranks.keys():
        for b in S:

            W = block_loadings[b][S]
            sv = svals[b][S]
            U = signal_scores[S]

            block_signal_mats[b] += U @ diags(sv) @ W.T

    # set data blocks
    blocks = [None for b in range(n_blocks)]
    for b in range(n_blocks):
        blocks[b] = block_signal_mats[b] + errors[b]

    return blocks, {'ranks': ranks,
                    'svals': svals,
                    'signal_scores': signal_scores,
                    'block_signal_mats': block_signal_mats,
                    'block_loadings': block_loadings}


def get_part_shared_struct_ranks(n_blocks=3, rank=2,
                                 min_size=1, max_size=None):
    """
    Sets the signal rank for all possible partially shared structures.

    Parameters
    ----------
    n_blocks: int
        Number of blocks.

    rank: int
        The rank for all partially shared signal ranks.

    min_size: int
        Minimum size for partially shared sets; ignores sets smaller than this.
        For example, if min_size=2 then there will be no block individual signals.

    min_size: int
        Maximum size for partially shared sets; ignores sets larget than this.
        For example, if max_size=1 then there will be only block individual signals.

    Output
    ------
    ranks: dict
        The keys of this dict are frozenset(S) where S in 2^{n_blocks}.
        The value is the rank corresponding to S.
    """
    ranks = {}
    for S in powerset(x=range(n_blocks), min_size=min_size, max_size=max_size):
        ranks[frozenset(S)] = rank
    return ranks


def get_rank_info(n_blocks, ranks):
    """
    Gets rank information from the partially shared ranks.

    Parameters
    ----------
    n_blocks: int
        Number of blocks.

    rank: dict
        The ranks for each partially shared structure.
        The keys of this dict are frozenset(S) where S in 2^{n_blocks}.
        The value is the rank corresponding to S.

    Output
    ------
    K_tot, K_shared, block_ranks, block_indiv_ranks

    K_tot: int
        Total signal rank.

    K_shared: int
        Total signal rank of all shared signals (i.e. excludes block individual signals)

    block_ranks: list of int
        The rank of each block.

    block_indiv_ranks: list of int
        The block individual rank for each block.
    """

    block_ranks = [0 for b in range(n_blocks)]
    block_indiv_ranks = [0 for b in range(n_blocks)]
    K_tot = 0
    K_shared = 0

    for S in ranks.keys():
        r = ranks[S]

        for b in S:
            block_ranks[b] += r

        K_tot += r

        if len(S) == 1:
            block_indiv_ranks[b] = r

        elif len(S) >= 2:
            K_shared += r

    return K_tot, K_shared, block_ranks, block_indiv_ranks

# test cases
# ranks = get_part_shared_struct_ranks(max_size=1)
# get_rank_info(n_blocks=3, ranks=ranks)
# ranks = get_part_shared_struct_ranks(min_size=2)
# get_rank_info(n_blocks=3, ranks=ranks)
# ranks = get_part_shared_struct_ranks()
# get_rank_info(n_blocks=3, ranks=ranks)


def scale_svals(svals, n_samples, n_features, noise_std, m=1.5):

    n_blocks = len(n_features)

    # setup noise std
    if isinstance(noise_std, Number):
        noise_std = [noise_std] * n_blocks
    assert len(noise_std) == n_blocks

    sval_bases = []
    for b in range(n_blocks):

        # from equation (2.15) of (Choi et al, 2017).
        base = m * noise_std[b] * (n_samples * n_features[b]) ** (.25)
        sval_bases.append(base)

    for S in svals.keys():
        for b in S:
            svals[b][S] = svals[b][S] * sval_bases[b]

    return svals


def get_ps_sval_spikes(n_blocks, ranks, random_state=None):
    """
    Gets (random) singular values for partially shared structures.

    Parameters
    ----------
    n_blocks: int
        Number of blocks.

    rank: dict
        The ranks for each partially shared structure.
        The keys of this dict are frozenset(S) where S in 2^{n_blocks}.
        The value is the rank corresponding to S.

    random_state: None, int
        Random state

    Output
    ------
    svals: list of dicts

    """
    # TODO: may want to think a bit more about how we want to do this

    rng = check_random_state(random_state)

    K_tot = get_rank_info(n_blocks=n_blocks, ranks=ranks)[0]
    # tot = 0
    # for S in ranks.keys():
    #     tot += len(S) * ranks[S]
    low = 1
    high = 1 + K_tot

    svals = [{} for b in range(n_blocks)]
    for S in ranks.keys():
        K = ranks[S]
        for b in S:
            svals[b][S] = rng.uniform(low=low, high=high, size=K)

    return svals


def get_scaled_ps_sval_spikes(ranks, n_samples, n_features, noise_std, m=1.5,
                              random_state=None):

    svals = get_ps_sval_spikes(n_blocks=len(n_features), ranks=ranks,
                               random_state=random_state)

    return scale_svals(svals=svals, n_samples=n_samples,
                       n_features=n_features, noise_std=noise_std, m=m)
