"""
Other methods kernel MCCA. These are not yet tested; they may work or may contain silly mistakes.
"""
import numpy as np
from scipy.linalg import block_diag
from itertools import combinations
from warnings import warn
from textwrap import dedent


from mvdr.mcca.block_processing import center_kernel_blocks, split,\
    get_blocks_metadata, initial_svds
from mvdr.mcca.mcca import check_regs, mcca_det_output, flag_mean, mcca_gevp, \
    mcca_basic_params
from mvdr.linalg_utils import eigh_wrapper, normalize_cols
from mvdr.mcca.k_mcca import kmcca_params


def k_mcca_gevp(Ks, n_components=None, center=True, regs=None, diag_mode='A'):
    n_blocks, n_samples = len(Ks), Ks[0].shape[0]

    regs = check_regs(regs=regs, n_blocks=n_blocks)
    # if regs is None or any(r is None for r in regs) \
    #         or any(r == 0 for r in regs):
    #     warn("Kernel MCCA is usually best if some regularization is used.")

    if n_components is None:
        # TODO: figure out better default
        n_components = 2

    #########################################
    # solve generalized eigenvector problem #
    #########################################

    # center data blocks
    Ks, centerers = center_kernel_blocks(Ks, center=center)

    LHS, RHS = get_k_mcca_gevp_data(Ks, regs=regs, diag_mode=diag_mode)
    gevals, dual_vars = eigh_wrapper(A=LHS, B=RHS, rank=n_components)

    #################
    # Format output #
    #################
    block_scores = split(dual_vars, dims=[n_samples] * n_blocks, axis=0)
    common_scores = sum(bs for bs in block_scores)

    # Ensure common scores are unit vectors
    # TODO: is this the behavior we want when regularization is used?
    common_norm_scores, col_norms = normalize_cols(common_scores)

    # enforce deterministic output due to possible sign flips
    common_norm_scores, block_scores, _ = \
        mcca_det_output(common_norm_scores, block_scores)

    return {'block_scores': block_scores,
            'common_norm_scores': common_norm_scores,
            'cs_col_norms': col_norms,
            'evals': gevals,
            'centerers': centerers}


k_mcca_gevp.__doc__ = dedent("""
    Computes kernel MCCA using the generalized eigenvector formulation.
    This will fail in diag_mode 'A' when the kernel matrices are singular.

    Parameters
    ----------
    {}

    {}

    {}

    Output
    ------
    {}

    {}
    """).format(kmcca_params['Ks'],
                mcca_basic_params['basic'],
                kmcca_params['diag_mode'],
                mcca_basic_params['score_out'],
                kmcca_params['centerers'])


def ik_mcca(Ks, signal_ranks=None, sval_thresh=1e-3,
            n_components=None, center=True, regs=None, precomp_svds=None,
            method='auto'):
    n_blocks, n_samples = len(Ks), Ks[0].shape[0]

    if method == 'auto':
        if regs is not None:
            method = 'gevp'
        else:
            # TODO: decide which method to use more intelligently based
            # on the shape of the input matrices
            method = 'svd'

    if method == 'svd':
        # SVD won't method does not work with regularization
        assert regs is None

    if sval_thresh is not None:
        # put sval_thresh on the scale of (1/n) K.
        # since we compute SVD of K, put _sval_thresh on scale of svals of K
        _sval_thresh = sval_thresh * n_samples
    else:
        _sval_thresh = None

    # center data blocks
    Ks, centerers = center_kernel_blocks(Ks, center=center)

    #######################
    # Compute initial SVD #
    #######################

    use_norm_scores = regs is None
    reduced, init_svds, _ = \
        initial_svds(Xs=Ks, signal_ranks=signal_ranks,
                     normalized_scores=use_norm_scores,
                     center=False,
                     precomp_svds=precomp_svds,
                     sval_thresh=_sval_thresh)

    # set n_components
    n_features_reduced = get_blocks_metadata(reduced)[2]
    if n_components is None:
        n_components = sum(n_features_reduced)
    assert n_components <= sum(n_features_reduced)
    if n_components > sum(n_features_reduced):
        warn("Requested too many components!")
    n_components = min(n_components, sum(n_features_reduced))

    ################################
    # Compute MCCA on reduced data #
    ################################
    if method == 'svd':

        # left singluar vectors for each block
        bases = [reduced[b] for b in range(n_blocks)]
        results = flag_mean(bases, n_components=n_components)

        # rename flag mean output to correspond to MCCA conventions
        results['common_norm_scores'] = results.pop('flag_mean')
        results['evals'] = results.pop('sqsvals')
        results['cs_col_norms'] = np.sqrt(results['evals'])

    elif method == 'gevp':

        # compute MCCA gevp problem on reduced data
        results = mcca_gevp(Xs=reduced, n_components=n_components,
                            center=False,  # we already took care of this!
                            regs=regs)

    results.pop('block_loadings')  # Dont need these

    results['init_svds'] = init_svds
    results['centerers'] = centerers

    return results


ik_mcca.__doc__ = dedent("""

    Computes informative kernel MCCA i.e. first computes kernel PCA then runs MCCA. This always uses diag_mode A.

    Parameters
    ----------
    {}

    {}

    signal_ranks: None, int, list
        SVD ranks to compute for each block.

    sval_thresh: float
        For each block we throw out singular values of (1/n)K that are too small (i.e. zero or essentially zero). Setting this value to be non-zero is how we deal with the singular block gram matrices.


    precomp_svds: None, list of tuples
        Precomputed SVDs of each blocks kernel matrix.

    Output
    ------
    {}

    {}

    """).format(kmcca_params['Ks'],
                mcca_basic_params['basic'],
                mcca_basic_params['score_out'],
                kmcca_params['centerers'])


def get_k_mcca_gevp_data(Ks, regs=None, diag_mode='A', ret_lists=False):

    n_blocks, n_samples = len(Ks), Ks[0].shape[0]
    regs = check_regs(regs=regs, n_blocks=n_blocks)

    LHS = [[None for b in range(n_blocks)]
           for b in range(n_blocks)]

    RHS = [None for b in range(n_blocks)]

    # cross covariance matrices
    for (a, b) in combinations(range(n_blocks), 2):
        LHS[a][b] = Ks[a] @ Ks[b]
        LHS[b][a] = LHS[a][b].T

    # block covariance matrices, possibly regularized
    for b in range(n_blocks):
        if regs is None or regs[b] is None or regs[b] == 0:
            RHS[b] = Ks[b] @ Ks[b]

        elif diag_mode == 'A':
            RHS[b] = (1 - regs[b]) * Ks[b] @ Ks[b] + \
                regs[b] * Ks[b]

        elif diag_mode == 'B':
            r = regs[b]

            if np.isclose(r, 1):
                RHS[b] = np.eye(n_samples)
            else:
                kappa = r / (1 - r)
                kale = Ks[b] + 0.5 * n_samples * kappa * np.eye(n_samples)
                RHS[b] = (1 - r) * kale @ kale

        elif diag_mode == 'C':
            RHS[b] = (1 - regs[b]) * Ks[b] @ Ks[b] + \
                regs[b] * np.eye(n_samples)

        LHS[b][b] = RHS[b]

    if not ret_lists:
        LHS = np.block(LHS)
        RHS = block_diag(*RHS)

    return LHS, RHS


get_k_mcca_gevp_data.__doc__ = dedent("""
    Constructs the matrices for the kernel MCCA generalized eigenvector problem. Allows for different options kernelizing the
    block covariance matrices.

    Parameters
    ----------
    {}

    regs: None, float, list of floats
        Regularization to use for each block.

    {}

    ret_lists: bool
        Return returns the blocks of the block matrices instead of the two matrices.

    Output
    ------
    LHS, RHS: array-like, (B * n_samples, B * n_samples)

    Left and right hand sides of GEVP

    LHS v = lambda RHS v
    """).format(kmcca_params['Ks'], kmcca_params['diag_mode'])
