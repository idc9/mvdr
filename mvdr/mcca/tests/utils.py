import numpy as np
from sklearn.utils import check_random_state

from mvlearn.utils import check_Xs

from mvdr.linalg_utils import normalize_cols
from mvdr.mcca.mcca import check_regs, get_mcca_gevp_data


def generate_mcca_test_data():
    rng = check_random_state(849)

    n_samples = 100
    yield [rng.normal(size=(n_samples, 10)), rng.normal(size=(n_samples, 20))]

    yield [rng.normal(size=(n_samples, 10)), rng.normal(size=(n_samples, 20)),
           rng.normal(size=(n_samples, 30))]


def generate_mcca_test_settings():

    yield {'n_components': None, 'regs': None, 'center': True}
    yield {'n_components': 5, 'regs': None, 'center': True}

    yield {'n_components': None, 'regs': .1, 'center': True}
    yield {'n_components': 5, 'regs': .1, 'center': True}

    yield {'n_components': None, 'regs': 1, 'center': True}
    yield {'n_components': 5, 'regs': 1, 'center': True}

    yield {'n_components': None, 'regs': None, 'center': False}
    yield {'n_components': 5, 'regs': None, 'center': False}


def check_mcca_scores_and_loadings(Xs, out,
                                   # common_norm_scores,
                                   # block_scores, block_loadings,
                                   regs=None,
                                   check_normalization=False):
    """
    Checks the scores and loadings output for regularized mcca.

    - block scores are projections of blocks onto loadings
    - common noramlized scores are column normalized version of sum of scores

    - (optional) check normalization of loadings; this should be done for MCCA, but not for informative MCCA.
    """

    block_loadings = out['block_loadings']
    block_scores = out['block_scores']
    common_norm_scores = out['common_norm_scores']
    centerers = out['centerers']

    Xs, n_blocks, n_samples, n_features = check_Xs(Xs, multiview=True,
                                                   return_dimensions=True)


    # make sure to apply centering transformations
    Xs = [centerers[b].transform(Xs[b]) for b in range(n_blocks)]

    for b in range(n_blocks):

        # check block scores are projections of blocks onto block loadings
        assert np.allclose(Xs[b] @ block_loadings[b], block_scores[b])

    # check common norm scores are the column normalized sum of the
    # block scores
    cns_pred = normalize_cols(sum(bs for bs in block_scores))[0]
    assert np.allclose(cns_pred, common_norm_scores)

    if check_normalization:

        # concatenated loadings are orthonormal in the inner produce
        # induced by the RHS of the GEVP
        W = np.vstack(block_loadings)
        RHS = get_mcca_gevp_data(Xs, regs=regs)[1]
        assert np.allclose(W.T @ RHS @ W, np.eye(W.shape[1]))

        # possibly check CNS are orthonormal
        # this is only true for SUMCORR-AVRVAR MCCA i.e.
        # if no regularization is used
        if regs is None:
            assert np.allclose(common_norm_scores.T @ common_norm_scores,
                               np.eye(common_norm_scores.shape[1]))


def check_mcca_gevp(Xs, out, regs):
    """
    Checks the block loadings are the correct generalized eigenvectors.
    """
    block_loadings = out['block_loadings']
    evals = out['evals']
    centerers = out['centerers']

    Xs, n_blocks, n_samples, n_features = check_Xs(Xs, multiview=True,
                                                   return_dimensions=True)

    regs = check_regs(regs=regs, n_blocks=n_blocks)

    # make sure to apply centering transformations
    Xs = [centerers[b].transform(Xs[b]) for b in range(n_blocks)]

    # concatenated block loadings are the eigenvectors
    W = np.vstack(block_loadings)

    LHS, RHS = get_mcca_gevp_data(Xs, regs=regs)

    # check generalized eigenvector equation
    assert np.allclose(LHS @ W, RHS @ W @ np.diag(evals))

    # check normalization
    assert np.allclose(W.T @ RHS @ W, np.eye(W.shape[1]))


def check_mcca_class(mcca, Xs):
    assert np.allclose(mcca.common_norm_scores_, mcca.transform(Xs))
    for b in range(mcca.n_blocks_):
        assert np.allclose(mcca.blocks_[b].block_scores_,
                           mcca.blocks_[b].transform(Xs[b]))


def compare_kmcca_to_mcca(k_out, mcca_out):
    """
    Kernek MCCA with a linear kernel should give the same output as mcca i.e.
    the block scores, common normalized scores and evals should all be equal.
    """
    n_blocks = len(mcca_out['block_scores'])

    for b in range(n_blocks):
        ks = k_out['block_scores'][b]
        ms = mcca_out['block_scores'][b]

        assert np.allclose(ks, ms)

    for k in ['common_norm_scores', 'evals']:
        assert np.allclose(k_out[k], mcca_out[k])
