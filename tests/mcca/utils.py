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
                                   # view_scores, view_loadings,
                                   regs=None,
                                   check_normalization=False):
    """
    Checks the scores and loadings output for regularized mcca.

    - view scores are projections of views onto loadings
    - common noramlized scores are column normalized version of sum of scores

    - (optional) check normalization of loadings; this should be done for MCCA, but not for informative MCCA.
    """

    view_loadings = out['view_loadings']
    view_scores = out['view_scores']
    common_norm_scores = out['common_norm_scores']
    centerers = out['centerers']

    Xs, n_views, n_samples, n_features = check_Xs(Xs, multiview=True,
                                                  return_dimensions=True)

    # make sure to apply centering transformations
    Xs = [centerers[b].transform(Xs[b]) for b in range(n_views)]

    for b in range(n_views):

        # check view scores are projections of views onto view loadings
        assert np.allclose(Xs[b] @ view_loadings[b], view_scores[b])

    # check common norm scores are the column normalized sum of the
    # view scores
    cns_pred = normalize_cols(sum(bs for bs in view_scores))[0]
    assert np.allclose(cns_pred, common_norm_scores)

    if check_normalization:

        # concatenated loadings are orthonormal in the inner produce
        # induced by the RHS of the GEVP
        W = np.vstack(view_loadings)
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
    Checks the view loadings are the correct generalized eigenvectors.
    """
    view_loadings = out['view_loadings']
    evals = out['evals']
    centerers = out['centerers']

    Xs, n_views, n_samples, n_features = check_Xs(Xs, multiview=True,
                                                  return_dimensions=True)

    regs = check_regs(regs=regs, n_views=n_views)

    # make sure to apply centering transformations
    Xs = [centerers[b].transform(Xs[b]) for b in range(n_views)]

    # concatenated view loadings are the eigenvectors
    W = np.vstack(view_loadings)

    LHS, RHS = get_mcca_gevp_data(Xs, regs=regs)

    # check generalized eigenvector equation
    assert np.allclose(LHS @ W, RHS @ W @ np.diag(evals))

    # check normalization
    assert np.allclose(W.T @ RHS @ W, np.eye(W.shape[1]))


def check_mcca_class(mcca, Xs):
    assert np.allclose(mcca.common_norm_scores_, mcca.transform(Xs))
    for b in range(mcca.n_views_):
        assert np.allclose(mcca.views_[b].view_scores_,
                           mcca.views_[b].transform(Xs[b]))


def compare_kmcca_to_mcca(k_out, mcca_out):
    """
    Kernek MCCA with a linear kernel should give the same output as mcca i.e.
    the view scores, common normalized scores and evals should all be equal.
    """
    n_views = len(mcca_out['view_scores'])

    for b in range(n_views):
        ks = k_out['view_scores'][b]
        ms = mcca_out['view_scores'][b]

        assert np.allclose(ks, ms)

    for k in ['common_norm_scores', 'evals']:
        assert np.allclose(k_out[k], mcca_out[k])
