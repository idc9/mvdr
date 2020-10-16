import numpy as np
from sklearn.utils import check_random_state

from mvdr.utils import draw_samples, sample_random_seeds


def get_wedin_samples(X, U, D, V, rank, n_draws=1000,
                      random_state=None, n_jobs=None, backend=None):
    """
    Computes the wedin bound using the sample-project procedure. This method
    does not require the full SVD.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The data block.

    U, D, V:
        The partial SVD of X

    rank: int
        The rank of the signal space

    n_draws: int
        Number of samples to draw for resampling procedure.

    random_state: int, None
        Seed for random samples.

    n_jobs: None, -1, int
        The maximum number of concurrently running jobs,
        Number of cores to use. If None, will not sample in parralel.
        If -1 will use all available cores. See joblib.Parallel.

    backend: str, ParallelBackendBase instance or None, default: 'loky
        Specify the parallelization backend implementation.
        See joblib.Parallel

    """
    # TODO: what to do about seed for parallelism

    random_states = sample_random_seeds(2, random_state)

    basis = V[:, 0:rank]
    V_norm_samples = draw_samples(fun=_get_sample,
                                  n_draws=n_draws,
                                  random_state=random_states[0],
                                  n_jobs=n_jobs,
                                  backend=backend,
                                  kws={'X': X, 'basis': basis})

    basis = U[:, 0:rank]
    U_norm_samples = draw_samples(fun=_get_sample,
                                  n_draws=n_draws,
                                  random_state=random_states[1],
                                  n_jobs=n_jobs,
                                  backend=backend,
                                  kws={'X': X.T, 'basis': basis})

    V_norm_samples = np.array(V_norm_samples)
    U_norm_samples = np.array(U_norm_samples)

    sigma_min = D[rank - 1]  # TODO: double check -1
    wedin_bound_samples = [min(max(U_norm_samples[r],
                                   V_norm_samples[r]) / sigma_min, 1)
                           for r in range(n_draws)]

    return wedin_bound_samples


def _get_sample(X, basis, random_state=None):
    dim, rank = basis.shape

    rng = check_random_state(random_state)

    # sample from isotropic distribution
    vecs = rng.normal(size=(dim, rank))

    # project onto space orthogonal to cols of B
    # vecs = (np.eye(dim) - np.dot(basis, basis.T)).dot(vecs)
    vecs = vecs - np.dot(basis, np.dot(basis.T, vecs))

    # orthonormalize
    vecs, _ = np.linalg.qr(vecs)

    # compute  operator L2 norm
    return np.linalg.norm(X.dot(vecs), ord=2)
