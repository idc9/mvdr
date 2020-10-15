import numpy as np
from sklearn.utils import check_random_state

from mvdr.ajive.utils import sample_parallel
from mvdr.linalg_utils import rand_orthog


def sample_randdir(n, dims, n_samples=1000, random_state=None, n_jobs=None):
    """
    Draws samples for the random direction bound.

    Parameters
    ----------
    n: int
        Dimension of the ambient space.

    dims: list of ints
        Dimensions of each subspace.

    n_samples: int
        Number of samples to draw.

    random_state: None, int
        Seed for samples.

    n_jobs: int, None
        Number of jobs for parallel processing using
        sklearn.externals.joblib.Parallel. If None, will not use parallel
        processing.

    Output
    ------
    random_sv_samples: np.array, shape (n_samples, )
        The samples.
    """

    # TODO: what to do about seed for parallelism

    random_sv_samples = sample_parallel(fun=_get_rand_sample,
                                        n_jobs=n_jobs,
                                        n_samples=n_samples,
                                        random_state=random_state,
                                        n=n,
                                        dims=dims)
    return np.array(random_sv_samples)


def _get_rand_sample(n, dims, random_state=None):
    rng = check_random_state(random_state)

    # compute largest squared singular value of random joint matrix
    M = [rand_orthog(n, d, random_state=rng) for d in dims]
    M = np.bmat(M)
    return np.linalg.norm(M, ord=2) ** 2
