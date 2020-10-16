from numbers import Number
from itertools import combinations
import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from inspect import getargspec


def powerset(x, min_size=1, max_size=None, descending=True):
    """
    Iterates over the power set.

    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    Parameters
    ----------
    x: iterable, int

    min_size: int
        Ignore sets smaller than this.

    max_size: int
        Ignore sets larger than this.

    descending: bool
        Go from largest to smallest.
    """

    if isinstance(x, Number):
        x = range(x)
    s = list(x)

    assert min_size >= 0

    if max_size is None:
        max_size = len(s)
    assert max_size <= len(s)
    assert min_size <= max_size

    sizes = range(min_size, max_size + 1)
    if descending:
        sizes = reversed(sizes)

    for size in sizes:
        for S in combinations(s, size):
            yield S


def sample_random_seeds(n_seeds=1, random_state=None):
    """
    Samples a bunch of random seeds.

    Parameters
    ----------
    n_seeds: int
        Number of seeds to draw.

    random_state: None, int
        Metaseed used to determine the seeds.

    """
    # get seeds for each sample
    rng = check_random_state(random_state)
    return rng.randint(np.iinfo(np.int32).max, size=n_seeds)


def draw_samples(fun, n_draws=1, n_jobs=None, backend=None,
                 random_state=None, args=[], kws={}):
    """
    Computes samples possibly in parralel using joblib.
    Each sample gets its own seed.

    Parameters
    ---------
    fun: callable
        The sampling function. It should take random_state as a key word argument.

    n_draws: int
        Number of samples to draw.

    n_jobs: None, -1, int
        The maximum number of concurrently running jobs.
        If None, will not sample in parralel.
        If -1 will use all available cores. See joblib.Parallel.

    backend: str, ParallelBackendBase instance or None, default: 'loky
        Specify the parallelization backend implementation.
        See joblib.Parallel

    args: list
        The list of positional arguments to be passed into fun as fun(*arg).

    kws: list
        The dict of key word arguments to be passed into fun as fun(**kws).

    random_state: None, int
        Metaseed used to determine the seeds for each sample.

    Output
    ------
    samples: list
        Each entry of samples is the output of one call to
        fun(random_state=seed, *args, **kwargs)
    """
    if 'random_state' not in getargspec(fun).args:
        raise ValueError("func must take 'random_state' as an argument")

    seeds = sample_random_seeds(n_draws, random_state)

    if n_jobs is not None:
        return Parallel(n_jobs=n_jobs, backend=backend) \
            (delayed(fun)(random_state=seeds[d],
                          *args, **kws) for d in range(n_draws))

    else:
        return [fun(random_state=seeds[s], *args, **kws)
                for s in range(n_draws)]
