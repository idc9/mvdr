from numbers import Number
from itertools import combinations


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
