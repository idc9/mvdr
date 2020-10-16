import numpy as np
from sklearn.preprocessing import KernelCenterer, StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels
from numbers import Number
from textwrap import dedent

from mvdr.linalg_utils import svd_wrapper


def initial_svds(Xs, signal_ranks=None, normalized_scores=False,
                 center=True, precomp_svds=None, sval_thresh=None):
    """
    Computes a low rank SVD of each view in a list of data views.

    Parameters
    ----------
    Xs : list of array-likes or numpy.ndarray
        The list of data matrices each shaped (n_samples, n_features_b).

    signal_ranks: None, list of ints
        SVD rank to compute for each data view.
        If None, will compute full SVD.

    normalized_scores: bool
        Whether or not to return the normalized scores matrix U
        as the primary output (left singular vectors) or the unnormalized scores i.e. UD.

    center: bool or list
        Whether or not to center data views.
        Can pass in either one bool that applies to all views or a list of bools to specify different options for each view.

    precomp_svds: None, list of tuples
        (optional) Precomputed SVDs for some of the views. Each entry of the list should be None or the SVD output tuple (U, D, V).

    sval_thresh: None, float, list of floats
        (optional) Whether or not to theshold singular values i.e. delete SVD components whose singular value is below this threshold.
        If a list is passed in, different arguments to each view can be supplied.

    Output
    ------
    reduced, svds, means

    reduced: list of array-like
        The SVD reduced data matrices for each view.
        These are either U or UD.

    svds: list of tuples
        The low rank SVDs for each data view.
        Each entry is (U, D, V)

    means: list
        The means for each data view if they have been demeaned.
    """

    n_views = len(Xs)

    if precomp_svds is None:
        precomp_svds = [None] * n_views
    assert len(precomp_svds) == n_views

    if signal_ranks is None or isinstance(signal_ranks, Number):
        signal_ranks = [signal_ranks] * n_views
    assert len(signal_ranks) == n_views

    if sval_thresh is None or isinstance(sval_thresh, Number):
        sval_thresh = [sval_thresh] * n_views

    # center data views
    Xs, centerers = center_views(Xs, center=center)

    # possibly perform SVDs on some views
    svds = [None for b in range(n_views)]
    reduced = [None for b in range(n_views)]
    for b in range(n_views):

        if precomp_svds[b] is None:
            U_b, D_b, V_b = svd_wrapper(Xs[b], rank=signal_ranks[b])
        else:
            U_b, D_b, V_b = precomp_svds[b]

        # possibly threshold SVD components
        if sval_thresh[b] is not None:
            to_keep = D_b >= sval_thresh[b]
            if sum(to_keep) == 0:
                raise ValueError("all singular values of view {}"
                                 "where thresholded at {}. Either this"
                                 "view is zero or you should try a"
                                 " smaller threshold value".
                                 format(b, sval_thresh[b]))

            U_b = U_b[:, to_keep]
            D_b = D_b[to_keep]
            V_b = V_b[:, to_keep]

        svds[b] = U_b, D_b, V_b

        if normalized_scores:
            reduced[b] = U_b
        else:
            reduced[b] = U_b * D_b

    return reduced, svds, centerers


def process_view_kernel_args(n_views, kernel='linear', kernel_params={}):

    if not is_array(kernel):
        kernel = [kernel] * n_views

    if kernel_params is None:
        kernel_params = {}

    if isinstance(kernel_params, dict) or not is_array(kernel_params):
        kernel_params = [kernel_params] * n_views

    assert len(kernel) == n_views
    assert len(kernel_params) == n_views

    return kernel, kernel_params


def get_view_kernels(Xs, kernel='linear', kernel_params={},
                     filter_params=False, n_jobs=None):
    # dimension of the full co-kernel matrix
    n_views = len(Xs)

    kernel, kernel_params = \
        process_view_kernel_args(n_views=n_views,
                                 kernel=kernel,
                                 kernel_params=kernel_params)

    Ks = [None for _ in range(n_views)]
    for b in range(n_views):
        Ks[b] = pairwise_kernels(X=Xs[b], metric=kernel[b],
                                 filter_params=filter_params,
                                 n_jobs=n_jobs,
                                 **kernel_params[b])

    return Ks


_view_kern_docs = dict(basic=dedent("""
    kernel: str, callable or list of str/collable
        Which kernel to use. This is the metric argument to sklearn.metrics.pairwise.pairwise_kernels.
        If a list is provided, each view can have its own kernel.

    kernel_params: dict or list of dicts
        Key word arguments to sklearn.metrics.pairwise.pairwise_kernels.
        If a list is provided, each view can have its own key work arguments
    """), other=dedent("""
    filter_params: bool
        See sklearn.metrics.pairwise.pairwise_kernels documentation.

    n_jobs: int, None
        See sklearn.metrics.pairwise.pairwise_kernels documentation.
    """))

process_view_kernel_args.__doc__ = dedent("""
    Processes view kernel arguments.

    Parameters
    ----------
    {basic}

    Output
    ------
    kernel, kernel_params

    kernel: list

    kernel_params: list
    """.format(**_view_kern_docs))

get_view_kernels.__doc__ = dedent("""
    Gets the kernel matrices for each view.

    Parameters
    ----------
    Xs : list of array-likes or numpy.ndarray
        The list of data matrices each shaped (n_samples, n_features_b).

    {basic}

    {other}

    Output
    ------
    Ks: list of array-like
        The kernel matrices all size (n_samples, n_samples)
    """.format(**_view_kern_docs))


def is_array(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return True
    else:
        return False


def center_views(Xs, center=True):
    """
    Mean centers a list of data views.

    Parameters
    ----------
    Xs : list of array-likes or numpy.ndarray
        The list of data matrices each shaped (n_samples, n_features_b).

    center: bool or list
        Whether or not to center data views.
        Can pass in either one bool that applies to all views or a list of bools to specify different options for each view.

    Output
    ------
    views_centered, means

    views_centered: list of array-like
        The centered data view kernels.

    centerers: list StandardScaler()s
        The StandardScaler() object for each data view.
    """
    n_views = len(Xs)

    if isinstance(center, bool):
        center = [center] * n_views
    assert len(center) == n_views

    Xs_centered = [None for b in range(n_views)]
    centerers = [None for b in range(n_views)]

    for b in range(n_views):
        centerers[b] = StandardScaler(copy=True, with_mean=center[b],
                                      with_std=False)
        Xs_centered[b] = centerers[b].fit_transform(Xs[b])

    return Xs_centered, centerers


def center_kernel_views(Ks, center=True):
    """
    Centers a list of kernel matrix data views.

    Parameters
    ----------
    Ks: list of array-like
        The kernel matrices for each data view.

    center: bool or list
        Whether or not to center data views.
        Can pass in either one bool that applies to all views or a list of bools to specify different options for each view.

    Output
    ------
    views_centered, means

    views_centered: list of array-like
        The centered data view kernels.

    centerers: list KernelCenterer()s
        The KernelCenterer() object for each data view.
    """
    n_views = len(Ks)

    if isinstance(center, bool):
        center = [center] * n_views
    assert len(center) == n_views

    Ks_centered = [None for b in range(n_views)]
    centerers = [None for b in range(n_views)]

    for b in range(n_views):

        if center[b]:
            centerers[b] = KernelCenterer()
            Ks_centered[b] = centerers[b].fit_transform(Ks[b])
        else:
            centerers[b] = None
            Ks_centered[b] = Ks[b]

    return Ks_centered, centerers


def split(C, dims, axis=1):
    """
    Splits the columns or rows of C.
    Suppse C = [X_1, X_2, ..., X_B] is an (n x sum_b d_b) matrix.
    Returns a list of the constituent matrices as a list.

    Parameters
    ----------
    C: array-like, shape (n, sum_b d_b)
        The concatonated block matrix.

    dims: list of ints
        The dimensions of each matrix i.e. [d_1, ..., d_B]

    axis: int [0, 1]
        Which axis to split (1 mean columns 0 means rows)
    Output
    ------
    blocks: list of array-like
        [X_1, X_2, ..., X_B]
    """
    idxs = np.append([0], np.cumsum(dims))

    blocks = []
    if axis == 1:
        assert idxs[-1] == C.shape[1]

        for b in range(len(dims)):
            blocks.append(C[:, idxs[b]:idxs[b + 1]])

    elif axis == 0:
        for b in range(len(dims)):
            blocks.append(C[idxs[b]:idxs[b + 1], :])

    else:
        raise ValueError('axis must be either 0 or 1')

    return blocks
