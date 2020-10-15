import numpy as np

from mvdr.linalg_utils import svd_wrapper


def get_principal_angles(A, B, n_components=None, is_ortho=True, deg=False):
    """
    Returns the vector of principal angles between two subspaces.

    Parameters
    ----------
    A: array-like, (n, p)
        Orthonormal basis for the first subspace.

    B: array-like, (n, q)
        Orthonormal basis for the second subspace.

    n_components: None, int
        Only compute the first few principal angles.

    is_ortho: bool
        Are A, B orthonormal? If they are not, we will orthonormalized them with a QR decomposition.

    deg: bool
        Return the vector of principal angles in degrees. If False, will use radians.

    Output
    ------
    theta: array-like, (n_components, )
        The principal angles in radians.
    """
    A = np.array(A)
    B = np.array(B)

    A_is_vec = False
    B_is_vec = False
    if A.ndim == 1:
        A = A.reshape(-1, 1)
        A_is_vec = True

    if B.ndim == 1:
        B = B.reshape(-1, 1)
        B_is_vec = True

    if not is_ortho:
        A = np.linalg.qr(A)[0]
        B = np.linalg.qr(B)[0]

    if A_is_vec and B_is_vec:
        cos_theta = A.T @ B
        cos_theta = cos_theta.reshape(-1)
    else:
        _, cos_theta, __ = svd_wrapper(A.T @ B, rank=n_components)

    cos_theta = np.clip(cos_theta, a_min=-1, a_max=1)
    theta = np.arccos(cos_theta)

    if deg:
        theta = np.rad2deg(theta)

    return theta


def subspace_dist(A, B, method='proj_f', is_ortho=True):
    """
    Computes  distance between two subspaces.

    See section 4.3 of (Edelman et al, 1998) for definitions
    http://www.cnbc.cmu.edu/cns/papers/edelman98.pdf

    Parameters
    ----------
    A, B: array-like, (n, p)
        Bases of the two subspace

    method: str
        Must be one of ['arc', 'fubini',
                        'proj_2', 'proj_f',
                        'chordal_2', 'chordal_f']

    is_ortho: bool
        Are A, B orthonormal? If they are not, we will orthonormalized them with a QR decomposition.
    """
    # TODO: check for edge cases e.g. n = 1 etc

    method = method.lower()
    if method not in ['arc_length', 'fubini',
                      'proj_2', 'proj_f',
                      'chordal_2', 'chordal_f']:
        raise ValueError(" method = {} is not currently an option".
                         format(method))

    A = np.array(A)
    B = np.array(B)

    if A.ndim == 1:
        A = A.reshape(-1, 1)

    if B.ndim == 1:
        B = B.reshape(-1, 1)

    if not is_ortho:
        A = np.linalg.qr(A)[0]
        B = np.linalg.qr(B)[0]

    if method in ['arc_length', 'fubini', 'chordal_2', 'chordal_f']:
        theta = get_principal_angles(A, B)

    if method == 'arc_length':
        return np.linalg.norm(theta)

    elif method == 'fubini':
        cos_theta = np.cos(theta)
        return np.arccos(np.product(cos_theta))

    elif method == 'chordal_2':
        return max(2 * np.sin(.5 * theta))

    elif method == 'chordal_f':
        return np.linalg.norm(2 * np.sin(.5 * theta))

    if method == 'proj_f':
        diff = A @ A.T - B @ B.T
        return np.sqrt(.5) * np.linalg.norm(diff, ord='fro')

    elif method == 'proj_2':
        diff = A @ A.T - B @ B.T
        return np.linalg.norm(diff, ord=2)
