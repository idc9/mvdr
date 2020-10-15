import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def rand_2block_fig2(n=100, dx=200, dy=10000, scale_x=5000, seed=None):
    """
    Samples from the distribution described in (Feng et al, 2018) figure 2.
    Note here rows are the observations i.e. data matrices are n x d_b where n = # observations.

    Parameters
    ----------
    n: int
        Number of observations. Must be divisible by 20.

    dx: int
        Number of x features. Must be divisible by 2.

    dy: int
        Number of y features. Must be divisible by 10.

    scale_x: float
        Scale of the x data.

    seed: None, int
        Random seed to generate data.
    """
    # TODO: also return loadings vectors

    if seed:
        np.random.seed(seed)

    #################
    # Sample X data #
    #################
    X_joint = np.hstack([np.vstack([np.ones((n // 2, dx // 2)),
                                    -1 * np.ones((n // 2, dx // 2))]),
                         np.zeros((n, dx // 2))]) * scale_x

    X_indiv = np.vstack([-1 * np.ones((n // 4, dx)),
                         np.ones((n // 4, dx)),
                         -1 * np.ones((n // 4, dx)),
                         np.ones((n // 4, dx))]) * scale_x

    X_noise = np.random.normal(loc=0, scale=1, size=(n, dx)) * scale_x

    X = X_joint + X_indiv + X_noise

    #################
    # Sample Y data #
    #################

    Y_joint = np.hstack([np.zeros((n, 4 * dy // 5)),
                         np.vstack([-1 * np.ones((n // 2, dy // 5)),
                                    np.ones((n // 2, dy // 5))])])

    Y_indiv_a = np.vstack([np.ones((n // 5, dy // 2)),
                           -1 * np.ones((n // 5, dy // 2)),
                           np.zeros((n // 5, dy // 2)),
                           np.ones((n // 5, dy // 2)),
                           -1 * np.ones((n // 5, dy // 2))])

    Y_indiv_b = np.vstack([np.ones((n // 4, dy // 2)),
                           -1 * np.ones((n // 2, dy // 2)),
                           np.ones((n // 4, dy // 2))])

    Y_indiv = np.hstack([Y_indiv_a, Y_indiv_b])

    Y_noise = np.random.normal(loc=0, scale=1, size=(n, dy))

    Y = Y_joint + Y_indiv + Y_noise

    return {'x': {'obs': X, 'joint': X_joint,
                  'indiv': X_indiv, 'noise': X_noise},

            'y': {'obs': Y, 'joint': Y_joint,
                  'indiv': Y_indiv, 'noise': Y_noise}}


def plot_ajive_fig2(data):
    """
    Plots figure 2 from AJIVE
    """
    plt.figure(figsize=[10, 20])

    # Observed
    plt.subplot(4, 2, 1)
    show_heatmap(data['x']['obs'])
    plt.title('X observed')
    plt.xlabel('x features')
    plt.ylabel('observations')

    plt.subplot(4, 2, 2)
    show_heatmap(data['y']['obs'])
    plt.title('Y observed')
    plt.xlabel('y features')

    # joint
    plt.subplot(4, 2, 3)
    show_heatmap(data['x']['joint'])
    plt.title('X Joint')

    plt.subplot(4, 2, 4)
    show_heatmap(data['y']['joint'])
    plt.title('Y Joint')

    # individual
    plt.subplot(4, 2, 5)
    show_heatmap(data['x']['indiv'])
    plt.title('X individual')

    plt.subplot(4, 2, 6)
    show_heatmap(data['y']['indiv'])
    plt.title('Y individual')

    # Noise
    plt.subplot(4, 2, 7)
    show_heatmap(data['x']['noise'])
    plt.title('X noise')

    plt.subplot(4, 2, 8)
    show_heatmap(data['y']['noise'])
    plt.title('Y noise')


def plot_jive_results_2blocks(estimates):
    """
    Heat map of JIVE results
    """

    plt.figure(figsize=[10, 20])

    # Observed
    plt.subplot(4, 2, 1)
    show_heatmap(estimates['x']['obs'])
    plt.title('X observed')
    plt.xlabel('x features')
    plt.ylabel('observations')

    plt.subplot(4, 2, 2)
    show_heatmap(estimates['y']['obs'])
    plt.title('Y observed')
    plt.xlabel('y features')

    # joint
    plt.subplot(4, 2, 3)
    show_heatmap(estimates['x']['joint'])
    plt.title('X Joint (estimated)')

    plt.subplot(4, 2, 4)
    show_heatmap(estimates['y']['joint'])
    plt.title('Y Joint (estimated)')

    # individual
    plt.subplot(4, 2, 5)
    show_heatmap(estimates['x']['indiv'])
    plt.title('X individual (estimated)')

    plt.subplot(4, 2, 6)
    show_heatmap(estimates['y']['indiv'])
    plt.title('Y individual (estimated)')

    # Noise
    plt.subplot(4, 2, 7)
    show_heatmap(estimates['x']['noise'])
    plt.title('X noise (estimated)')

    plt.subplot(4, 2, 8)
    show_heatmap(estimates['y']['noise'])
    plt.title('Y noise (estimated)')


def threshold(x, epsilon=1e-10):
    x[abs(x) < epsilon] = 0
    return x


def plot_jive_residuals_2block(data, estimates):

    plt.figure(figsize=[10, 15])

    # compute residuals
    epsilon = 1e-8
    R_joint_x = threshold(data['x']['joint'] - estimates['x']['joint'],
                          epsilon)
    R_indiv_x = threshold(data['x']['indiv'] - estimates['x']['indiv'],
                          epsilon)
    R_noise_x = threshold(data['x']['noise'] - estimates['x']['noise'],
                          epsilon)

    R_joint_y = threshold(data['y']['joint'] - estimates['y']['joint'],
                          epsilon)
    R_indiv_y = threshold(data['y']['indiv'] - estimates['y']['indiv'],
                          epsilon)
    R_noise_y = threshold(data['y']['noise'] - estimates['y']['noise'],
                          epsilon)

    # joint
    plt.subplot(3, 2, 1)
    show_heatmap(R_joint_x)
    plt.title('X Joint (residuals)')
    plt.xlabel('x features')
    plt.ylabel('observations')

    plt.subplot(3, 2, 2)
    show_heatmap(R_joint_y)
    plt.title('Y Joint (residuals)')
    plt.xlabel('y features')

    # individual
    plt.subplot(3, 2, 3)
    show_heatmap(R_indiv_x)
    plt.title('X individual (residuals)')

    plt.subplot(3, 2, 4)
    show_heatmap(R_indiv_y)
    plt.title('Y individual (residuals)')

    # Noise
    plt.subplot(3, 2, 5)
    show_heatmap(R_noise_x)
    plt.title('X noise (residuals)')

    plt.subplot(3, 2, 6)
    show_heatmap(R_noise_y)
    plt.title('Y noise (residuals)')


def show_heatmap(A):
    n, d = A.shape

    sns.heatmap(A, xticklabels=d // 5, yticklabels=n // 5,
                cmap='RdBu')
