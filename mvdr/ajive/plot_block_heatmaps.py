import matplotlib.pyplot as plt
import seaborn as sns


def block_heatmaps(Xs):
    """
    Plots a heat map of a bunch of data blocks
    """
    n_blocks = len(Xs)

    plt.figure(figsize=[5 * n_blocks, 5])

    for k in range(n_blocks):

        plt.subplot(1, n_blocks, k + 1)
        sns.heatmap(Xs[k], xticklabels=False, yticklabels=False)
        plt.title('block ' + str(k))


def plot_jive_full_estimates(full_block_estimates, Xs):
    """
    Plots the full JVIE estimates: X, J, I, E
    """
    n_blocks = len(full_block_estimates)

    plt.figure(figsize=[10, n_blocks * 10])

    for b in range(n_blocks):

        # grab data
        X = Xs[b]
        J = full_block_estimates[b]['J']
        I = full_block_estimates[b]['I']
        E = full_block_estimates[b]['E']

        # observed data
        plt.subplot(4, n_blocks, b + 1)
        sns.heatmap(X, xticklabels=False, yticklabels=False)
        plt.title('block ' + str(b) + ' observed data')

        # full joint estimate
        plt.subplot(4, n_blocks, b + n_blocks + 1)
        sns.heatmap(J, xticklabels=False, yticklabels=False)
        plt.title('joint')

        # full individual estimate
        plt.subplot(4, n_blocks, b + 2 * n_blocks + 1)
        sns.heatmap(I, xticklabels=False, yticklabels=False)
        plt.title('individual')

        # full noise estimate
        plt.subplot(4, n_blocks, b + 3 * n_blocks + 1)
        sns.heatmap(E, xticklabels=False, yticklabels=False)
        plt.title('noise ')
