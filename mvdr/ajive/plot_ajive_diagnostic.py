import numpy as np
import matplotlib.pyplot as plt


def plot_joint_diagnostic(all_common_svals,
                          joint_rank,
                          wedin_cutoff,
                          rand_cutoff,
                          wedin_sv_samples,
                          rand_sv_samples,
                          min_signal_rank,
                          identif_dropped=None,
                          wedin_percentile=5,
                          rand_percentile=95,
                          fontsize=20):

    """
    Plots the AJIVE diagnostic plot described in https://arxiv.org/pdf/1704.02060.pdf

    TODO: finish documenting

    """
    # assert not ((rand_cutoff is None) and (wedin_cutoff is None))
    show_wedin = True
    show_rand = True
    if wedin_sv_samples is None:
        show_wedin = False
    if rand_sv_samples is None:
        show_rand = False
    assert show_rand or show_wedin

    fontsize_large = fontsize
    fontsize_small = int(fontsize_large * .75)

    # compute sv_threshold
    # wedin_cutoff = np.percentile(wedin_sv_samples, wedin_percentile)
    # rand_cutoff = np.percentile(random_sv_samples, rand_percentile)
    if rand_cutoff is None:
        svsq_cutoff = wedin_cutoff
    elif wedin_cutoff is None:
        svsq_cutoff = rand_cutoff
    else:
        svsq_cutoff = max(rand_cutoff, wedin_cutoff)
    # joint_rank_est = sum(joint_svals ** 2 > svsq_cutoff)

    if show_wedin:
        wedin_low = np.percentile(wedin_sv_samples, 100 - wedin_percentile)
        wedin_high = np.percentile(wedin_sv_samples, wedin_percentile)

    if show_rand:
        rand_low = np.percentile(rand_sv_samples, rand_percentile)
        rand_high = np.percentile(rand_sv_samples, 100 - rand_percentile)

    if show_rand and not show_wedin:
        rand_lw = 4
    elif show_wedin and not show_rand:
        wedin_lw = 4
    else:
        if rand_cutoff > wedin_cutoff:
            rand_lw = 4
            wedin_lw = 2

        else:
            rand_lw = 2
            wedin_lw = 4

    if show_rand:
        rand_label = 'random {:d}th percentile ({:1.3f})'.\
            format(rand_percentile, rand_cutoff)

    if show_wedin:
        wedin_label = 'wedin {:d}th percentile ({:1.3f})'.\
            format(wedin_percentile, wedin_cutoff)

    # wedin cutoff
    if show_wedin:
        plt.axvspan(wedin_low, wedin_high, alpha=0.1, color='blue')
        plt.axvline(wedin_cutoff,
                    color='blue',
                    ls='dashed',
                    lw=wedin_lw,
                    label=wedin_label)

    # random cutoff
    if show_rand:
        plt.axvspan(rand_low, rand_high, alpha=0.1, color='red')
        plt.axvline(rand_cutoff,
                    color='red',
                    ls='dashed',
                    lw=rand_lw,
                    label=rand_label)

    # plot joint singular values
    first_joint = True
    first_nonjoint = True
    svals = all_common_svals[0:min_signal_rank]
    for i, sv in enumerate(svals):
        sv_sq = sv**2

        if sv_sq > svsq_cutoff:

            label = 'joint singular value' if first_joint else ''
            first_joint = False

            color = 'black'
        else:

            label = 'nonjoint singular value' if first_nonjoint else ''
            first_nonjoint = False

            color = 'grey'

        # components dropped due to identifiablility constraint
        # TODO: maybe add a line type to this
        if identif_dropped is not None and i in identif_dropped:
            color = 'grey'

        plt.axvline(sv_sq, ymin=.05, ymax=.95,
                    color=color,
                    label=label,
                    lw=2, zorder=2)

        spread = .05 * (max(svals) - min(svals))
        plt.hlines(y=1 - (.25 + .25 * i / min_signal_rank),
                   xmin=sv_sq - spread, xmax=sv_sq + spread,
                   color=color)

    plt.xlabel('squared singular value', fontsize=fontsize_large)
    plt.legend(fontsize=fontsize_small)
    plt.ylim([0, 1])
    # plt.xlim(1)
    plt.title('joint singular value thresholding'
              ' (joint rank estimate = {:d})'.format(joint_rank),
              fontsize=fontsize_large)

    # format axes
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axes.get_yaxis().set_ticks([])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize_small)
