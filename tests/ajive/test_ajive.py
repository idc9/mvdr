import numpy as np
from ..mcca.utils import generate_mcca_test_data

from mvdr.ajive.AJIVE import AJIVE


def test_AJIVE():

    for Xs in generate_mcca_test_data():
        n_views = len(Xs)
        init_signal_ranks = [3] * n_views

        ajive = AJIVE(init_signal_ranks=init_signal_ranks).fit(Xs)

        # make sure it runs when with user specified joint rank
        ajive = AJIVE(init_signal_ranks=init_signal_ranks,
                      check_joint_identif=False,
                      usr_joint_rank=1).fit(Xs)

        # make sure we setup common MCCA correctly
        ajive.common_.transform(Xs)
        assert np.allclose(ajive.common_.common_norm_scores_,
                           ajive.common_.transform(Xs))

        # make sure it runs when joint ranks = 0
        ajive = AJIVE(init_signal_ranks=init_signal_ranks,
                      usr_joint_rank=0).fit(Xs)
