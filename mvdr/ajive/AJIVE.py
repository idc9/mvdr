from sklearn.base import BaseEstimator
from warnings import warn
from textwrap import dedent

from ya_pca.PCA import PCA
from mvlearn.utils import check_Xs

from mvdr.mcca.mcca import MCCA
from mvdr.mcca.MCCABlock import MCCABlock
from mvdr.ajive.ajive_fun import ajive, _ajive_docs
from mvdr.ajive.plot_ajive_diagnostic import plot_joint_diagnostic


class AJIVE(BaseEstimator):
    def __init__(self, init_signal_ranks=None,
                 center=True, common_loading_method='map_back',
                 check_joint_identif=True,
                 wedin_percentile=5, n_wedin_samples=1000,
                 rand_percentile=95, n_rand_samples=1000,
                 usr_joint_rank=None, usr_iniv_ranks=None,
                 store_full=True, final_decomp=True,
                 n_jobs=None):

        self.init_signal_ranks = init_signal_ranks

        self.center = center
        self.common_loading_method = common_loading_method

        self.wedin_percentile = wedin_percentile
        self.n_wedin_samples = n_wedin_samples

        self.rand_percentile = rand_percentile
        self.n_rand_samples = n_rand_samples

        self.check_joint_identif = check_joint_identif

        self.store_full = store_full
        self.final_decomp = final_decomp

        self.usr_joint_rank = usr_joint_rank
        self.usr_iniv_ranks = usr_iniv_ranks

        self.n_jobs = n_jobs

    def fit(self, Xs):

        Xs, n_blocks, n_samples, n_features = check_Xs(Xs, multiview=True,
                                                       return_dimensions=True)

        if self.usr_joint_rank is not None and self.check_joint_identif:
            warn('usr_joint_rank has been specififed, but check joint identifiability is also True.')

        usr_iniv_ranks = arg_checker(Xs=Xs,
                                     usr_iniv_ranks=self.usr_iniv_ranks)

        assert self.init_signal_ranks is not None and \
            len(self.init_signal_ranks) == n_blocks

        ajive_out = ajive(Xs=Xs,
                          init_signal_ranks=self.init_signal_ranks,
                          joint_rank=self.usr_joint_rank,
                          indiv_ranks=usr_iniv_ranks,
                          center=self.center,
                          check_joint_identif=self.check_joint_identif,
                          wedin_percentile=self.wedin_percentile,
                          n_wedin_samples=self.n_wedin_samples,
                          rand_percentile=self.rand_percentile,
                          n_rand_samples=self.n_rand_samples,
                          final_decomp=self.final_decomp,
                          store_full=self.store_full,
                          n_jobs=self.n_jobs)

        ##################
        # common results #
        ##################
        self.common_ = get_mcca_from_ajive_out(ajive_out)

        ##################
        # block specific #
        ##################
        self.block_specific_ = {}
        for b in range(n_blocks):
            self.block_specific_[b] = BlockSpecificResults(
                decomps=ajive_out['decomps'][b],
                centerer=ajive_out['centerers'][b],
                sv_threshold=ajive_out['sv_thresholds'][b],
                block_idx=b)

        #############
        # other data #
        #############

        self.rank_est_ = ajive_out['rank_est']
        self.n_blocks_ = n_blocks

        return self

    def get_ranks(self):
        """
        Output
        ------
        joint_rank: int

        indiv_ranks: dict of ints
            The individual rank for each block.
        """
        return self.common_.n_components, \
            [self.block_specific_[b].indiv_rank_
             for b in range(self.n_blocks_)]

    @property
    def is_fit(self):
        return hasattr(self, 'common_')

    def get_block_decomps(self):
        """
        Output
        ------
        full: dict of dict of np.arrays
            The joint, individual, and noise full estimates for each block.
        """
        full = {}
        for b in range(self.n_blocks_):
            full[b] = {'joint': self.block_specific_[b].joint_.full_,
                       'individual': self.block_specific_[b].individual_.full_,
                       'noise': self.block_specific_[b].noise_}

        return full

    def summary(self):
        """
        Returns a summary of AJIVE.
        """

        if self.is_fit:
            joint_rank, indiv_ranks = self.get_ranks()
            r = 'AJIVE, joint rank: {}'.format(joint_rank)
            for b in range(self.n_blocks_):
                r += ', block {} indiv rank: {}'.format(b, indiv_ranks[b])
            return r

        else:
            return 'AJIVE has not been fit.'

    ###########
    # sklearn #
    ###########

    def transform(self, Xs):
        """

        Parameters
        ----------
        X: array-like, shape (n_new_samples, n_features)
            The data to project.

        Output
        ------
        s: array-like, shape (n_new_samples, n_components)
            The projections of the new data. If X is a pd.DataFrame
            then s will be as well.
        """
        # TODO: what to do when there is no joint rank
        return self.common_.transform(Xs)

    #################
    # visualization #
    #################

    def plot_joint_diagnostic(self, fontsize=20):
        """
        Plots joint rank threshold diagnostic plot
        """

        rand_cutoff = self.rank_est_['rand']['threshold']
        rand_sv_samples = self.rank_est_['rand']['samples']

        wedin_cutoff = self.rank_est_['wedin']['threshold']
        wedin_sv_samples = self.rank_est_['wedin']['samples']

        all_common_svals = self.rank_est_['all_common_svals']
        identif_dropped = self.rank_est_['identif_dropped']
        joint_rank = self.get_ranks()[0]

        plot_joint_diagnostic(all_common_svals=all_common_svals,
                              joint_rank=joint_rank,
                              wedin_cutoff=wedin_cutoff,
                              rand_cutoff=rand_cutoff,
                              wedin_sv_samples=wedin_sv_samples,
                              rand_sv_samples=rand_sv_samples,
                              wedin_percentile=self.wedin_percentile,
                              rand_percentile=self.rand_percentile,
                              min_signal_rank=min(self.init_signal_ranks),
                              identif_dropped=identif_dropped,
                              fontsize=fontsize)


AJIVE.__doc__ = dedent("""
    Angle-based Joint and Individual Variation Explained

    Parameters
    ----------
    {basic_args}

    Attributes
    ----------
    common_: mvdr.mcca.MCCA
        Stores the common/joint space estimates as a MCCA object.

    block_specific_: mvdr.ajive.AJIVE.BlockSpecificResults
        Stores the block specific results including the block specific
        joint and individual decompositions.

    rank_est_: dict
        Data for joint rank selection e.g. the wedin samples.

    init_signal_ranks: list
        The initial signal ranks
    """.format(**_ajive_docs))


def get_mcca_from_ajive_out(ajive_out):
    joint_rank = ajive_out['common']['rank']
    if joint_rank == 0:
        return None

    common = MCCA(n_components=joint_rank,
                  signal_ranks=ajive_out['init_signal_ranks'],
                  center=ajive_out['center'])

    common_out = ajive_out['common']
    common.common_norm_scores_ = common_out['common_scores']
    common.evals_ = common_out['sqsvals']

    n_blocks = len(ajive_out['centerers'])
    blocks = [None for b in range(n_blocks)]
    for b in range(n_blocks):
        bs = common_out['block_scores'][b]
        bl = common_out['block_loadings'][b]
        cent = ajive_out['centerers'][b]
        blocks[b] = MCCABlock(block_scores=bs,
                              block_loadings=bl,
                              centerer=cent)
    return common


def arg_checker(Xs, usr_iniv_ranks):

    n_blocks = len(Xs)

    ################################
    # parse block specific options #
    ################################

    block_indiv_ranks = [None for b in range(n_blocks)]
    # block_init_svds = [None for b in range(n_blocks)]

    if usr_iniv_ranks is not None:
        for b in range(n_blocks):
            block_indiv_ranks[b] = usr_iniv_ranks[b]

    return block_indiv_ranks


def get_pca(decomps, centerer):
    U = decomps['scores']
    D = decomps['svals']
    V = decomps['loadings']
    n_components = decomps['rank']

    # setup PCA object
    pca = PCA(n_components=n_components, center=centerer.mean_ is not None)
    pca.scores_ = U
    pca.svals_ = D
    pca.loadings_ = V
    pca.centerer_ = centerer
    pca.tot_variance_ = sum(D ** 2)

    return pca


class BlockSpecificResults(object):
    """
    Contains the block specific results.

    Attributes
    ----------
    joint_: ya_pca.PCA.PCA
        Block specific joint PCA.
        Has an extra attribute joint.full_ which contains the full block
        joint estimate.

    individual_: ya_pca.PCA.PCA
        Block specific individual PCA.
        Has an extra attribute individual.full_ which contains the full block
        individual estimate.

    noise_: array-like
        The full noise block estimate.

    block_idx_:
        Index of this block.

    shape_: tuple
        (n_observations, n_features)

    Note that both joint_ and individual_ have an additional attibute .full_
    (e.g. joint_.full_) which contains the full reconstruced matrix.

    """
    def __init__(self, decomps, centerer,
                 sv_threshold=None, block_idx=None):

        self.block_idx_ = block_idx

        ########################
        # block specific joint #
        ########################
        if decomps['joint']['scores'] is None:
            self.joint_ = None
            self.joint_rank_ = 0
        else:
            self.joint_ = get_pca(decomps['joint'], centerer=centerer)
            self.joint_.full_ = decomps['joint']['full']
            self.joint_rank_ = self.joint_.n_components_

        #############################
        # block specific individual #
        #############################
        if decomps['individual']['scores'] is None:
            self.individual_ = None
            self.indiv_rank_ = 0
        else:
            self.individual_ = get_pca(decomps['individual'],
                                       centerer=centerer)
            self.individual_.full_ = decomps['individual']['full']
            self.indiv_rank_ = self.individual_.n_components_
        #################################
        # block specific noise estimate #
        ################################
        self.noise_ = decomps['noise']

        # other metadata
        self.sv_threshold_ = sv_threshold

    def __repr__(self):
        return 'Block: {}, individual rank: {}, joint rank: {}'.\
            format(self.block_idx_, self.indiv_rank_, self.joint_rank_)
