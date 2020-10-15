# Multi-view dimensionality reduction

A sklearn compatible python package for multi-view dimensionality reduction including [multi-view canonical correlation analysis](docs/mcca_gevp_docs.pdf) and [AJIVE](https://www.sciencedirect.com/science/article/pii/S0047259X1730204X).


## Installation

<!--
```
pip install mvdr (coming soon!)
```
-->

```
git clone https://github.com/idc9/mvdr.git
python setup.py install
```

Note the mvdr.ajive assumes you have installed `ya_pca` which can be found at [https://github.com/idc9/ya_pca.git](https://github.com/idc9/ya_pca.git).

## Example

```python
from mvdr.mcca.mcca import MCCA
from mvdr.mcca.k_mcca import KMCCA
from mvdr.toy_data.joint_fact_model import sample_joint_factor_model

# sample data from a joint factor model with 3 components
# each data block is X_b = U diag(svals) W_b^T + E_b where
# where the joint scores matrix U and each of the block loadings matrices, W_b, are orthonormal and E_b is a random noise matrix.
Xs, U_true, Ws_true = sample_joint_factor_model()

# fit MCCA (this is the SUMCORR-AVGVAR flavor of multi-CCA)
mcca = MCCA(n_components=3).fit(Xs)

# MCCA with regularization
mcca = MCCA(n_components=3, regs=0.1).fit(Xs) # add regularization

# informative MCCA where we first apply PCA to each data matrix
mcca = MCCA(n_components=3, signal_ranks=[5, 5, 5]).fit(Xs)

# kernel-MCCA
kmcca = KMCCA(n_components=3, regs=.1, kernel='linear')
kmcca.fit(Xs)
```

# Help and support

Additional documentation, examples and code revisions are coming soon. For questions, issues or feature requests please reach out to Iain: <idc9@cornell.edu>.

<!--
## Testing
Testing is done using nose.
-->

## Contributing

We welcome contributions to make this a stronger package: data examples, bug fixes, spelling errors, new features, etc.

<!--
# Citation

You can use the below badge to generate a DOI and bibtex citation

 [![DOI](https://zenodo.org/badge/TODO.svg)](https://zenodo.org/badge/latestdoi/TODO)
-->
