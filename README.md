StoPy
========

StoPy is a numerical software for stoch Stochastic modelling in Python, which includes also inverse problems using Bayesian updating.


## Requirements
The code is optimised for [Python](https://www.python.org) (version 3.6) and
depends on the following numerical libraries:
- [NumPy](http://www.numpy.org) (version 1.15.4) and [SciPy](https://www.scipy.org) (version 1.1.0) for scientific computing as well as on the
- [Scikit-sparse](https://pypi.org/project/scikit-sparse/) (version 0.4.4) for Cholesky decomposition

## Structure of the code
At the moment the code contains the examples presented in the paper in [References](#references),
which is focused on Bayesian updating.

The structure of the code is following:

- `general` package contains all auxiliary functions used in the code,
- `research_articles` package contains the numerial examples used in the paper,
- `uq` package contains all code related to uncertainty quantification and Bayesian updating.


All files in `research_articles` can be run as a python script,
i.e. the file `name_of_file.py` can be run using the following shell command

```
python3 name_of_file.py
```


## License
This repository is distributed under an open MIT license.
If you find the code and approach interesting, you are kindly asked to cite the papers
in [References](#references).

## References
The code is based on the following papers, where you can find more theoretical information.

- Jaroslav Vondřejc and Hermann G. Matthies: *Accurate computation of conditional expectation for highly non-linear problems*. 2018. arXiv:1806.03234

 Particularly see the folder 'research_papers' with code used in the publication.
