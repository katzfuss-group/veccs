# veccs

A Python package for efficiently computing conditioning sets for Vecchia
approximations.

Gaussian processes (GP) are prevalent in statistics and machine learning.
However, the computational costs in large data settings render their use
infeasible. Among various other approximate approaches, Vecchia approximations
are a promising and popular method for using GPs with rich data (see [Katzfuss &
Guinness (2021)](https://doi.org/10.1214/19-STS755) for details). The quality of
the Vecchia approximation depends crucially on the order of the data ([Guinness
(2018)](https://doi.org/10.1080/00401706.2018.1437476)). However, determining a
good ordering can be difficult and computationally expensive. This package
efficiently implements exact and approximate methods for ordering the data.

These include

- maximin ordering
- conditioning sets

## Installation

Run the following command in your local virtual environment. Please make sure
that a c++ compiler and the python headers are installed.

`pip install -e .`

### Legacy dependencies

Previous versions of this package used `faiss-cpu` and `sklearn`. These
functions are still available, but are not installed by default. To install
them, run `pip install -e .[legacy]`.

Please note that that these functions are deprecated and maybe removed in the
future. It has been observed that the legacy code caused issues on some Mac
Systems when using specific versions of, e.g., `faiss-cpu >= 1.8` and soemtimes
in combination with `pytorch`.



## How to contribute

1. install the package with the additional dependencies for development using
   `pip install -e .[dev,docs]`
2. before pushing on `main` or a PR, run `pre-commit run --all-files` and
   `pytest`.
3. before pushing on `main` or merging a PR, make sure the code is well
   documented.

The documentenation can be viewed while editing the code using `mkdocs serve`.
