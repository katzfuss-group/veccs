# veccs

A Python package for efficiently computing conditioning sets for Vecchia
approximations.

Gaussian processes (GP) are prevalent in statistics and machine learning.
However, the computational costs in large data settings render their use
infeasible. Among various other approximate approaches, Vecchia approximations
are a promising and popular method for using GPs with rich data (see [Katzfuss &
Guinness (2021)](https://doi.org/10.1214/19-STS755) for details). However, the
quality of the Vecchia approximation depends crucially on the order of the data
([Guinness (2018)](https://doi.org/10.1080/00401706.2018.1437476)). However,
determining a good/optimal order can be difficult and computationally expensive.
This package efficiently implements exact and approximate methods for ordering
the data.

These include

- minmax ordering
- conditioning sets

## Installation

Run the following command in your local virtual environment. Please make sure
that a c++ compiler and the python headers are installed.

`pip install -e .`

## How to contribute

1. install the package with the additional dependencies for development using
   `pip install -e .[dev,docs]`
2. before pushing on `main` or a PR, run `pre-commit run --all-files` and
   `pytest`.
3. before pushing on `main` or merging a PR, make sure the code is well
   documented.

The documentenation can be viewed while editing the code using `mkdocs serve`.
