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

## Disclaimer

This package is experimental and under active development. That means:

- The API cannot be considered stable.
- Testing has not been extensive as of now. Please check and verify!
- There is currently not much documentation beyond this readme.

In any case, this package comes with no warranty or guarantees.

## Usage

If you want to depend on this package in your own code, you can install it
using
```
pip install pip install https://github.com/katzfuss-group/veccs.git
```
or
similar using other managers. Please consider the disclaimer above and
consider to depend on a specific version or commit, e.g.,
```
pip install pip install https://github.com/katzfuss-group/veccs.git@v0.0.2
```
refers to the commit with the tag `v0.0.2`.

## Installation

The project is managed using [uv](https://docs.astral.sh/uv/). To install the
package, first install `uv`. Then, clone the repository and run the following
command

`pip uv sync`

in the root directory of the project. If you want to use the locked version of
the dependencies, run `pip uv sync --locked`.

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
   `pip uv sync --dev --group docs`
2. before pushing on `main` or a PR, run `pre-commit run --all-files` and
   `pytest`.
3. before pushing on `main` or merging a PR, make sure the code is well
   documented. The documentenation can be viewed while editing the code using
   `mkdocs serve`.

To run the commands, without activating the virtual environment, prepend `uv
run `, e.g., `uv run mkdocs serve`.
