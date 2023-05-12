# veccs

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

## documentation tools

The documentation is created with mkdocs using mkdocstrings. Please refer to

- https://www.mkdocs.org/
- https://mkdocstrings.github.io/
- https://mkdocstrings.github.io/griffe/docstrings/#google-style

To change the style of the docstrings change the mkdocs.yml file.
