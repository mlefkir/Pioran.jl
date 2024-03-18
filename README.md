![Banner of pioran power spectrum inference of random time series](./docs/src/assets/banner_desc.svg)

[![Documentation](https://github.com/mlefkir/Pioran.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/mlefkir/Pioran.jl/actions/workflows/documentation.yml) [![Build](https://github.com/mlefkir/Pioran.jl/actions/workflows/testbuild.yml/badge.svg)](https://github.com/mlefkir/Pioran.jl/actions/workflows/testbuild.yml)
[![codecov](https://codecov.io/gh/mlefkir/Pioran.jl/graph/badge.svg?token=88LNFU2VKD)](https://codecov.io/gh/mlefkir/Pioran.jl)

Pioran is a Julia package to estimate the power spectrum of any time series using bending power laws. This method is based on Gaussian process regression with the fast algorithm of [Foreman-Mackey, et al. 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F/abstract) and by approximating the power spectrum model with basis functions. The method is described in Lefkir, et al. 2023 (in prep).

## Installation

```julia
using Pkg; Pkg.add("Pioran")
```

## Documentation

See the documentation at [https://www.mehdylefkir.fr/Pioran.jl](https://www.mehdylefkir.fr/Pioran.jl).

## Examples

Example scripts are provided in the [examples](./examples) directory. To infer the parameters of the power spectrum, I use either [`Turing.jl`](https://github.com/TuringLang/Turing.jl) for Hamiltonian Monte Carlo or the Python library [`ultranest`](https://github.com/JohannesBuchner/UltraNest) for nested sampling. The scripts are written in a way that you can use either of these libraries.
