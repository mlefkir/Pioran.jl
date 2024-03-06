![Banner of pioran power spectrum inference of random time series](./docs/src/assets/banner_desc.svg)

Pioran is a Julia package to estimate the continuum bending power-law power spectrum of any time series. This method is based on Gaussian process regression with the fast algorithm of [Foreman-Mackey, et al. 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F/abstract) and by approximating the power spectrum model with basis functions. The method is described [Lefkir, et al. 2023 (in prep)].

## Installation

```julia
using Pkg; Pkg.add("Pioran")
```

## Documentation

See the documentation at [https://www.mehdylefkir.fr/Pioran.jl](https://www.mehdylefkir.fr/Pioran.jl).