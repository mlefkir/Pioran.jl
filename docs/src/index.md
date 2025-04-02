# Pioran

`Pioran` is a Julia package for the inference of bending power-law power spectra from arbitrarily sampled time series using scalable Gaussian processes.

This method was designed to aid the estimation of bending frequencies and slopes in the power spectra of irregularly sampled light curves of active galaxies but it can be applied to any time series data. The method is formally introduced in [2025arXiv250105886L](@citet) and uses the fast `celerite` algorithm of [2017AJ....154..220F](@citet).

## Installation

The package can be installed add follows:

```julia
import Pkg; Pkg.add("Pioran")
```

## Content

```@contents
Pages = vcat(["getting_started.md"],Main.BASIC_PAGES,Main.ADVANCED_PAGES,["api.md"])
Depth = 2
```
