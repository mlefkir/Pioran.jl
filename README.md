[![Banner of pioran power spectrum inference of random time series](./docs/src/assets/banner_desc.svg)](#)

[![Documentation](https://github.com/mlefkir/Pioran.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/mlefkir/Pioran.jl/actions/workflows/documentation.yml) [![Build](https://github.com/mlefkir/Pioran.jl/actions/workflows/testbuild.yml/badge.svg)](https://github.com/mlefkir/Pioran.jl/actions/workflows/testbuild.yml)
[![codecov](https://codecov.io/gh/mlefkir/Pioran.jl/graph/badge.svg?token=88LNFU2VKD)](https://codecov.io/gh/mlefkir/Pioran.jl)

Pioran is a Julia package to estimate bending power-law power spectrum of time series. This method uses Gaussian process regression with the fast algorithm of [Foreman-Mackey, et al. 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F/abstract). The bending power-law model is approximated using basis functions as shown in the Figure below:

[![Basis functions of the bending power-law model](./extra/approximation.svg)](#)

The method is described in [Lefkir et. al 2025](https://ui.adsabs.harvard.edu/abs/2025MNRAS.539.1775L/abstract), where it is used to model the random flux variability observed in active galaxies.

## Installation

```julia
using Pkg; Pkg.add("Pioran")
```

## Documentation

Read the documentation here: [https://mlefkir.github.io/Pioran.jl/stable/](https://mlefkir.github.io/Pioran.jl/stable/).

## Examples

Example scripts are provided in the [examples](./examples) directory. To infer the parameters of the power spectrum, I use either [`Turing.jl`](https://github.com/TuringLang/Turing.jl) for Hamiltonian Monte Carlo or the Python library [`ultranest`](https://github.com/JohannesBuchner/UltraNest) for nested sampling.

### Ultranest

Here a very quick example on how to use it with `ultranest`. I assume you have installed `PyCall` and `ultranest` following the guide in the documentation [`here`](https://mlefkir.github.io/Pioran.jl/stable/ultranest/).

Assuming you have a Gaussian time series `y` at times `t` with errorbars `σ`. The GP can be built using a function as follows:

```julia
using Pioran, Distributions

function GP_model(t, y, σ, params, n_components = 20, basis_function = "DRWCelerite")
    T = (t[end] - t[1]) # duration of the time series
    Δt = minimum(diff(t)) # min time separation

    f_min, f_max = 1 / T, 1 / Δt / 2

    # Get the parameters
    α₁, f₁, α₂, variance, ν, μ = params

    # Rescale the measurement variance
    σ² = ν .* σ .^ 2

    # Define the power spectral density function
    𝓟 = SingleBendingPowerLaw(α₁, f₁, α₂)

    # Approximate the PSD to form a covariance function
    𝓡 = approx(𝓟, f_min, f_max, n_components, variance, basis_function = basis_function)

    # Build the GP
    GP = ScalableGP(μ, 𝓡)

    # return the conditioned GP on the times and errors and the transformed values
    return GP(t, σ²)
end
```

The log-likelihood can be obtained using the `logpdf` function from `Distributions.jl`:

```julia
function loglikelihood(t, y, σ, params)
    GP = GP_model(t, y, σ, params)
    return logpdf(GP, y)
end
logl(pars) = loglikelihood(t, y, yerr, pars)
```
We use distributions from  `Distributions.jl` to define the priors for nested sampling. For this example, we can have:

```julia
function prior_transform(cube)
    α₁ = quantile(Uniform(0.0, 1.5), cube[1])
    f₁ = quantile(LogUniform(1e-3, 1e3), cube[2])
    α₂ = quantile(Uniform(α₁, 4.0), cube[3])
    variance = quantile(LogNormal(0, 1), cube[4])
    ν = quantile(Gamma(2, 0.5), cube[5])
    μ = quantile(Normal(x̄, 5 * sqrt(va)), cube[6])
    return [α₁, f₁, α₂, variance, ν, μ]
end
paramnames = ["α₁", "f₁", "α₂", "variance", "ν", "μ"]
```

We can load `ultranest` using `PyCall`:
```julia
using PyCall
ultranest = pyimport("ultranest")
```

and start sampling the posterior:

```julia
sampler = ultranest.ReactiveNestedSampler(paramnames, logl, resume = true, transform = prior_transform, log_dir = "path/to/dir", vectorized = false)
results = sampler.run()
sampler.print_results()
sampler.plot()
```

## Citing the method

If this method or code was useful to you, you can cite [Lefkir et. al 2025](https://ui.adsabs.harvard.edu/abs/2025MNRAS.539.1775L/abstract) for method and [Foreman-Mackey, et al. 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F/abstract) for the celerite algorithm.

If you have used [`ultranest`](https://github.com/JohannesBuchner/UltraNest) or [`Turing.jl`](https://github.com/TuringLang/Turing.jl) to sample the posterior, have a look at their documentation on how to cite them properly.