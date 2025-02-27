# Getting started

As `Pioran` is written in Julia, you need to install Julia first. Please refer to [the official website](https://julialang.org/downloads/) for the installation.

## Installation

Once Julia is installed, you can install `Pioran` by running the following command in the Julia REPL:

```julia
import Pkg; Pkg.add("Pioran")
```

You may need to install other packages such as `Distributions`, `Plots`, `DelimitedFiles` to run the examples.

!!! note
    Another way to install packages in the Julia REPL is to use the `]` key to enter the package manager and then type `add MyPackage`. See below:
    ```julia
    pkg> add Distributions Plots DelimitedFiles
    ```

## Usage

First, load the package using the following command:

```@example getting_started
using Pioran
```

Assuming you have a time series `y`  with measurement error `yerr` indexed by time `t`.

```@example getting_started
using DelimitedFiles # hide
using Plots
A = readdlm("data/simu.txt",comments=true) # hide
t, y, yerr = A[:,1], A[:,2], A[:,3] # hide
σ² = yerr .^ 2  # hide
scatter(t, y,yerr=yerr, label="data",xlabel="Time (days)",ylabel="Value",legend=false,framestyle = :box,ms=3)
```

Let's assume we can model the power spectrum with a single-bending power-law model [`SingleBendingPowerLaw`](@ref).

```@example getting_started
α₁, f₁, α₂ = 0.3, 0.03, 3.2
𝓟 = SingleBendingPowerLaw(α₁, f₁, α₂)
f = 10 .^ range(-3, stop=3, length=1000)
plot(f, 𝓟.(f), label="Single Bending Power Law",xlabel="Frequency (day^-1)",ylabel="Power Spectral Density",legend=true,framestyle = :box,xscale=:log10,yscale=:log10)
```
To compute the corresponding covariance function, we approximate the power spectral density by a sum of `SHO` power spectral densities using the [`approx`](@ref) function. We need to specify the frequency range `f0` and `fM` over which the approximation is performed. The `variance` of the process can also be given. More details about approximating the power spectral density can be found in the [Approximating the power spectral density](@ref) section of [Modelling](@ref).

```@example getting_started
f0, fM = 1e-3, 1e3
variance = 12.3
𝓡 = approx(𝓟, f0, fM, 20, variance, basis_function="SHO")
τ = range(0, stop=300, length=1000)
plot(τ, 𝓡.(τ,0.), label="Covariance function",xlabel="Time lag (days)",ylabel="Autocovariance",legend=true,framestyle = :box)
```

We can now build a Gaussian process $f$ which uses the quasi-separable struct of the covariance function to speed up the computations. If the mean of the process $\mu$ is known, it can be given as an argument. Otherwise, the mean is assumed to be zero.

```@example getting_started
μ = 1.3
f = ScalableGP(μ, 𝓡)
```

We can compute the log-likelihood of the Gaussian process given data `y`, times `t` and measurement variances `σ²` using the function `logpdf` from the [`Distributions`](https://juliastats.org/Distributions.jl/stable/) package. `f(t, σ²)` is the Gaussian process where we incorporate the knowledge of measurement variance `σ²` and the time values `t`.
```@example getting_started
using Distributions
logpdf(f(t, σ²), y)
```

We can combine all these steps in a single function to compute the log-likelihood of the data given the parameters of the power spectral density and the Gaussian process.

```@example getting_started

function loglikelihood(y, t, σ, params)

    α₁, f₁, α₂, variance, μ = params

    σ² = σ .^ 2

    # Define power spectral density function
    𝓟 = SingleBendingPowerLaw(α₁, f₁, α₂)

    # Approximation of the PSD to form a covariance function
    𝓡 = approx(𝓟, f0, fM, 20, variance, basis_function="SHO")

    # Build the GP
    f = ScalableGP(μ, 𝓡)

    # sample the conditioned distribution
    return logpdf(f(t, σ²), y)
end
```
