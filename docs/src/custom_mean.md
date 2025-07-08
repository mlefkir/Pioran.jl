# Using custom mean function

In some cases, we want to use a mean function that is not constant over time. For instance, this can be useful when one wants to model a periodic signal embedded in red noise.

Let's say we want to model a mean function of the form:
```math
m(t ; A, T_0, \phi,\mu) = A \sin{(2\pi t / T_0 +\phi)} +\mu
```
and the broadband noise is modelled with a power-law power spectrum, typical for the variability of active galaxies.

## Implementation


```@example custommean
using Plots
using Pioran
using Random
using Distributions
```

To use a custom mean function, first define the function like:
```@example custommean
mean_function(x, A=A, ϕ=ϕ, T₀=T₀, μ=μ) = @. A * sin(2π * x / T₀ + ϕ) + μ
```
Then, create a `CustomMean` struct with the function as an argument:
```@example custommean
μ_fun = CustomMean(mean_function)
```
And use it in the [`ScalableGP`](@ref) struct like any mean value and that's it! This whole process can be summarised in a function as follows:

```@example custommean
function GP_model(t, σ, params)

    α₁, f₁, α₂, variance, ν, μ, A, ϕ, T₀ = params

    T = (t[end] - t[1]) # duration of the time series
    Δt = minimum(diff(t)) # min time separation
    f_min, f_max = 1 / T, 1 / Δt / 2

    # Rescale the measurement variance
    σ² = ν .* σ .^ 2

    # Define the mean
    mean_function(x, A=A, ϕ=ϕ, T₀=T₀, μ=μ) = @. A * sin(2π * x / T₀ + ϕ) + μ
    μ_fun = CustomMean(mean_function)

    # Define power spectral density function
    𝓟 = SingleBendingPowerLaw(α₁, f₁, α₂)

    # Approximation of the PSD to form a covariance function
    𝓡 = approx(𝓟, f_min, f_max, 20, variance, basis_function="SHO")

    # Build the GP using the mean function and (auto)covariance function
    f = ScalableGP(μ_fun, 𝓡)

    # Condition on the times and errors of the observation
    fx = f(t, σ²)
    return fx
end
```

## Sampling from the GP

We first define the GP to sample from:
```@example custommean
t = LinRange(0,2000,200)
σ = 0.5 * ones(length(t))

params = [0.3,1e-2,2.9,1.03,1.0,0.2,2.3,0.3,320]
fx = GP_model(t, σ, params)
```

We now sample a few realisations from the GP and see that the realisations are indeed periodic.
```@example custommean
rng = MersenneTwister(12)
y = [rand(rng,fx) for i in 1:3]
Plots.scatter(t,y,yerr=σ,xlabel="Time",ylabel="Value",legend=false,framestyle = :box,ms=3)
```

## Inference

To use such process for inference it is as easy as before. You need to get the loglikelihood using the `logpdf` function as follows:

```@example custommean
fx = GP_model(t, σ, params)
logpdf(fx,y[1])
```
