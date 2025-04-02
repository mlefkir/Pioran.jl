# Simulations
In addition to inference, the Gaussian process modelling framework allows simulations and predictions of the underlying process. This is done by conditioning the Gaussian process on some observations and then sampling from the conditioned distribution.

## Sampling from the Gaussian process

Assuming a [`SingleBendingPowerLaw`](@ref) we can draw realisations from the Gaussian process with a mean.
First, we define the power spectral density function and plot it.
```@example drawing_samples
using Random
using Plots
using Pioran

rng = MersenneTwister(1234)
# Define power spectral density function
𝓟 = SingleBendingPowerLaw(0.3,1e-2,2.9)
f_min, f_max = 1e-3, 1e3
f0,fM = f_min/20.,f_max*20.
f = 10 .^ range(log10(f0), log10(fM), length=1000)
plot(f, 𝓟(f), xlabel="Frequency (day^-1)",ylabel="Power Spectral Density",legend=false,framestyle = :box,xscale=:log10,yscale=:log10,lw=2)
```
Then we approximate it to form a covariance function. We then build the Gaussian process and draw realisations from it using [`rand`](@ref).
```@example drawing_samples

variance = 15.2
# Approximation of the PSD to form a covariance function
𝓡 = approx(𝓟, f_min, f_max, 20, variance, basis_function="SHO")
μ = 1.3
# Build the GP
f = ScalableGP(μ, 𝓡) # Define the GP

T = 450 # days
t = range(0, stop=T, length=1000)
σ² = ones(length(t))*0.25

fx = f(t, σ²) # Gaussian process
realisations = [rand(rng,fx) for _ in 1:5]
plot(t, realisations, label="Realisations", xlabel="Time [days]", framestyle = :box,ylabel="Value")
```

!!! info "Note"
    Sampling from a Gaussian process built with semi-separable covariance functions is very efficient. The time complexity is O(N) where N is the number of data points.

## Conditioning the Gaussian process

We can compute the conditioned or posterior distribution of the Gaussian process given some observations. Let's use a subset of the realisations to condition the Gaussian process and then sample from the conditioned distribution.

```@example drawing_samples
using StatsBase
idx = sort(sample(1:length(t), 50, replace = false));
t_obs = t[idx]
y_obs = realisations[1][idx]
yerr = 0.25*ones(length(t_obs))
fx = f(t_obs, yerr.^2) # Gaussian process
```

We can compute the posterior distribution of the Gaussian process given the observations.
```@example drawing_samples
fp = posterior(fx, y_obs) # Posterior distribution
```
The mean and standard deviation of this distribution, can be computed using the [`mean`](@ref) and [`std`](@ref) functions. The posterior covariance matrix can be computed using the [`cov`](@ref) function.

```@example drawing_samples
m = mean(fp,t);
s = std(fp,t);
```

We can plot the realisations, the observations and the posterior distribution.
```@example drawing_samples
plot(t, realisations[1], label="Realisation", xlabel="Time [days]", framestyle = :box,ylabel="Value")
plot!(t_obs, y_obs,yerr=yerr, label="Observations", seriestype=:scatter)
plot!(t,m, ribbon=s,label="Posterior distribution", lw=2)
plot!(t,m,ribbon=2*s, label=nothing)
```

!!! info "Note"
    The computation of the mean of the distribution is very efficient. The time complexity is O(N) where N is the number of data points. However, the computation of the covariance is very inefficient as the posterior covariance is not semi-separable.

### Sampling from the conditioned distribution

We can draw realisations from the conditioned distribution using [`rand`](@ref).
```@example drawing_samples
samples_cond = rand(rng,fp,t,5);
```
We can plot the realisations, the observations and the posterior distribution.
```@example drawing_samples
plot(t_obs, y_obs,yerr=yerr, label="Observations", seriestype=:scatter, xlabel="Time [days]", framestyle = :box,ylabel="Value",color=:black,lw=2)
plot!(t, samples_cond, label=nothing)
```

!!! info "Note"
    Sampling from the conditioned Gaussian process is very inefficient as the posterior distribution is not semi-separable.