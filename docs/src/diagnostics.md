# Diagnostics post-inference

After performing inference, we can use the Gaussian process to diagnose the quality of the fit. This can be done in the time domain or the frequency domain.

In this section, we will work with the following time series `y` with measurement error `yerr` indexed by time `t`.

```@example diagnostics
using DelimitedFiles
using Plots
using CairoMakie
using MCMCChains
using PairPlots
using Pioran
using Random

data = readdlm("data/subset_simu_single_subset_time_series.txt",comments=true)
t, y, yerr = data[:,1], data[:,2], data[:,3]
Plots.scatter(t, y,yerr=yerr, label="data",xlabel="Time (days)",ylabel="Value",legend=false,framestyle = :box,ms=3)
```

Let's take posterior samples from a run of the inference with `ultranest` and plot the pairplot of the samples.

```@example diagnostics
data = readdlm("data/inference/chains/equal_weighted_post.txt", comments=true)
samples = convert(Array{Float64},reshape(Array(data[2:end,:]),(size(data[2:end,:])...,1)))
c = Chains(samples,data[1,:])
pairplot(c)
```

We can now use the function [`run_posterior_predict_checks`](@ref) to perform the diagnostics. This function calls several other functions to plot the graphical diagnostics.

```@example diagnostics
f0, fM = 1 / (t[end] - t[1])/4.0, 1 / minimum(diff(t)) / 2 * 4.0
basis_function = "SHO"
n_components = 20
model = SingleBendingPowerLaw
paramnames = data[1,:]
array = data[2:end,:]
figs = run_posterior_predict_checks(array, paramnames, t, y, yerr, f0, fM, model, true; path="", basis_function=basis_function, n_components=n_components)
```

## In the Fourier domain

We can plot the posterior predictive distribution of the power spectral density and the approximate power spectral density.
We can verify if the approximation is valid with the posterior samples. The noise level is given by $2 \nu \sigma^2_{\rm yerr}\Delta t$.

This figure is plotted using the function [`Pioran.plot_psd_ppc`](@ref).
```@example diagnostics
figs[1]
```
A second diagnostic in the frequency domain is the posterior predictive periodograms using the Lomb-Scargle periodogram of simulated data. Using the posterior samples, we can draw realisations of the process at the same instants `t` and compute the Lomb-Scargle periodogram for each realisation. To do so, we use the package [`LombScargle.jl`](https://github.com/JuliaAstro/LombScargle.jl). We can then compare the median of the periodogram with the periodogram of the data. This figure is plotted using the function [`Pioran.plot_lsp_ppc`](@ref).

```@example diagnostics
figs[2]
```

## In the time domain

In the time domain, we can draw realisations of the process conditioned on the observations and compare it with the data with the function [`Pioran.plot_ppc_timeseries`](@ref). The predictive time series is plotted with the function [`Pioran.plot_simu_ppc_timeseries`](@ref).

```@example diagnostics
figs[3]
```

!!! note
    As mentioned in [Sampling from the conditioned distribution](@ref) it can be very expensive to sample from the conditioned distribution, especially if the number of points is large.

Finally, we can compute the residuals of the realisations and the data. This figure is plotted using the function [`Pioran.plot_residuals_diagnostics`](@ref). The distribution of the residuals is also plotted, the lower panel shows the autocorrelation of the residuals. One should note that the lags of the autocorrelation are not the same as the lags of the time series. The lags are in indexes of the residuals, therefore it may be difficult to interpret the autocorrelation in terms of time and realisation of a white noise process.

```@example diagnostics
figs[4]
```