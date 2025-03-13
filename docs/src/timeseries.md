
# Time series and priors

I usually store time series in a text file, this file can be read using the `readdlm` function from the [`DelimitedFiles`](https://docs.julialang.org/en/v1/stdlib/DelimitedFiles/) package.
```julia
using DelimitedFiles
data = readdlm("data.txt", comments=true) # comments=true to ignore the header
t, y, σ = data[:, 1], data[:, 2], data[:, 3]
```

!!! warning
    The time series should be sorted in ascending order of time to use the celerite algorithm.


## Log-normal distributed time series

If we assume the time series to be log-normally distributed, we can use the `log` function to make it normally distributed. In addition, we need to rescale the measurement variance to account for the transformation. We also add a constant `c` to the time series to account for a possible offset. The parameter `ν` is used to rescale the measurement variance, in case the errors are underestimated or overestimated.
The transformation is given by
```julia
    σ² = ν .* σ .^ 2 ./ (y .- c) .^ 2
    yn = log.(y .- c)
```

Then the Gaussian process can be built as shown before:
```julia
    f = ScalableGP(μ, 𝓡)
    fx = f((t, σ²), yn)
```

## Defining the priors

The prior probability distribution of the parameters of the Gaussian process can be defined using the `Distributions` package. See the
[documentation](https://juliastats.org/Distributions.jl/stable/) for more details.

```@example priors
using Distributions
using Plots
```

### Bending power-law parameters

As mentioned before the power spectrum can be modelled using a bending power-law. We do not model the amplitude of the power spectrum but its integral, the variance (see below). The indices or slopes of the power spectrum are modelled with a Uniform distribution as we restrict their range for the approximation to hold. In the case of `SHO` basis functions, the range of the indices is between `0` and `4` and `DRWCelerite` is between `0` and `6`. We can model slightly rising power spectra by giving a small negative lower bounder such as `-0.5`.
```@example priors
α₁ = Uniform(-0.5, 1.5)
α₂ = Uniform(-0.5, 4.0)
x = range(-1, 5, length=100)
plot(x, pdf.(α₁, x), label="α₁ prior", xlabel="α", ylabel="pdf(α)",framestyle = :box,lw=2)
plot!(x, pdf.(α₂, x), label="α₂ prior", xlabel="α", ylabel="pdf(α)",framestyle = :box,lw=2)
```

The bend frequencies can be modelled with a log-uniform prior distribution depending on the time values of the time series. For instance, we have:
```@example priors
T = 364.3# t[end]-t[1] # duration
Δt = 0.64#minimum(diff(t)) # minimum sampling
f_min, f_max = 1/T, 1/2/Δt
f0, fM = f_min / 5.0 , f_max * 5.0

f₁ = LogUniform(f0,fM)
x = 10 .^(range(-3,2,1000))
plot(x,pdf.(f₁,x),label="f₁ prior",xlabel="f₁",ylabel="pdf(f₁)",framestyle=:box,lw=2,xscale=:log10)
vline!([f_min,f_max],label="f_min or f_max")
```


### Scale on the errors

The parameter `ν` is used to rescale the measurement variance, in case the errors are underestimated or overestimated. We expect the find the value of `ν` to be close to 1, we model the prior distribution on `ν` as a gamma distribution with shape `2` and rate `0.5`:
```@example priors
ν = Gamma(2, 0.5)
x = range(0, 5, length=100)
println("Mean: ", mean(ν), " Variance: ", var(ν))
```

```@example priors
plot(x, pdf.(ν, x), label="ν prior", xlabel="ν", ylabel="pdf(ν)",framestyle = :box,lw=2)
```

### Mean of the time series

In a Bayesian framework defining the prior for the mean `μ` and the variance `variance` can be challenging if we have no a priori information about the time series. A solution can be to randomly sample values from the time series and use them to define the prior distributions. The function [`extract_subset`](@ref) can be used to extract a small - 3 per cent - subset of the time series and compute the mean `x̄` and variance `va` of the subset. The remaining values are returned to be used in the inference.

If we assume the time series to be log-normally distributed then the log of the subset is taken to provide an estimate of the mean and variance this is done by setting `take_log=true`. See the example below:

```julia
seed = 1234
t, y, σ, x̄, va = extract_subset(seed, t_all, y_all, σ_all; take_log=true,suffix="_seed1234")
```
The prior on the mean can be constructed using a normal distribution with the mean and variance of the subset.
```@example priors
x̄ = 0.23 # hide
va = 1.2# hide
μ = Normal(x̄, 5*sqrt(va))
x = range(-10, 10, length=1000)
plot(x, pdf.(μ, x), label="μ prior", xlabel="μ", ylabel="pdf(μ)",framestyle = :box,lw=2)
```

### Variance of the time series for active galaxies

For astronomical light curves from active galaxies, we can constrain the variance using our prior knowledge of the fractional variability amplitude $F_\mathrm{var}=\sqrt{\dfrac{s^2-\sigma^2_\mathrm{err}}{\bar{x}^2}}$ where $s^2$ is the sample variance, $\bar{x}$ and $\sigma^2_\mathrm{err}$ is mean square error.
 We know it is very unlikely to be higher than $1$ therefore we can use a log-normal distribution. As $F_\mathrm{var}^2$ is proportional to the variance, we can assume a log-normal prior for the variance.
```@example priors
# Distribution for F_var
μᵥ, σᵥ = -1.5, 1/√2
μₙ, σₙ² = 2μᵥ, 4(σᵥ)^2
variance = LogNormal(μₙ, sqrt(σₙ²))
x = range(1e-3, 10, length=10000)

plot(x, pdf.(variance, x), label="variance prior", xlabel="variance", ylabel="pdf(variance)",framestyle = :box,lw=2,xscale=:log10)
```
