# Hamiltonian Monte Carlo with Turing.jl

In this section, we will use the `Turing.jl` package to model the inference problem of Gaussian process regression.
We advise the reader to check the [Turing.jl documentation](https://turinglang.org/stable/) for a more detailed explanation of the package and its capabilities.

## Modelling function

In `Turing` we can write an observation model with the `@model` as follows:
```@example turing_model
using Turing
using Pioran

@model function inference_model(y, t, σ)

    # Prior distribution for the parameters
    α₁ ~ Uniform(0., 1.25)
    f₁ ~ LogUniform(min_f_b, max_f_b)
    α₂ ~ Uniform(1, 4)
    variance ~ LogNormal(log(0.5), 1.25)
    ν ~ Gamma(2, 0.5)
    μ ~ Normal(0, 2)

    # Rescale the measurement variance
    σ² = ν .* σ .^ 2

    # Define power spectral density function
    𝓟 = SingleBendingPowerLaw(α₁, f₁, α₂)

    # Approximation of the PSD to form a covariance function
    𝓡 = approx(𝓟, f0, fM, 20, variance)

    # Build the GP
    f = ScalableGP(μ,𝓡)

    # sample the conditioned distribution
    return y ~ f(t, σ²) # <- this means that our data y is distributed according
    # to the GP f conditioned with input t and variance σ²
end
```

The order of the parameters in the `@model` block is important, first we define the prior distribution for the parameters, then we rescale the measurement variance and define the power spectral density function and its approximation to form a covariance function. Finally, we build the Gaussian process.

The last line says that the data `y` is distributed according to the Gaussian process `f` conditioned with input time `t` and measurement variance `σ²`.

### Prior distributions

In practice, if we have several slopes $\alpha_i$ and frequencies $f_i$, we would like to order them such that $\alpha_i < \alpha_{i+1}$ and $f_i< f_{i+1}$. However, in `Turing.jl` it is not yet possible to sample from distributions with dynamic support, see these issues [[1]](https://github.com/TuringLang/Turing.jl/issues/1558),[[2]](https://github.com/TuringLang/Turing.jl/issues/1708),[[3]](https://github.com/TuringLang/Turing.jl/issues/1270).

The solution proposed in these issues is to define a custom multivariate distribution with bijectors to map from the constrained space to the unconstrained space.

Here we provide three distributions:
[`TwoUniformDependent`](@ref), [`ThreeUniformDependent`](@ref) and [`TwoLogUniformDependent`](@ref).

In a double-bending power-law model they can be used as follows:
```julia
@model function inference_model(y, t, σ)

    α ~ ThreeUniformDependent(0, 1.25, 4)
    α₁, α₂, α₃ = α
    fb ~ TwoLogUniform(min_f_b, max_f_b)
    f₁, f₂ = fb
    variance ~ LogNormal(log(0.5), 1.25)
    ν ~ Gamma(2, 0.5)
    μ ~ LogNormal(log(3), 1)

    σ² = ν .* σ .^ 2
    𝓟 = DoubleBendingPowerLaw(α₁, f₁, α₂, f₂, α₃)
    𝓡 = approx(𝓟, f0, fM, 20, variance)
    f = ScalableGP(μ, 𝓡)
    return y ~ f(t, σ²)
end
```
## Sampling

Once the model is defined, we can choose a sampler. Here we use the No-U-Turn Sampler (NUTS) which is a variant of Hamiltonian Monte Carlo (HMC). In `Turing.jl` we can define the sampler with $0.8$  as target acceptance probability as follows:

```julia
sampler = NUTS(0.8)
```
or we can use the `AdvancedHMC.jl` package to define the sampler as follows:
```julia
using AdvancedHMC
tap = 0.8 #target acceptance probability
nuts = AdvancedHMC.NUTS(tap)
sampler = externalsampler(nuts)
```
For more information on the `AdvancedHMC.jl` package, see the [documentation](https://turinglang.org/AdvancedHMC.jl/stable/). This allows more control over the sampler, for instance the choice of metric.

We can then sample `2000` points from the posterior distribution using the `sample` function. We can also specify the number of adaptation steps with the `n_adapts` keyword argument. The `progress` keyword argument is used to display the progress of the sampling process.
```julia
mychain = sample(inference_model(y, t, yerr), sampler, 2000; n_adapts=1000, progress=true)
```

## Sampling several chains

In practice, we may want to sample from the posterior distribution using multiple chains which is essential for convergence diagnostics $\hat{R}$ and the ESS (effective sample size). We can use the `Distributed` package or the `MCMCDistributed()` function to sample from the posterior distribution using multiple chains. For a more detailed explanation see the Turing.jl [Guide here](https://turinglang.org/v0.30/docs/using-turing/guide/#sampling-multiple-chains).

### With Distributed.jl

We can use the [`Distributed`](https://docs.julialang.org/en/v1/stdlib/Distributed/) package to sample from the posterior distribution using multiple chains as follows:
```julia
using Distributed
using Turing

num_chains = nworkers();
@everywhere filename = $(ARGS[1]);

@everywhere begin
    using Turing
    using MCMCChains
    using AdvancedHMC
    using Pioran
    using DelimitedFiles

    data = readdlm(filename, comments=true)
    t, y, yerr = ...
    # do something
end

@everywhere @model function ...
# Define the model here
end

@everywhere begin
    sampler = ...
end

HMCchains = pmap(c -> sample(inference_model(y, t, yerr), sampler, 2000; n_adapts=1000, progress=true), 1:num_chains)
total_chainHMC = chainscat(HMCchains...) # concatenate the chains
```

This script is run with the following command:
```bash
julia -p 6 script.jl data.txt
```
Where `6` is the number of chains and `data.txt` is the file containing the time series.

### With MCMCDistributed()

When using the `MCMCDistributed()` function, the script is essentially the same, only the last two lines are replaced by:

```julia
total_chainHMC = sample(GP_inference(y, t, yerr), sampler, MCMCDistributed(),2000,num_chains, n_adapts=n_adapts, progress=true)
```

## Saving the chains

As mentioned in the [`documentation`](https://turinglang.org/MCMCChains.jl/stable/getting-started/#Saving-and-Loading-Chains) of the `MCMCChains` package, we can save the chains using [`MCMCChainsStorage`](https://github.com/farr/MCMCChainsStorage.jl/tree/main) as follows:

```julia
using HDF5
using MCMCChains
using MCMCChainsStorage

total_chainHMC = sample(GP_inference(y, t, yerr), sampler, MCMCDistributed(),2000,num_chains, n_adapts=n_adapts, progress=true)

h5open("total_chain.h5", "w") do file
    write(file, total_chainHMC)
end
```

## Example

Here is an example of a script which can be found in the example directory of the `Pioran.jl` package. This script is used to sample from the posterior distribution using multiple chains.

```julia
# to run with 6 workers: julia -p 6 single_pl.jl data/simu.txt
using Distributed
using Turing
using HDF5
using MCMCChains
using MCMCChainsStorage

num_chains = nworkers();
@everywhere filename = $(ARGS[1]);

@everywhere begin
    using Turing
    using MCMCChains
    using AdvancedHMC
    using Pioran
    using DelimitedFiles

    fname = replace(split(filename, "/")[end], ".txt" => "_single")
    dir = "inference/" * fname
    data = readdlm(filename, comments=true)
    t, y, yerr = data[:, 1], data[:, 2], data[:, 3]

    # Frequency range for the approx and the prior
    f_min, f_max = 1 / (t[end] - t[1]), 1 / minimum(diff(t)) / 2
    f0, fM = f_min / 20.0, f_max * 20.0
    min_f_b, max_f_b = f0 * 4.0, fM / 4.0

    # F var^2 is distributed as a log-normal
    μᵥ, σᵥ = -1.5, 1.0
    μₙ, σₙ² = 2μᵥ, 2(σᵥ)^2
    σₙ = sqrt(σₙ²)

    # options for the approximation
    basis_function = "SHO"
    n_components = 20
    model = SingleBendingPowerLaw
    prior_checks = true
end

@everywhere @model function inference_model(y, t, σ)

    # Prior distribution for the parameters
    α₁ ~ Uniform(0.0, 1.25)
    f₁ ~ LogUniform(min_f_b, max_f_b)
    α₂ ~ Uniform(1, 4)
    variance ~ LogNormal(log(0.5), 1.25)
    ν ~ Gamma(2, 0.5)
    μ ~ Normal(0, 2)
    c ~ LogUniform(1e-6, minimum(y) * 0.99)

    # Rescale the measurement variance
    σ² = ν .* σ .^ 2 ./ (y .- c) .^ 2

    # Make the flux Gaussian
    y = log.(y .- c)

    # Define power spectral density function
    𝓟 = model(α₁, f₁, α₂)

    # Approximation of the PSD to form a covariance function
    𝓡 = approx(𝓟, f0, fM, n_components, variance, basis_function=basis_function)

    # Build the GP
    f = ScalableGP(μ, 𝓡)

    # sample the conditioned distribution
    return y ~ f(t, σ²) # <- this means that our data y is distributed according
    # to the GP f conditioned with input t and variance σ²
end

@everywhere begin
    n_adapts = 500 # number of adaptation steps
    tap = 0.8 #target acceptance probability
    nuts = AdvancedHMC.NUTS(tap)
end

# either
# HMCchains = sample(GP_inference(y, t, yerr), externalsampler(nuts), MCMCDistributed(),1000,num_chains, n_adapts=n_adapts, progress=true)
# or
HMCchains = pmap(c -> sample(inference_model(y, t, yerr), externalsampler(nuts), 1000; n_adapts=n_adapts,save_state=true, progress=true), 1:num_chains)
total_chainHMC = chainscat(HMCchains...)# not needed in the previous case

if !isdir("inference/")
    mkpath("inference/")
end
h5open(dir*".h5", "w") do file
    write(file, total_chainHMC)
end
```
The results of the sampling can be found in the `inference` directory. The chains are saved in the `h5` format and can be loaded as shown below:
```@example turing_model
using MCMCChains
using MCMCChainsStorage
using HDF5

total_chain = h5open("data/subset_simu_single.h5", "r") do f
  read(f, Chains)
end
total_chain
```
We can then use the `StatsPlots` package to visualize the chains. The first 50 points are discarded.
```@example turing_model
using StatsPlots
plot(total_chain[50:end,:,:])
```