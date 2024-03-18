# to run with 6 workers: julia -p 6 double_pl.jl data/simu.txt
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

    fname = replace(split(filename, "/")[end], ".txt" => "_double")
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
    α ~ ThreeUniformDependent(0, 1.25, 4)
    α₁, α₂, α₃ = α
    fb ~ TwoLogUniform(min_f_b, max_f_b)
    f₁, f₂ = fb
    variance ~ LogNormal(log(0.5), 1.25)
    ν ~ Gamma(2, 0.5)
    μ ~ LogNormal(log(3), 1)
    c ~ LogUniform(1e-6, minimum(y) * 0.99)

    # Rescale the measurement variance
    σ² = ν .* σ .^ 2 ./ (y .- c) .^ 2

    # Make the flux Gaussian
    y = log.(y .- c)

    # Define power spectral density function
    𝓟 = DoubleBendingPowerLaw(α₁, f₁, α₂, f₂, α₃)

    # Approximation of the PSD to form a covariance function
    𝓡 = approx(𝓟, f0, fM, n_components, variance, basis_function=basis_function)

    # Build the GP
    f = ScalableGP(μ, 𝓡)

    # sample the conditioned distribution
    return y ~ f(t, σ²) # <- this means that our data y is distributed
    # according to f conditioned with input t and variance σ²
end

@everywhere begin
    n_adapts = 500 # number of adaptation steps
    tap = 0.65 #target acceptance probability
    sampler = externalsampler(AdvancedHMC.NUTS(tap))
end

# either 
# HMCchains = sample(GP_inference(y, t, yerr), externalsampler(nuts), MCMCDistributed(),1000,num_chains, n_adapts=n_adapts, progress=true)

# or 
HMCchains = pmap(c -> sample(inference_model(y, t, yerr), sampler, 1000; n_adapts=n_adapts,save_state=true, progress=true), 1:num_chains)
total_chainHMC = chainscat(HMCchains...)# not needed in the previous case

if !isdir("inference/")
    mkpath("inference/")
end
h5open(dir*".h5", "w") do file
    write(file, total_chainHMC)
end