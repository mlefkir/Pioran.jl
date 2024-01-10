
using BenchmarkTools
using DelimitedFiles
using Pioran
using Distributions
using Statistics
using Turing
# using PkgBenchmark

const SUITE = BenchmarkGroup()

SUITE["inference"] = BenchmarkGroup([])

n_samples = [50, 100, 200, 500, 800, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
n_components = [10, 20, 25, 30, 40, 50]


# load data
A = readdlm("simulate_long.txt")
t, y, yerr = collect.(eachcol(A))

# parameter values
ν = 1.0
α₁, f₁, α₂ = 0.82, 0.01, 3.3
c = 1e-5
variance = var(y, corrected=true)
μ = mean(y)

# define the model
@model function GP_inference(y, t, σ, J)

    f0 = 1 / (t[end] - t[1]) / 100
    fM = 1 / minimum(diff(t)) / 2 * 20
    min_f_b = f0 * 10
    max_f_b = fM / 10

    # Prior distribution for the parameters
    α₁ ~ Uniform(-2.0, 0.25)
    f₁ ~ LogUniform(min_f_b, max_f_b)
    α₂ ~ Uniform(-3.9, -0.5)
    variance ~ LogNormal(log(0.5), 1.25)
    ν ~ Gamma(2, 0.5)
    μ ~ LogNormal(log(3), 1)
    c ~ LogUniform(1e-7, minimum(y))

    # Make the data Gaussian
    σ² = ν .* σ .^ 2 ./ (y .- c) .^ 2
    y = log.(y .- c)

    # Define power spectral density function
    𝓟 = SimpleBendingPowerLaw(α₁, f₁, α₂)

    # Approximation of the PSD to form a covariance function
    𝓡 = approx(𝓟, f0, fM, J, variance)

    # Build the GP
    f = ScalableGP(μ, 𝓡)

    # sample the conditioned distribution
    return y ~ f(t, σ²) # <- this means that our data y is distributed
    # according to f conditioned with input t and variance σ²
end

for J in n_components
    SUITE["inference"][string(J)] = BenchmarkGroup()
    for N in n_samples


        SUITE["inference"][string(J)][N] = @benchmarkable (loglikelihood(GP_inference(y[1:$N], t[1:$N], yerr[1:$N], $J), (α₁=α₁, f₁=f₁, α₂=α₂, variance=variance, ν=ν, μ=μ, c=c)))
    end
end
tune!(SUITE)

#results = run(SUITE, verbose=true, seconds=1)
