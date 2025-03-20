using BenchmarkTools
using DelimitedFiles
using Pioran
using Distributions
using Statistics

const SUITE = BenchmarkGroup()

A = readdlm("benchmark/simu.txt")

t, y, yerr = collect.(eachcol(A))

f0 = 1 / (t[end] - t[1]) / 100;
fM = 1 / minimum(diff(t)) / 2 * 20;
min_f_b = f0 * 10
max_f_b = fM / 10
α₁, f₁, α₂ = 0.82, 0.01, 3.3
variance = var(y, corrected = true)

ν = 1.0
σ² = ν .* yerr .^ 2
μ = 0.0 # mean(y)

𝓟 = SimpleBendingPowerLaw(α₁, f₁, α₂)

# Approximation of the PSD to form a covariance function
𝓡 = approx(𝓟, f0, fM, 20, variance)

f = ScalableGP(μ, 𝓡)


SUITE["likelihood_solver"] = @benchmarkable Pioran.log_likelihood(𝓡, t, y .- μ, σ²)
SUITE["likelihood_gp"] = @benchmarkable logpdf(f(t, σ²), y)
SUITE["likelihood_direct"] = @benchmarkable -Pioran.log_likelihood_direct(𝓡, t, y .- μ, σ²)

tune!(SUITE)

# results = run(SUITE, verbose=true, seconds=1)
