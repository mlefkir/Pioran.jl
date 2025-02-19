using BenchmarkTools
using DelimitedFiles
using Pioran
using Distributions
using Random
using Turing

SUITE = BenchmarkGroup()
SUITE["celerite_likelihood"] = BenchmarkGroup([])
SUITE["pioran_likelihood"] = BenchmarkGroup([])

# groups for the basis functions
SUITE["pioran_likelihood"]["SHO"] = BenchmarkGroup([])
SUITE["pioran_likelihood"]["DRWCelerite"] = BenchmarkGroup([])

n_samples = 2 .^ (5:16)
n_components = 2 .^ (1:6)
n_bases = [10, 20, 30, 40, 50]

basis_functions = ["SHO", "DRWCelerite"]

# load data
A = readdlm("benchmark/simulate_long.txt")
t, y, yerr = collect.(eachcol(A))
σ² = yerr .^ 2

rng = MersenneTwister(1234)

# celerite_likelihood
function loglikelihood(a, b, c, d, t, y, σ²)
    return Pioran.logl(a, b, c, d, t, y, σ²)
end


# parameter values
ν = 1.0
α₁, f₁, α₂ = 0.82, 0.01, 3.3
c = 1.0e-5
variance = var(y, corrected = true)
μ = mean(y)

# define the model
@model function model_GP(y::AbstractVector, t::AbstractVector, σ::AbstractVector, J::Int64; basis_function::String = "SHO")

    f0 = 1 / (t[end] - t[1]) / 100
    fM = 1 / minimum(diff(t)) / 2 * 20
    min_f_b = f0 * 10
    max_f_b = fM / 10

    # Prior distribution for the parameters
    α₁ ~ Uniform(-0.25, 2.0)
    f₁ ~ LogUniform(min_f_b, max_f_b)
    α₂ ~ Uniform(1.5, 4)
    variance ~ LogNormal(log(0.5), 1.25)
    ν ~ Gamma(2, 0.5)
    μ ~ LogNormal(log(3), 1)
    c ~ LogUniform(1.0e-7, minimum(y))

    # Make the data Gaussian
    σ² = ν .* σ .^ 2 ./ (y .- c) .^ 2
    y = log.(y .- c)

    # Define power spectral density function
    𝓟 = SingleBendingPowerLaw(α₁, f₁, α₂)

    # Approximation of the PSD to form a covariance function
    𝓡 = approx(𝓟, f0, fM, J, variance, basis_function = basis_function)

    # Build the GP
    f = ScalableGP(μ, 𝓡)

    # sample the conditioned distribution
    return y ~ f(t, σ²) # <- this means that our data y is distributed
    # according to f conditioned with input t and variance σ²
end


a, b, c, d = collect.(eachcol(rand(rng, Float64, (maximum(n_components), 4))))

a .*= 5
for N in n_samples
    for j in n_components
        SUITE["celerite_likelihood"][string(j)] = BenchmarkGroup()
        SUITE["pioran_likelihood"][string(j)] = BenchmarkGroup()


        SUITE["celerite_likelihood"][string(j)][N] = @benchmarkable (
            loglikelihood(
                $a[1:$j],
                $b[1:$j],
                $c[1:$j],
                $d[1:$j],
                t[1:$N],
                y[1:$N],
                yerr[1:$N]
            )
        )
    end
    for J in n_bases
        SUITE["pioran_likelihood"]["SHO"][string(J)][N] = @benchmarkable (
            Turing.loglikelihood(
                model_GP(y[1:$N], t[1:$N], yerr[1:$N], $J, basis_function = "SHO"),
                (α₁ = $α₁, f₁ = $f₁, α₂ = $α₂, variance = $variance, ν = $ν, μ = $μ, c = $c)
            )
        )
        SUITE["pioran_likelihood"]["DRWCelerite"][string(J)][N] = @benchmarkable (
            Turing.loglikelihood(
                model_GP(y[1:$N], t[1:$N], yerr[1:$N], $J, basis_function = "DRWCelerite"),
                (α₁ = $α₁, f₁ = $f₁, α₂ = $α₂, variance = $variance, ν = $ν, μ = $μ, c = $c)
            )
        )
    end
end

#tune!(SUITE)
