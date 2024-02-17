
using DelimitedFiles
using Pioran, Test
using Distributions
using Statistics
using ForwardDiff

A = readdlm("data/simu_log.txt")

t, y, yerr = collect.(eachcol(A))

f0 = 1 / (t[end] - t[1]) / 100;
fM = 1 / minimum(diff(t)) / 2 * 20;
min_f_b = f0 * 10
max_f_b = fM / 10
α₁, f₁, α₂ = 0.82, 0.01, 3.3
ν = 1.0
μ = 0.0# mean(y)
variance = var(y, corrected=true)


function modelling(pars, t, y, yerr)

    α₁, f₁, α₂, variance, ν, μ = pars
    σ² = ν .* yerr .^ 2


    𝓟 = SingleBendingPowerLaw(α₁, f₁, α₂)

    # Approximation of the PSD to form a covariance function
    𝓡 = approx(𝓟, f0, fM, 20, variance)

    f = ScalableGP(μ, 𝓡)
    return f(t, σ²)
end

p = [α₁, f₁, α₂, variance, ν, μ]
fx = modelling(p, t, y, yerr)
celerite_gp_like = logpdf(fx, y)


function loglike(p)
    fx = modelling(p, t, y, yerr)
    return logpdf(fx, y)
end

σ² = ν .* yerr .^ 2


𝓟 = SingleBendingPowerLaw(α₁, f₁, α₂)

# Approximation of the PSD to form a covariance function
𝓡 = approx(𝓟, f0, fM, 20, variance)

celerite_like = Pioran.log_likelihood(𝓡, t, y .- μ, σ²)
direct_like = -Pioran.log_likelihood_direct(𝓡, t, y .- μ, σ²)
grad = ForwardDiff.gradient(loglike, p)

@testset "likelihood_solv" begin
    @test celerite_like ≈ celerite_gp_like
    @test celerite_gp_like ≈ direct_like
    @test all(isfinite.(grad))
end
