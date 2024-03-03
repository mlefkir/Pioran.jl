
using DelimitedFiles
using Pioran, Test
using Distributions
using Statistics
using Random

A = readdlm("data/simu.txt")

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
𝓡 = fx.f.kernel
σ² = ν .* yerr .^ 2


fp = posterior(fx, y)

τ = collect(range(minimum(t), stop=maximum(t), length=1000))
τ2 = collect(range(minimum(t)-30, stop=maximum(t)+30, length=1000))
τr = sort(rand(1000))*(t[end]-t[1])*2 .+(t[1]-t[end]/2)



@testset "prediction_mean" begin
    # test on the same 
    @test Pioran.predict_direct(𝓡,t,t,y,σ²) ≈ mean(fp)
    # test on more points
    @test Pioran.predict_direct(𝓡,τ,t,y,σ²) ≈ mean(fp,τ)
    # test on more points
    @test Pioran.predict_direct(𝓡,τ2,t,y,σ²) ≈ mean(fp,τ2)
    # test on random points
    @test Pioran.predict_direct(𝓡,τr,t,y,σ²) ≈ mean(fp,τr)   
end
