using Pioran, Random, Test, Distributions


function fun(x, A = 1.0, ϕ = 0.0, T₀ = 1.0, μ = 0.0)
    return @. A * sin(2π * x / T₀ + ϕ) + μ
end

function init_periodic_GP(
        α₁ = 0.4, f₁ = 1.0e-2, α₂ = 3.1, T₀ = 53.4,
        A = 1.3,
        μ = 0.84,
        ϕ = 0,
        variance = 0.3
    )


    f_min, f_max = 1.0e-3, 1.0e3
    # define the mean function
    m(x, A = A, ϕ = ϕ, T₀ = T₀, μ = μ) = fun(x, A, ϕ, T₀, μ)
    x̄ = CustomMean(m)

    # define PSD
    𝓟 = SingleBendingPowerLaw(0.4, 1.0e-2, 3.1)

    𝓡 = approx(𝓟, f_min, f_max, 20, variance, basis_function = "SHO")

    fp = ScalableGP(x̄, 𝓡)
    return fp
end

function sample_GP(
        rng, t, σ²; α₁ = 0.4, f₁ = 1.0e-2, α₂ = 3.1, T₀ = 53.4,
        A = 1.3,
        μ = 0.84,
        ϕ = 0,
        variance = 0.3
    )

    # rng = MersenneTwister(123)

    fp = init_periodic_GP(
        α₁, f₁, α₂, T₀,
        A,
        μ,
        ϕ,
        variance
    )
    fx = fp(t, σ²)
    y = rand(rng, fx)
    return y
end

""" Test the sampling of the GP with a CustomMean"""
function test_sample_mean()
    rng = MersenneTwister(12)

    t = LinRange(0, 1000, 100)
    σ² = zeros(length(t))
    y = sample_GP(rng, t, σ²)
    return @test all(isfinite.(y))
end

"""Test the likelihood calculation"""
function test_likelihood_mean()

    t = LinRange(0, 1000, 100)
    σ² = zeros(length(t))

    fp = init_periodic_GP()
    fx = fp(t, σ²)

    y = randn(length(t))
    return @test isfinite(logpdf(fx, y))
end

@testset "custom_mean" begin
    test_likelihood_mean()
    test_sample_mean()
end
