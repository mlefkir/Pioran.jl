using Pioran, Test
using Distributions
using LinearAlgebra
using Random


function test_scalableGP_Exp_init()

    𝓡 = Exp(1.4, 0.5)
    μ = 1.2

    f = ScalableGP(𝓡) # zero-mean GP
    fm = ScalableGP(μ, 𝓡)
    @testset "Scalable GP initialisation" begin
        @test f isa Pioran.ScalableGP
        @test f.kernel isa Pioran.SemiSeparable
        @test fm isa Pioran.ScalableGP
        @test fm.kernel isa Pioran.SemiSeparable

    end
    return @testset "Scalable GP mean value" begin
        @test f.f.mean.c == 0.0
        @test fm.f.mean.c == μ
    end
end

function test_scalableGP_carma_init()
    rα = [
        -0.042163209825323775 + 1.1115603157767922im,
        -0.042163209825323775 - 1.1115603157767922im,
        -0.7599101571312047 + 0.0im,
    ]
    β = [
        3.9413022090550216,
        11.38193903188344,
        1,
    ]
    𝓒 = CARMA(3, 2, rα, β, 1.3)
    μ = 1.2

    f = ScalableGP(𝓒) # zero-mean GP
    fm = ScalableGP(μ, 𝓒)
    @testset "Scalable GP initialisation" begin
        @test f isa Pioran.ScalableGP
        @test f.kernel isa Pioran.CARMA
        @test fm isa Pioran.ScalableGP
        @test fm.kernel isa Pioran.CARMA

    end
    return @testset "Scalable GP mean value" begin
        @test f.f.mean.c == 0.0
        @test fm.f.mean.c == μ
    end
end

function test_scalableGP_carma_likelihood()
    function get_quad(rng, p, q, a_min = -4, a_max = 4, b_min = -4, b_max = 4)
        log_a = rand(rng, Uniform(a_min, a_max), p)
        log_b = rand(rng, Uniform(b_min, b_max), q)
        return exp.(log_a), exp.(log_b)
    end
    p, q = 3, 2
    t = [0.0, 3.0, 3.2, 3.4, 45.5, 101.2]
    y = [1.3, 2.2, 4.21, 2.5, 3.3, 5.2]
    yerr = [0.1, 0.2, 0.1, 0.1, 0.2, 0.1]
    seeds = [567, 123, 890, 456, 321]
    variances = [1.32, 35.3, 242.2, 46.6, 0.3]
    μ_set = [1.2, 0.3, 0.1, 0.46, 0.1]

    for (k, s) in enumerate(seeds)
        rng = MersenneTwister(s)
        a, b = get_quad(rng, p, q)
        rα = quad2roots(a)
        β = roots2coeffs(quad2roots(b))
        𝓒 = CARMA(p, q, rα, β, variances[k])
        f = ScalableGP(μ_set[k], 𝓒)
        @test isfinite(logpdf(f(t, yerr .^ 2), y))
        @test logpdf(f(t, yerr .^ 2), y) ≈ -Pioran.log_likelihood_direct(𝓒, t, y .- μ_set[k], yerr .^ 2)
    end
    return
end


function test_scalableGP_init()
    α₁, f₁, α₂ = 0.2, 0.02, 3.1
    𝓟 = SingleBendingPowerLaw(α₁, f₁, α₂)
    va = 2.31
    𝓡 = approx(𝓟, 1.0e-4, 1.0e1, 30, va, basis_function = "SHO")
    μ = 1.2

    f = ScalableGP(𝓡) # zero-mean GP
    fm = ScalableGP(μ, 𝓡)
    @testset "Scalable GP initialisation" begin
        @test f isa Pioran.ScalableGP
        @test f.kernel isa Pioran.SumOfSemiSeparable
        @test fm isa Pioran.ScalableGP
        @test fm.kernel isa Pioran.SumOfSemiSeparable

    end
    return @testset "Scalable GP mean value" begin
        @test f.f.mean.c == 0.0
        @test fm.f.mean.c == μ
    end
end

function test_scalableGP_likelihood()
    t = [0.0, 3.0, 3.2, 3.4, 45.5, 101.2]
    y = [1.3, 2.2, 4.21, 2.5, 3.3, 5.2]
    yerr = [0.1, 0.2, 0.1, 0.1, 0.2, 0.1]

    α₁_set = [0.2, 0.03, 0.1, 0.46, 0.1, 0.21, 0.74, 0.1, 0.03, 0.92]
    f₁_set = [1.3e-2, 1.32e-1, 5.53e-2, 3.3, 0.342, 3.2e1, 1.3, 4.0e1, 1.0e-2, 0.5]
    α₂_set = [3.2, 3.1, 2.3, 2.57, 3.6, 2.3, 2.1, 2.79, 3.3, 3.8]
    variances = [1.32, 35.3, 242.2, 46.6, 0.3, 0.244, 9.64, 0.75, 0.193, 0.21]
    μ_set = [1.2, 0.3, 0.1, 0.46, 0.1, 0.21, 0.74, 0.1, 0.03, 0.92]
    for i in 1:10
        @testset "Check GP likelihood for α₁ = $(α₁_set[i]), f₁ = $(f₁_set[i]), α₂ = $(α₂_set[i])" begin
            α₁, f₁, α₂ = α₁_set[i], f₁_set[i], α₂_set[i]
            va, μ = variances[i], μ_set[i]

            𝓟 = SingleBendingPowerLaw(α₁, f₁, α₂)
            𝓡 = approx(𝓟, 1.0e-4, 1.0e1, 30, va, basis_function = "SHO")
            f = ScalableGP(μ, 𝓡)
            @test isfinite(logpdf(f(t, yerr .^ 2), y))
            @test logpdf(f(t, yerr .^ 2), y) ≈ -Pioran.log_likelihood_direct(f.kernel, t, y .- μ, yerr .^ 2)
        end
    end
    return
end


function test_scalableGP_posterior()
    t = [0.0, 3.0, 3.2, 3.4, 45.5, 101.2]
    tx = [0.0, 1.4, 2.3, 3.0, 3.1, 3.2, 3.3, 3.4, 45.5, 101.2, 202.32]
    y = [1.3, 2.2, 4.21, 2.5, 3.3, 5.2]
    yerr = [0.1, 0.2, 0.1, 0.1, 0.2, 0.1]
    α₁, f₁, α₂ = 0.2, 0.02, 3.1
    𝓟 = SingleBendingPowerLaw(α₁, f₁, α₂)
    va = 2.31
    𝓡 = approx(𝓟, 1.0e-4, 1.0e1, 30, va, basis_function = "SHO")
    μ = 1.2

    f = ScalableGP(μ, 𝓡)
    fx = f(t, yerr .^ 2)
    fp = posterior(fx, y)
    @testset "Scalable GP posterior initialisation" begin
        @test fp isa Pioran.PosteriorGP
    end
    @testset "Scalable GP check mean" begin
        m = mean(fp)
        mx = mean(fp, tx)
        @test isfinite(m)
        @test m ≈ Pioran.predict_direct(fp.f.f.kernel, fp.f.x, fp.f.x, fp.y, yerr .^ 2)
        @test mx ≈ Pioran.predict_direct(fp.f.f.kernel, tx, fp.f.x, fp.y, yerr .^ 2)
    end
    @testset "Scalable GP check cov" begin
        c = cov(fp)
        cx = cov(fp, tx)
        @test isfinite(c)
        @test isposdef(c)
        @test c ≈ Pioran.predict_cov(fp.f.f.kernel, fp.f.x, fp.f.x, yerr .^ 2)
        @test isposdef(cx)
        @test cx ≈ Pioran.predict_cov(fp.f.f.kernel, tx, fp.f.x, yerr .^ 2)
    end
    return @testset "Scalable GP check std" begin
        s = std(fp)
        sx = std(fp, tx)
        @test isfinite(s)
        @test s ≈ sqrt.(diag(Pioran.predict_cov(fp.f.f.kernel, fp.f.x, fp.f.x, yerr .^ 2)))
        @test isfinite(sx)
        @test sx ≈ sqrt.(diag(Pioran.predict_cov(fp.f.f.kernel, tx, fp.f.x, yerr .^ 2)))
    end
end


function test_scalableGP_posterior_sample()
    Random.seed!(1234)

    t = [0.0, 3.0, 3.2, 3.4, 45.5, 101.2]
    tx = [0.0, 1.4, 2.3, 3.0, 3.1, 3.2, 3.3, 3.4, 45.5, 101.2, 202.32]
    y = [1.3, 2.2, 4.21, 2.5, 3.3, 5.2]
    yerr = [0.1, 0.2, 0.1, 0.1, 0.2, 0.1]
    α₁, f₁, α₂ = 0.2, 0.02, 3.1
    𝓟 = SingleBendingPowerLaw(α₁, f₁, α₂)
    va = 2.31
    𝓡 = approx(𝓟, 1.0e-4, 1.0e1, 30, va, basis_function = "SHO")
    μ = 1.2

    f = ScalableGP(μ, 𝓡)
    fx = f(t, yerr .^ 2)
    fp = posterior(fx, y)

    s = rand(fp)
    s10 = rand(fp, 10)
    sx = rand(fp, tx)
    sx10 = rand(fp, tx, 10)
    return @testset "Rand posterior GP" begin
        @test isfinite(s)
        @test isfinite(s10)
        @test size(s10) == (length(t), 10)
        @test size(sx) == (length(tx), 1)
        @test size(sx10) == (length(tx), 10)
    end
end

function test_scalableGP_sample()
    Random.seed!(1234)

    t = [0.0, 3.0, 3.2, 3.4, 45.5, 101.2]
    tx = [0.0, 1.4, 2.3, 3.0, 3.1, 3.2, 3.3, 3.4, 45.5, 101.2, 202.32]
    y = [1.3, 2.2, 4.21, 2.5, 3.3, 5.2]
    yerr = [0.1, 0.2, 0.1, 0.1, 0.2, 0.1]
    α₁, f₁, α₂ = 0.2, 0.02, 3.1
    𝓟 = SingleBendingPowerLaw(α₁, f₁, α₂)
    va = 2.31
    𝓡 = approx(𝓟, 1.0e-4, 1.0e1, 30, va, basis_function = "SHO")
    μ = 1.2

    f = ScalableGP(μ, 𝓡)
    fx = f(t, yerr .^ 2)

    s = rand(fx)
    sx = rand(fx, tx)
    return @testset "Rand posterior GP" begin
        @test isfinite(s)
        @test isfinite(sx)
    end

end

@testset "Scalable GP" begin
    test_scalableGP_init()
    test_scalableGP_Exp_init()
    test_scalableGP_likelihood()
    test_scalableGP_posterior()
    test_scalableGP_posterior_sample()
    test_scalableGP_sample()
end

@testset "Scalable GP CARMA" begin
    test_scalableGP_carma_init()
    test_scalableGP_carma_likelihood()
end
