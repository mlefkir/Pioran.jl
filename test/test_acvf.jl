using Pioran, Test, Random

function test_sum_covariance()
    t = collect(range(0, stop = 10, length = 500))
    e1 = Exp(1.0, 0.34)
    e2 = Exp(2.4, 0.21)
    e = e1 + e2
    return @test e.(t, 0.0) ≈ e1.(t, 0.0) + e2.(t, 0.0)
end

function test_scale_sum_covariance()
    t = collect(range(0, stop = 10, length = 500))
    e1 = Exp(1.0, 0.34)
    e2 = Exp(2.4, 0.21)
    e = 12.5 * (e1 + e2)
    return @test e.(t, 0.0) ≈ 12.5(e1.(t, 0.0) + e2.(t, 0.0))
end

function test_sum_covariance_celerite_coeff()
    e1 = Exp(1.0, 0.34)
    e2 = Exp(2.4, 0.21)
    e = 12.5 * (e1 + e2)
    return @test Pioran.celerite_coefs(e) == ([12.5, 30.0], [0.0, 0.0], [0.34, 0.21], [0.0, 0.0])
end

function test_large_sum_covariance()
    e1 = Exp(1.0, 0.34)
    c1 = Celerite(1.3, 4.2, 1.3, 5.2)
    e2 = Exp(2.4, 0.21)
    c2 = Celerite(3.3, 1.2, 3.3, 2.13)
    e = e1 + c1 + e2 + c2
    return @test Pioran.celerite_coefs(e) == ([1.0, 1.3, 2.4, 3.3], [0.0, 4.2, 0.0, 1.2], [0.34, 1.3, 0.21, 3.3], [0.0, 5.2, 0.0, 2.13])
end

function test_sumofCelerite()
    rng = MersenneTwister(1234)
    J = 10
    a = 2rand(rng, J)
    b = rand(rng, J)
    c = rand(rng, J)
    d = rand(rng, J)
    cov = Pioran.StructArray{Celerite}((a, b, c, d))
    C = Pioran.SumOfCelerite(cov)

    @test cov == C.cov
    @test C isa Pioran.SumOfTerms
    @test C isa Pioran.SumOfCelerite
    @test C.a == a
    @test C.b == b
    @test C.c == c
    @test C.d == d

    @test Pioran.celerite_coefs(C) == (a, b, c, d)

    D = Pioran.SumOfCelerite(a, b, c, d)
    @test D isa Pioran.SumOfTerms
    @test D isa Pioran.SumOfCelerite
    @test D.a == a
    @test D.b == b
    @test D.c == c
    @test D.d == d

    return @test D.cov == C.cov
end

function test_rescaleSumOfCelerite()
    rng = MersenneTwister(1234)
    J = 10
    a = 2rand(rng, J)
    b = rand(rng, J)
    c = rand(rng, J)
    d = rand(rng, J)

    D = Pioran.SumOfCelerite(a, b, c, d)
    u = 3.4 * D
    @test u isa Pioran.SumOfCelerite
    @test u.a == 3.4 * a
    @test u.b == 3.4 * b
    @test u.c == c
    @test u.d == d

    covu = Pioran.StructArray{Celerite}((3.4 * a, 3.4 * b, c, d))
    return @test u.cov == covu
end

function test_evalSumOfCelerite()
    rng = MersenneTwister(1234)
    J = 10
    a = 2rand(rng, J)
    b = rand(rng, J)
    c = rand(rng, J)
    d = rand(rng, J)
    τ = LinRange(-10, 10, 1000)

    D = Pioran.SumOfCelerite(a, b, c, d)
    u = 3.4 * D

    Dval = D.(τ, 0.0)
    uval = u.(τ, 0.0)

    @test Dval ≈ sum([Celerite(a[i], b[i], c[i], d[i]).(τ, 0.0) for i in 1:J])
    return @test uval ≈ sum([Celerite(3.4 * a[i], 3.4 * b[i], c[i], d[i]).(τ, 0.0) for i in 1:J])
end

@testset "Sum of ACVFs" begin
    test_sum_covariance()
    test_scale_sum_covariance()
    test_sum_covariance_celerite_coeff()
    test_large_sum_covariance()
    test_sumofCelerite()
    test_rescaleSumOfCelerite()
    test_evalSumOfCelerite()
end
