using Pioran, Test, QuadGK

function test_exp_covariance()
    α = 2.4
    e = Exp(1.0, α)
    e(0.0, 0.0) ≈ 1.0
    t = collect(range(0, stop = 10, length = 500))
    return @test e.(t, 0.0) ≈ exp.(-t * α)
end

function test_cel_covariance()
    a, b, c, d = 1.3, 4.0, 0.5, 3.2
    cel = Celerite(a, b, c, d)
    t = collect(range(0, stop = 25, length = 500))
    return @test cel.(t, 0.0) ≈ exp.(-c .* t) .* (a * cos.(d .* t) .+ b * sin.(d .* t))
end

function test_cel_celeritecoefs()
    a, b, c, d = 1.3, 4.0, 0.5, 3.2
    cel = Celerite(a, b, c, d)
    return @test Pioran.celerite_coefs(cel) == [a, b, c, d]
end

function test_SHO_covariance()
    A, ω₀, Q = 1.5, 2π * 0.23, 1 / √2
    s = SHO(A, ω₀, Q)
    t = collect(range(0, stop = 15, length = 500))
    η = √(abs(1 - 1 / (4 * Q^2)))
    return @test s.(t, 0.0) ≈ A * exp.(-ω₀ .* t / 2 / Q) .* (cos.(η * ω₀ * t) .+ sin.(η * ω₀ * t) / (2η * Q))
end

function test_celerite_coef()
    A, ω₀, Q = 1.5, 2π * 0.23, 1 / 2
    s = SHO(A, ω₀, Q)
    return @test_throws "SHO with Q≠1/√2 not implemented yet" Pioran.celerite_coefs(s)
end

function test_SHO_celerite_coef()
    A, ω₀, Q = 1.5, 2π * 0.23, 1 / √2
    s = SHO(A, ω₀, Q)
    return @test Pioran.celerite_coefs(s) == [A, A, √2 / 2 * ω₀, √2 / 2 * ω₀]
end

function test_Exp_celerite_coef()
    e = Exp(2.3, 0.2)
    return @test Pioran.celerite_coefs(e) == [2.3, 0.0, 0.2, 0.0]
end

function test_integral_celerite()
    a, b, c, d = [3.3, 0.2, 0.3, 2.2]
    x1, x2 = 1.0e-2, 1.0e1
    integ_num = quadgk(x -> Pioran.Celerite_psd(x, a, b, c, d), x1, x2, rtol = 1.0e-10)[1] * 2π
    integ_ana = Pioran.integral_celerite(a, b, c, d, 2π * x2) - Pioran.integral_celerite(a, b, c, d, 2π * x1)
    return @test integ_num ≈ integ_ana

end

@testset "test_covariance_functions" begin
    test_exp_covariance()
    test_cel_covariance()
    test_SHO_covariance()
    test_cel_celeritecoefs()
    test_celerite_coef()
    test_SHO_celerite_coef()
    test_Exp_celerite_coef()
    test_integral_celerite()
end
