using Pioran, Test

function test_sum_covariance()
    t = collect(range(0, stop=10, length=500));
    e1 = Exp(1.0,0.34)
    e2 = Exp(2.4,0.21)
    e = e1 + e2
    @test e.(t, 0.0) ≈ e1.(t, 0.0) + e2.(t, 0.0)
end

function test_scale_sum_covariance()
    t = collect(range(0, stop=10, length=500));
    e1 = Exp(1.0,0.34)
    e2 = Exp(2.4,0.21)
    e = 12.5*(e1 + e2)
    @test e.(t, 0.0) ≈ 12.5(e1.(t, 0.0) + e2.(t, 0.0))
end

function test_sum_covariance_celerite_coeff()
    e1 = Exp(1.0,0.34)
    e2 = Exp(2.4,0.21)
    e = 12.5*(e1 + e2)
    @test Pioran.celerite_coefs(e) == ([12.5,30.0],[.0,.0],[.34,.21],[.0,.0])
end

function test_large_sum_covariance()
    e1 = Exp(1.0,0.34)
    c1 = Celerite(1.3,4.2,1.3,5.2)
    e2 = Exp(2.4,0.21)
    c2 = Celerite(3.3,1.2,3.3,2.13)
    e = e1+c1+e2+c2
    @test Pioran.celerite_coefs(e) == ([1.0,1.3,2.4,3.3],[.0,4.2,.0,1.2],[.34,1.3,.21,3.3],[.0,5.2,.0,2.13])
end

@testset "Sum of ACVFs" begin
    test_sum_covariance()
    test_scale_sum_covariance()
    test_sum_covariance_celerite_coeff()
    test_large_sum_covariance()
end