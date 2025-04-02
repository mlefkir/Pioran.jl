using Pioran, Test, Random, DelimitedFiles

function test_prediction_exp_covariance()
    A = readdlm("data/simu.txt")
    t, y, yerr = collect.(eachcol(A))


    k = Exp(1.0, 2.4)
    tp = LinRange(minimum(t), maximum(t), 1000)
    f = ScalableGP(k)(t, yerr .^ 2)
    fp = posterior(f, y)

    return @test Pioran.predict_direct(k, tp, t, y, yerr .^ 2) ≈ mean(fp, tp)
end

function test_cel_covariance()
    a, b, c, d = 3.2, 0.2, 3.0, 0.2
    k = Celerite(a, b, c, d)
    A = readdlm("data/simu.txt")
    t, y, yerr = collect.(eachcol(A))


    tp = LinRange(minimum(t), maximum(t), 1000)
    f = ScalableGP(k)(t, yerr .^ 2)
    fp = posterior(f, y)

    return @test Pioran.predict_direct(k, tp, t, y, yerr .^ 2) ≈ mean(fp, tp)
end


@testset "test_prediction_general" begin
    test_prediction_exp_covariance()
    test_cel_covariance()
end
