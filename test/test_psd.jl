using Pioran, Test

function test_SingleBendingPowerLaw()
    PS = SingleBendingPowerLaw(0.3, 0.02, 2.93)
    f = 10 .^ range(-3, stop=2, length=1000)
    @test PS(f) == (f / 0.02) .^ (-0.3) ./ (1 .+ (f / 0.02) .^ (2.93 - 0.3))
end

function test_DoubleBendingPowerLaw()
    DS = DoubleBendingPowerLaw(0.3, 0.02, 1.4, 10.2, 2.93)
    f = 10 .^ range(-3, stop=3, length=1000)
    @test DS(f) == (f / 0.02) .^ (-0.3) ./ (1 .+ (f / 0.02) .^ (1.4 - 0.3)) ./ (1 .+ (f / 10.2) .^ (2.93 - 1.4))
end

function test_normalised_psd()
    PS = SingleBendingPowerLaw(0.3, 0.02, 2.93)
    f = 10 .^ range(-3, stop=2, length=1000)
    @test PS(f) / PS(f[1]) == Pioran.get_normalised_psd(PS, f)

    DS = DoubleBendingPowerLaw(0.3, 0.02, 1.4, 10.2, 2.93)
    @test DS(f) / DS(f[1]) == Pioran.get_normalised_psd(DS, f)
end

function test_build_approx_SHO()
    f0, fM, J = 0.02, 1.52e2, 20
    spectral_points, _ = Pioran.build_approx(J, f0, fM)
    @test length(spectral_points) == 20
    @test f0 * ((fM / f0)^(1 / (J - 1))) .^ (range(start=0, stop=J - 1, step=1)) ≈ spectral_points
end


function test_get_coefs()
    f0, fM, J = 0.02, 1.52e2, 20
    PS = SingleBendingPowerLaw(0.3, 0.02, 2.93)
    a = Pioran.get_approx_coefficients(PS, f0, fM, n_components=J)
    @test length(a) == J
    @test isfinite(a)
    @test a ≈ [1.3749158408973243, 0.26031747510091013, 0.06961116778917277, 0.013679642568525807, 0.0037949128465199307, 0.0008858780578830132, 0.00023278915565955668, 5.714159750636342e-5, 1.463191298808472e-5, 3.6532013241322788e-6, 9.262211884550235e-7, 2.3267166983266322e-7, 5.877072005450016e-8, 1.4801031386988674e-8, 3.728877337268077e-9, 9.44575715327315e-10, 2.3313738171903584e-10, 6.377629826311069e-11, 1.119218106083312e-11, 6.962520986945091e-12]
end

function test_approx_psd()
    α₁_set = [0.2, 0.03, 0.1, 0.46, 0.1, 0.21, 0.74, 0.1, 0.03, 0.92]
    f₁_set = [1.3e-2, 1.32e-1, 5.53e-2, 3.3, 0.342, 3.20e1, 1.3, 4e1, 1e-2, 0.5]
    α₂_set = [3.2, 3.1, 2.3, 2.57, 3.6, 2.3, 2.1, 2.79, 3.3, 3.8]
    f0, fM, J = 2e-3, 3.52e2, 25
    f = 10 .^ range(log10(f0), stop=log10(fM), length=1000)

    @testset "Check various psd shapes" begin
        for i in range(1, 10)
            @testset "Check approximated psd for α₁ = $(α₁_set[i]), f₁ = $(f₁_set[i]), α₂ = $(α₂_set[i])" begin
                α₁, f₁, α₂ = α₁_set[i], f₁_set[i], α₂_set[i]
                PS = SingleBendingPowerLaw(α₁, f₁, α₂)
                papprox = Pioran.approximated_psd(f, PS, f0, fM, n_components=J)
                @test maximum(abs.(PS(f) / PS(f[1]) - papprox / papprox[1])) < 1.0
                @test (all(isapprox.(PS(f) / PS(f[1]), papprox / papprox[1], atol=1e-2)))
            end
        end
    end
end

function test_approx_psd_DRWCelerite()
    α₁_set = [0.2, 0.03, 0.1, 0.46, 0.1, 0.21, 0.74, 0.1, 0.03, 0.92]
    f₁_set = [1.3e-2, 1.32e-1, 5.53e-2, 3.3, 0.342, 3.20e1, 1.3, 4e1, 1e-2, 0.5]
    α₂_set = [4.2, 3.1, 4.3, 5.57, 4.6, 2.3, 5.1, 2.79, 4.3, 5.8]
    f0, fM, J = 2e-3, 3.52e2, 35
    f = 10 .^ range(log10(f0), stop=log10(fM), length=1000)

    @testset "Check various psd shapes" begin
        for i in range(1, 10)
            @testset "Check approximated psd for α₁ = $(α₁_set[i]), f₁ = $(f₁_set[i]), α₂ = $(α₂_set[i])" begin
                α₁, f₁, α₂ = α₁_set[i], f₁_set[i], α₂_set[i]
                PS = SingleBendingPowerLaw(α₁, f₁, α₂)
                papprox = Pioran.approximated_psd(f, PS, f0, fM, n_components=J, basis_function="DRWCelerite")
                @test maximum(abs.(PS(f) / PS(f[1]) - papprox / papprox[1])) < 1.0
                @test (all(isapprox.(PS(f) / PS(f[1]), papprox / papprox[1], atol=1e-2)))
            end
        end
    end
end

function test_approx_cov()
    α₁_set = [0.2, 0.03, 0.1, 0.46, 0.1, 0.21, 0.74, 0.1, 0.03, 0.92]
    f₁_set = [1.3e-2, 1.32e-1, 5.53e-2, 3.3, 0.342, 3.20e1, 1.3, 4e1, 1e-2, 0.5]
    α₂_set = [3.2, 3.1, 2.3, 2.57, 3.6, 2.3, 2.1, 2.79, 3.3, 3.8]
    f0, fM, J = 2e-3, 3.52e2, 25
    variances = [1.32, 35.3, 242.2, 46.6, 0.3, 0.244, 9.64, 0.75, 0.193, 0.21]

    @testset "various psd shapes" begin
        for i in range(1, 10)
            @testset "Check approximated cov for α₁ = $(α₁_set[i]), f₁ = $(f₁_set[i]), α₂ = $(α₂_set[i])" begin
                α₁, f₁, α₂ = α₁_set[i], f₁_set[i], α₂_set[i]
                va = variances[i]
                PS = SingleBendingPowerLaw(α₁, f₁, α₂)
                Rapprox = Pioran.approx(PS, f0, fM, J, va)
                @test Rapprox(0, 0) ≈ va
            end
        end
        α₁, f₁, α₂ = α₁_set[1], f₁_set[1], α₂_set[1]
        va = variances[1]
        PS = SingleBendingPowerLaw(α₁, f₁, α₂)
        Rapprox = Pioran.approx(PS, f0, fM, J, va)
        @test Rapprox isa Pioran.SumOfSemiSeparable
        @test all([Rapprox.cov[i] isa Pioran.SHO for i in 1:J])
    end
end

function test_approx_cov_DRWCelerite()
    α₁_set = [0.2, 0.03, 0.1, 0.46, 0.1, 0.21, 0.74, 0.1, 0.03, 0.92]
    f₁_set = [1.3e-2, 1.32e-1, 5.53e-2, 3.3, 0.342, 3.20e1, 1.3, 4e1, 1e-2, 0.5]
    α₂_set = [3.2, 3.1, 2.3, 2.57, 3.6, 2.3, 2.1, 2.79, 3.3, 3.8]
    f0, fM, J = 2e-3, 3.52e2, 25
    variances = [1.32, 35.3, 242.2, 46.6, 0.3, 0.244, 9.64, 0.75, 0.193, 0.21]

    @testset "various psd shapes DRWCelerite" begin
        for i in range(1, 10)
            @testset "Check approximated cov for α₁ = $(α₁_set[i]), f₁ = $(f₁_set[i]), α₂ = $(α₂_set[i])" begin
                α₁, f₁, α₂ = α₁_set[i], f₁_set[i], α₂_set[i]
                va = variances[i]
                PS = SingleBendingPowerLaw(α₁, f₁, α₂)
                Rapprox = Pioran.approx(PS, f0, fM, J, va, basis_function="DRWCelerite")
                @test Rapprox(0, 0) ≈ va
            end
        end
        α₁, f₁, α₂ = α₁_set[1], f₁_set[1], α₂_set[1]
        va = variances[1]
        PS = SingleBendingPowerLaw(α₁, f₁, α₂)
        Rapprox = Pioran.approx(PS, f0, fM, J, va, basis_function="DRWCelerite")
        @test Rapprox isa Pioran.SumOfSemiSeparable
        @test all([Rapprox.cov[i] isa Pioran.Celerite && Rapprox.cov[i+1] isa Pioran.Exp for i in 1:2:2J])
    end
end

@testset "Power spectral density" begin
    test_SingleBendingPowerLaw()
    test_DoubleBendingPowerLaw()
    test_normalised_psd()
    test_build_approx_SHO()
    test_get_coefs()
    test_approx_psd()
    test_approx_psd_DRWCelerite()
    test_approx_cov()
    test_approx_cov_DRWCelerite()
end
