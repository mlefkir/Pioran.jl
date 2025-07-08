using Pioran, Test, Random, Distributions, DelimitedFiles

function single_prior_transform(cube, μₙ = 0.0, σₙ = 1.0)
    α₁ = quantile(Uniform(0.0, 1.5), cube[1])
    f₁ = quantile(LogUniform(1.0e-3, 1.0e3), cube[2])
    α₂ = quantile(Uniform(α₁, 6.0), cube[3])
    variance = quantile(LogNormal(μₙ, σₙ), cube[4])
    return [α₁, f₁, α₂, variance]
end

function test_diagnostics_single()
    rng = MersenneTwister(123)
    unif = rand(rng, 4, 3000) # uniform samples from the unit hypercube
    priors = mapreduce(permutedims, hcat, [single_prior_transform(unif[:, i]) for i in 1:3000]') # transform the uniform samples to the prior

    f_min, f_max = 1.0e-3, 1.0e3
    model = SingleBendingPowerLaw
    basis_function = "DRWCelerite"
    n_components = 20

    figs, _ = run_diagnostics(priors[1:3, :], priors[4, :], f_min, f_max, model, path = "plots/", basis_function = basis_function, n_components = n_components)

    @test length(figs) == 3
    for i in 1:3
        @test figs[i] isa Pioran.CairoMakie.Figure
    end
    return
end

function test_posterior_plots()
    dir = "data/simu_single_123_factor/"
    samples = readdlm(dir * "/chains/equal_weighted_post.txt", skipstart = 1)
    filename = "data/simu.txt"
    A = readdlm(filename, comments = true, comment_char = '#')
    t, y, yerr = A[:, 1], A[:, 2], A[:, 3]
    t = t .- t[1]
    model = SingleBendingPowerLaw
    S_low = S_high = 20.0
    path = "plots/"
    basis_function = "DRWCelerite"
    n_components = 30
    n_frequencies = 500
    with_log_transform = true
    plot_f_P = true
    is_integrated_power = true
    paramnames = ["α₁", "f₁", "α₂", "variance", "ν", "μ"]

    paramnames_split = Dict(
        "psd" => ["α₁", "f₁", "α₂"],
        "norm" => "variance",
        "scale_err" => "ν",
        "mean" => "μ"
    )

    function get_GP(t, y, yerr, params, basis_function = "SHO", n_components = 20, model = SingleBendingPowerLaw)

        T = (t[end] - t[1]) # duration of the time series
        Δt = minimum(diff(t)) # min time separation

        f_min, f_max = 1 / T, 1 / Δt / 2


        α₁, f₁, α₂, variance, ν, μ = params

        # Rescale the measurement variance
        σ² = ν .* yerr .^ 2

        # Define the mean

        # Define power spectral density function
        𝓟 = model(α₁, f₁, α₂)

        # Approximation of the PSD to form a covariance function
        𝓡 = approx(𝓟, f_min, f_max, n_components, variance, basis_function = basis_function)

        # Build the GP
        f = ScalableGP(μ, 𝓡)

        # Condition on the times and errors
        fx = f(t, σ²)
        return fx
    end


    figs = run_posterior_predict_checks(samples, paramnames, paramnames_split, t, y, yerr, model, get_GP, with_log_transform; S_low = S_low, S_high = S_high, is_integrated_power = is_integrated_power, plots = ["psd", "lsp", "timeseries"], n_samples = 10, path = path, basis_function = basis_function, n_frequencies = n_frequencies, plot_f_P = plot_f_P, n_components = n_components)

    @test length(figs) == 4

    @test figs[1] isa Pioran.CairoMakie.Figure
    @test figs[2] isa Pioran.CairoMakie.Figure
    @test figs[3] isa Pioran.CairoMakie.Figure
    @test figs[4] isa Pioran.CairoMakie.Figure

    @test length(figs[2].content) > 1
    return @test length(figs[1].content) > 1

end

@testset "plots" begin

    test_diagnostics_single()
    test_posterior_plots()
end
