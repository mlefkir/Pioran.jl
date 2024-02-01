using CairoMakie
using VectorizedStatistics
using LombScargle


theme = Pioran.get_theme()
set_theme!(theme)

""" Plot the boxplot of the residuals and ratios for the PSD approximation """
function plot_boxplot_psd_approx(residuals, ratios; path="")

    if !ispath(path)
        mkpath(path)
    end
    meta_mean = vec(vmean(residuals, dims=1))
    meta_median = vec(vmedian(residuals, dims=1))
    meta_max = vec(maximum(abs.(residuals), dims=1))

    meta_mean_rat = vec(vmean(ratios, dims=1))
    meta_median_rat = vec(vmedian(ratios, dims=1))
    meta_max_rat = vec(maximum(abs.(ratios), dims=1))

    y = vec([meta_mean meta_median meta_max])
    x = rand(1:3, length(meta_mean))
    y2 = vec([meta_mean_rat meta_median_rat meta_max_rat])


    fig = Figure(size=(800, 600))
    ax1 = Axis(fig[1, 1], xticks=([1, 2, 3], ["mean", "median", "max"]), ylabel="Residuals", title="Distribution of the meta-(mean,max,median) of
         residuals and ratios for the PSD approximation")
    ax2 = Axis(fig[2, 1], xticks=([1, 2, 3], ["mean", "median", "max"]), ylabel="Ratios")

    CairoMakie.boxplot!(ax1, x, y)
    CairoMakie.boxplot!(ax2, x, y2, show_outliers=true)

    save(path * "boxplot_psd_approx.pdf", fig)
end

"""
    plot_quantiles_approx(f, f_min, f_max, residuals, ratios; path="")
    
Plot the quantiles of the residuals and ratios (with respect to the approximated PSD) of the PSD 

"""
function plot_quantiles_approx(f, f_min, f_max, residuals, ratios; path="")
    if !ispath(path)
        mkpath(path)
    end
    res_quantiles = vquantile!.(Ref(residuals), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)
    rat_quantiles = vquantile!.(Ref(ratios), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)

    fig = Figure(size=(800, 600))
    ax1 = Axis(fig[1, 1], xscale=log10, ylabel="Residuals", xminorticks=IntervalsBetween(9))
    ax2 = Axis(fig[2, 1], xscale=log10, ylabel="Ratios", xminorticks=IntervalsBetween(9), xlabel="Frequency")

    lines!(ax1, f, vec(res_quantiles[3]), label="Median")
    band!(ax1, f, vec(res_quantiles[1]), vec(res_quantiles[5]), color=(:blue, 0.2), label="95%")
    band!(ax1, f, vec(res_quantiles[2]), vec(res_quantiles[4]), color=(:blue, 0.4), label="68%")
    hlines!(ax1, 0, color=:red, linestyle=:dash)
    vlines!(ax1, [f_min; f_max], color=:black, linestyle=:dash, label="Observed range")
    lines!(ax2, f, vec(rat_quantiles[3]))
    band!(ax2, f, vec(rat_quantiles[1]), vec(rat_quantiles[5]), color=(:blue, 0.2))
    band!(ax2, f, vec(rat_quantiles[2]), vec(rat_quantiles[4]), color=(:blue, 0.4))
    hlines!(ax2, 1, color=:red, linestyle=:dash)
    vlines!(ax2, [f_min; f_max], color=:black, linestyle=:dash, label="Observed range")

    fig[3, 1] = Legend(fig, ax1, orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
        halign=:center, valign=:bottom,
        fontsize=10,
        framevisible=false)
    save(path * "quantiles_psd_approx.pdf", fig)
end

""" Plot the frequency-averaged residuals and ratios """
function plot_mean_approx(f, residuals, ratios; path="")
    if !ispath(path)
        mkpath(path)
    end
    mean_res = vec(mean(residuals, dims=2))
    mean_rat = vec(mean(ratios, dims=2))

    fig = Figure(size=(800, 600))
    ax1 = Axis(fig[1, 1], xscale=log10, ylabel="Residuals", xminorticks=IntervalsBetween(9))
    ax2 = Axis(fig[2, 1], xlabel="Frequency", xscale=log10, ylabel="Ratios", xminorticks=IntervalsBetween(9), yminorticks=IntervalsBetween(5))
    lines!(ax1, f, mean_res)
    lines!(ax2, f, mean_rat)
    hlines!(ax2, 1, color=:red, linestyle=:dash)
    hlines!(ax1, 0, color=:red, linestyle=:dash)
    save(path * "diagnostics_psd_approx.pdf", fig)
end

""" Check the approximation of the PSD 

Args:
    samples (Array{Float64, N}): The samples of the model parameters
    variance_samples (Array{Float64, 1}): The variance samples
    f0 (Float64): The minimum frequency
    fM (Float64): The maximum frequency
    model (SimpleBendingPowerLaw): The model

"""
function sample_approx_model(samples, variance_samples, f0, fM, model; n_frequencies=1_000, basis_function="SHO")
    P = size(samples, 2)
    f = collect(10 .^ range(log10(f0), log10(fM), n_frequencies))

    psd = [Pioran.calculate_psd.(f, Ref(model(samples[:, k]...))) for k in 1:P]
    psd = mapreduce(permutedims, vcat, psd)'
    psd ./= psd[1, :]'
    psd .*= variance_samples'

    psd_approx = [Pioran.approximated_psd(f, model(samples[:, k]...), f0, fM, var=variance_samples[k], basis_function=basis_function) for k in 1:P]
    psd_approx = mapreduce(permutedims, vcat, psd_approx)'

    residuals = psd .- psd_approx
    ratios = psd_approx ./ psd
    return psd, psd_approx, residuals, ratios, f
end

function plot_diag(f, residuals, ratios, f_min, f_max; path="")
    plot_mean_approx(f, residuals, ratios, path=path)
    plot_quantiles_approx(f, f_min, f_max, residuals, ratios, path=path)
    plot_boxplot_psd_approx(residuals, ratios, path=path)
end

""" Run the diagnostics for the PSD approximation
"""
function run_diagnostics(prior_samples, variance_samples, f0, fM, model, f_min, f_max; path="")
    _, _, residuals, ratios, f = sample_approx_model(prior_samples, variance_samples, f0, fM, model)
    plot_diag(f, residuals, ratios, f_min, f_max, path=path)
end



"""
plot_psd_ppc(samples, variance_samples, f0, fM, model; path="")

Plot the posterior predictive power spectral density
    
"""
function plot_psd_ppc(samples, variance_samples, f0, fM, model; plot_f_P=false, n_frequencies=1000, path="")
    theme = Pioran.get_theme()
    set_theme!(theme)

    if !ispath(path)
        mkpath(path)
    end
    P = size(variance_samples, 1)
    spectral_points, _ = Pioran.build_approx(20, f0, fM)

    psd, psd_approx, _, _, f = sample_approx_model(samples, variance_samples, f0, fM, model, n_frequencies=n_frequencies)

    amplitudes = [Pioran.get_approx_coefficients.(Ref(model(samples[:, k]...)), f0, fM) for k in 1:P]
    amplitudes = mapreduce(permutedims, vcat, amplitudes)'

    psd_m = psd ./ sum(amplitudes .* spectral_points, dims=1)
    psd_approx_m = psd_approx ./ sum(amplitudes .* spectral_points, dims=1)

    psd_quantiles = vquantile!.(Ref(psd_m), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)
    psd_approx_quantiles = vquantile!.(Ref(psd_approx_m), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)

    fig = Figure(size=(800, 600))

    if plot_f_P
        psd_quantiles = vquantile!.(Ref(f.*psd_m), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)
        psd_approx_quantiles = vquantile!.(Ref(f.*psd_approx_m), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)
    
        ax1 = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel=L"Frequency (${d}^{-1}$)", ylabel="f PSD",
        xminorticks=IntervalsBetween(9), yminorticks=IntervalsBetween(9), title="Posterior predictive power spectral density")

        lines!(ax1, f, vec(psd_quantiles[3]), label="Model Median", color=:blue)
        band!(ax1, f, vec(psd_quantiles[1]), vec(psd_quantiles[5]), color=(:blue, 0.2), label="95%")
        band!(ax1, f, vec(psd_quantiles[2]), vec(psd_quantiles[4]), color=(:blue, 0.4), label="68%")

        lines!(ax1, f, vec(psd_approx_quantiles[3]), label="Approx Median", color=:red)
        band!(ax1, f, vec(psd_approx_quantiles[1]), vec(psd_approx_quantiles[5]), color=(:red, 0.2), label="95%")
        band!(ax1, f, vec(psd_approx_quantiles[2]), vec(psd_approx_quantiles[4]), color=(:red, 0.4), label="68%")
    else
        ax1 = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel=L"Frequency (${d}^{-1}$)", ylabel="PSD",
        xminorticks=IntervalsBetween(9), yminorticks=IntervalsBetween(9), title="Posterior predictive power spectral density")

        lines!(ax1, f, vec(psd_quantiles[3]), label="Model Median", color=:blue)
        band!(ax1, f, vec(psd_quantiles[1]), vec(psd_quantiles[5]), color=(:blue, 0.2), label="95%")
        band!(ax1, f, vec(psd_quantiles[2]), vec(psd_quantiles[4]), color=(:blue, 0.4), label="68%")

        lines!(ax1, f,vec(psd_approx_quantiles[3]), label="Approx Median", color=:red)
        band!(ax1, f, vec(psd_approx_quantiles[1]), vec(psd_approx_quantiles[5]), color=(:red, 0.2), label="95%")
        band!(ax1, f,vec(psd_approx_quantiles[2]), vec(psd_approx_quantiles[4]), color=(:red, 0.4), label="68%")
    end
    fig[2, 1] = Legend(fig, ax1, orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
        halign=:center, valign=:bottom,
        fontsize=10, nbanks=3,
        framevisible=false)

    a = vcat([f', psd_quantiles..., psd_approx_quantiles...])

    open(path * "ppc_data.txt"; write=true) do f
        write(f, "# Posterior predictive power spectral density\n# quantiles=[0.025, 0.16, 0.5, 0.84, 0.975] \n# f, psd_quantiles, psd_approx_quantiles\n")
        writedlm(f, a)
    end
    save(path * "psd_ppc.pdf", fig)
end

"""
    plot_ppc_lsp(samples_𝓟, samples_ν, samples_μ, samples_variance, t, y, yerr, f0, fM, model; n_frequencies=1000, n_samples=1000, n_components=20, bin_fact=10, path="")

Plot the posterior predictive Lomb-Scargle periodogram

"""
function plot_lsp_ppc(samples_𝓟, samples_ν, samples_μ, samples_variance, t, y, yerr, f0, fM, model; n_frequencies=1000, n_samples=1000, n_components=20, bin_fact=10, path="")
    #set theme and create output directory
    theme = Pioran.get_theme()
    set_theme!(theme)
    if !ispath(path)
        mkpath(path)
    end

    # get the frequencies
    freq = exp.(range(log(f0), log(fM), length=n_frequencies))

    Power = []
    # get the posterior predictive lombscargle periodogram
    for k in 1:n_samples
        𝓟 = model(samples_𝓟[k, :]...)
        𝓡 = approx(𝓟, f0, fM, n_components, samples_variance[k])
        f = ScalableGP(samples_μ[k], 𝓡)
        σ2 = yerr .^ 2 * samples_ν[k]
        fx = f(t, σ2)
        # draw a time series from the GP
        y_sim = rand(fx)
        ls = lombscargle(t, y_sim, frequencies=freq)
        push!(Power, freqpower(ls)[2][1:end-1])
    end

    ls_array = mapreduce(permutedims, vcat, Power)'
    ls_quantiles = vquantile!.(Ref(ls_array), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)

    # compute the LSP of the observed data
    ls_obs = lombscargle(t, y, frequencies=freq)
    lsp = freqpower(ls_obs)[2][1:end-1]
    freq_obs = freqpower(ls_obs)[1][1:end-1]

    # bin the LSP in log space
    binned_periodogram = []
    binned_freqs = []
    n = Int(round(length(lsp) / bin_fact, digits=0))
    println(n)
    n_start = 0
    for i in n_start:n-2
        push!(binned_periodogram, mean(log.(lsp[1+i*bin_fact:(i+1)*bin_fact])))
        push!(binned_freqs, mean(log.(freq_obs[1+i*bin_fact:(i+1)*bin_fact])))
    end
    binned_periodogram = exp.(binned_periodogram)
    binned_freqs = exp.(binned_freqs)

    # save the data
    quantiles_fre = vcat([freq[1:end-1]', ls_quantiles...])
    open(path * "lsp_ppc_data.txt"; write=true) do f
        write(f, "# Posterior predictive Lomb-Scargle\n# quantiles=[0.025, 0.16, 0.5, 0.84, 0.975] \n# freq, ls_quantiles\n")
        writedlm(f, quantiles_fre)
    end

    binned_data = hcat(binned_freqs, binned_periodogram)
    open(path * "binned_lsp_data.txt"; write=true) do f
        write(f, "# Binned Lomb-Scargle of the data\n# freq, lsp\n")
        writedlm(f, binned_data)
    end


    # min and max freqs of the obs data
    f_min = 1 / (t[end] - t[1])
    f_max = 1 / minimum(diff(t)) / 2

    # plot the posterior predictive LSP
    fig = Figure(size=(800, 600))
    ax1 = Axis(fig[1, 1],
        xscale=log10,
        yscale=log10,
        xlabel=L"Frequency (${d}^{-1}$)",
        ylabel="PSD",
        xminorticks=IntervalsBetween(9),
        yminorticks=IntervalsBetween(9),
        title="Posterior predictive Lomb-Scargle periodogram")
    vlines!(ax1, [f_min; f_max], color=:black, linestyle=:dash, label="Observed window")

    lines!(ax1, freq[1:end-1], vec(ls_quantiles[3]), label="LSP realisations", color=:blue)
    band!(ax1, freq[1:end-1], vec(ls_quantiles[1]), vec(ls_quantiles[5]), color=(:blue, 0.1), label="95%")
    band!(ax1, freq[1:end-1], vec(ls_quantiles[2]), vec(ls_quantiles[4]), color=(:blue, 0.2), label="68%")
    lines!(ax1, binned_freqs, binned_periodogram, color=:red, linewidth=2, label="Binned LSP")

    fig[2, 1] = Legend(fig, ax1, orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
        halign=:center, valign=:bottom,
        fontsize=10, nbanks=2,
        framevisible=false)

    fig
    save(path * "LSP_ppc.pdf", fig)
end
