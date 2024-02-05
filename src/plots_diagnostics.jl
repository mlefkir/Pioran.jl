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

    boxplot!(ax1, x, y)
    boxplot!(ax2, x, y2, show_outliers=true)

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

    #   figure
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
function sample_approx_model(samples, variance_samples, f0, fM, model; n_frequencies=1_000, basis_function="SHO", n_components=20)
    P = size(samples, 2)
    f = collect(10 .^ range(log10(f0), log10(fM), n_frequencies))

    psd = [Pioran.calculate_psd.(f, Ref(model(samples[:, k]...))) for k in 1:P]
    psd = mapreduce(permutedims, vcat, psd)'
    psd ./= psd[1, :]'
    psd .*= variance_samples'

    psd_approx = [Pioran.approximated_psd(f, model(samples[:, k]...), f0, fM, var=variance_samples[k], basis_function=basis_function, n_components=n_components) for k in 1:P]
    psd_approx = mapreduce(permutedims, vcat, psd_approx)'

    residuals = psd .- psd_approx
    ratios = psd_approx ./ psd
    return psd, psd_approx, residuals, ratios, f
end

""" Plot the diagnostics of the approximation of the PSD 
    The diagnostics include the mean, quantiles and boxplot of the residuals and ratios

    This function is a wrapper for the following functions:
    - plot_mean_approx
    - plot_quantiles_approx
    - plot_boxplot_psd_approx
"""
function plot_diag(f, residuals, ratios, f_min, f_max; path="")
    plot_mean_approx(f, residuals, ratios, path=path)
    plot_quantiles_approx(f, f_min, f_max, residuals, ratios, path=path)
    plot_boxplot_psd_approx(residuals, ratios, path=path)
end

function run_diagnostics(prior_samples, variance_samples, f0, fM, model, f_min, f_max; path="", basis_function="SHO", n_components=20)
    _, _, residuals, ratios, f = sample_approx_model(prior_samples, variance_samples, f0, fM, model, basis_function=basis_function, n_components=n_components)
    plot_diag(f, residuals, ratios, f_min, f_max, path=path)
end


"""
plot_psd_ppc(samples, samples_variance, f0, fM, model; path="")

Plot the posterior predictive power spectral density
    
"""
function plot_psd_ppc(samples, samples_variance, samples_ν, t, yerr, f0, fM, model; plot_f_P=false, n_frequencies=1000, path="", n_components=20, basis_function="SHO")
    theme = Pioran.get_theme()
    set_theme!(theme)

    if !ispath(path)
        mkpath(path)
    end

    dt = diff(t)

    f_min = 1 / (t[end] - t[1])
    f_max = 1 / minimum(dt) / 2.0

    mean_dt = mean(dt)
    median_dt = median(dt)
    mean_sq_err = mean(yerr .^ 2)
    median_sq_err = median(yerr .^ 2)
    mean_ν = mean(samples_ν)
    median_ν = median(samples_ν)
    mean_noise_level = 2 * mean_ν * mean_sq_err * mean_dt
    median_noise_level = 2 * median_ν * median_sq_err * median_dt

    P = size(samples_variance, 1)
    spectral_points, _ = Pioran.build_approx(n_components, f0, fM, basis_function=basis_function)

    psd, psd_approx, _, _, f = sample_approx_model(samples, samples_variance, f0, fM, model, n_frequencies=n_frequencies, basis_function=basis_function, n_components=n_components)

    amplitudes = [Pioran.get_approx_coefficients.(Ref(model(samples[:, k]...)), f0, fM, basis_function=basis_function, n_components=n_components) for k in 1:P]
    amplitudes = mapreduce(permutedims, vcat, amplitudes)'

    psd_m = psd ./ sum(amplitudes .* spectral_points, dims=1)
    psd_approx_m = psd_approx ./ sum(amplitudes .* spectral_points, dims=1)

    psd_quantiles = vquantile!.(Ref(psd_m), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)
    psd_approx_quantiles = vquantile!.(Ref(psd_approx_m), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)

    fig = Figure(size=(800, 600))

    if plot_f_P

        psd_quantiles = vquantile!.(Ref(f .* psd_m), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)
        psd_approx_quantiles = vquantile!.(Ref(f .* psd_approx_m), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)

        ax1 = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel=L"Frequency (${d}^{-1}$)", ylabel="f PSD",
            xminorticks=IntervalsBetween(9), yminorticks=IntervalsBetween(9), title="Posterior predictive power spectral density")

        lines!(ax1, f, vec(psd_quantiles[3]), label="Model Median", color=:blue)
        band!(ax1, f, vec(psd_quantiles[1]), vec(psd_quantiles[5]), color=(:blue, 0.2), label="95%")
        band!(ax1, f, vec(psd_quantiles[2]), vec(psd_quantiles[4]), color=(:blue, 0.4), label="68%")

        lines!(ax1, f, vec(psd_approx_quantiles[3]), label="Approx Median", color=:red)
        band!(ax1, f, vec(psd_approx_quantiles[1]), vec(psd_approx_quantiles[5]), color=(:red, 0.2), label="95%")
        band!(ax1, f, vec(psd_approx_quantiles[2]), vec(psd_approx_quantiles[4]), color=(:red, 0.4), label="68%")

        lines!(ax1, f, f .* mean_noise_level, label="Mean noise level", color=:black, linestyle=:dash)
        lines!(ax1, f, f .* median_noise_level, label="Median noise level", color=:black)

        vlines!(ax1, [f_min; f_max], color=:black, linestyle=:dot, label="Observed window")
    else
        ax1 = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel=L"Frequency (${d}^{-1}$)", ylabel="PSD",
            xminorticks=IntervalsBetween(9), yminorticks=IntervalsBetween(9), title="Posterior predictive power spectral density")

        lines!(ax1, f, vec(psd_quantiles[3]), label="Model Median", color=:blue)
        band!(ax1, f, vec(psd_quantiles[1]), vec(psd_quantiles[5]), color=(:blue, 0.2), label="95%")
        band!(ax1, f, vec(psd_quantiles[2]), vec(psd_quantiles[4]), color=(:blue, 0.4), label="68%")

        lines!(ax1, f, vec(psd_approx_quantiles[3]), label="Approx Median", color=:red)
        band!(ax1, f, vec(psd_approx_quantiles[1]), vec(psd_approx_quantiles[5]), color=(:red, 0.2), label="95%")
        band!(ax1, f, vec(psd_approx_quantiles[2]), vec(psd_approx_quantiles[4]), color=(:red, 0.4), label="68%")

        lines!(ax1, f, ones(length(f)) * mean_noise_level, label="Mean noise level", color=:black, linestyle=:dash)
        lines!(ax1, f, ones(length(f)) * median_noise_level, label="Median noise level", color=:black)
        vlines!(ax1, [f_min; f_max], color=:black, linestyle=:dot, label="Observed window")

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
    plot_ppc_lsp(samples_𝓟,samples_variance, samples_ν, samples_μ, t, y, yerr, f0, fM, model; plot_f_P=false, n_frequencies=1000, n_samples=1000, n_components=20, bin_fact=10, path="")

Plot the posterior predictive Lomb-Scargle periodogram

"""
function plot_lsp_ppc(samples_𝓟, samples_variance, samples_ν, samples_μ, t, y, yerr, f0, fM, model; plot_f_P=false, n_frequencies=1000, n_samples=1000, n_components=20, bin_fact=10, path="", basis_function="SHO")
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
    @showprogress for k in 1:n_samples
        𝓟 = model(samples_𝓟[k, :]...)
        𝓡 = approx(𝓟, f0, fM, n_components, samples_variance[k], basis_function=basis_function)
        f = ScalableGP(samples_μ[k], 𝓡)
        σ2 = yerr .^ 2 * samples_ν[k]
        fx = f(t, σ2)
        # draw a time series from the GP
        y_sim = rand(fx)
        # periodog = LombScargle.plan(t, s)
        ls = lombscargle(t, y_sim, yerr, frequencies=freq)
        push!(Power, freqpower(ls)[2][1:end-1])
    end

    ls_array = mapreduce(permutedims, vcat, Power)'
    if plot_f_P
        label = "f * Periodogram"
        ls_quantiles = vquantile!.(Ref(freq[1:end-1] .* ls_array), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)
    else
        label = "Periodogram"
        ls_quantiles = vquantile!.(Ref(ls_array), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)
    end
    # compute the LSP of the observed data
    ls_obs = lombscargle(t, y, yerr, frequencies=freq)
    lsp = freqpower(ls_obs)[2][1:end-1]
    freq_obs = freqpower(ls_obs)[1][1:end-1]

    # bin the LSP in log space
    binned_periodogram = []
    binned_freqs = []
    n = Int(round(length(lsp) / bin_fact, digits=0))
    n_start = 0
    for i in n_start:n-2
        push!(binned_periodogram, mean(log.(lsp[1+i*bin_fact:(i+1)*bin_fact])))
        push!(binned_freqs, mean(log.(freq_obs[1+i*bin_fact:(i+1)*bin_fact])))
    end
    binned_periodogram = exp.(binned_periodogram)
    binned_freqs = exp.(binned_freqs)
    if plot_f_P
        binned_periodogram = binned_periodogram .* binned_freqs
    end
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
        ylabel=label,
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

""" 
    get_ppc_timeseries(samples_𝓟, samples_variance, samples_ν, samples_μ, t, y, yerr,f0,fM, model,with_log_transform;samples_c=missing, n_samples=1000, path="")
"""
function get_ppc_timeseries(samples_𝓟, samples_variance, samples_ν, samples_μ, t, y, yerr, f0, fM, model, with_log_transform; t_pred=nothing, samples_c=nothing, n_samples=1000, n_components=20, basis_function="SHO", path="")
    theme = Pioran.get_theme()
    set_theme!(theme)
    if !ispath(path)
        mkpath(path)
    end
    P = n_samples

    if isnothing(samples_ν)
        samples_ν = ones(P)
    end

    if isnothing(samples_c)
        samples_c = zeros(P)
    end

    if isnothing(t_pred)
        t_pred = collect(range(t[1], t[end], length=2 * length(t)))
    end

    t_pred = sort(unique(vcat(t, t_pred)))
    samples_pred = []
    @showprogress for i in 1:P

        # Rescale the measurement variance and make the flux Gaussian
        if with_log_transform
            y_obs = log.(y .- samples_c[i])
            σ² = samples_ν[i] .* yerr .^ 2 ./ (y .- samples_c[i]) .^ 2
        else
            y_obs = y
            σ² = samples_ν[i] .* yerr .^ 2
        end

        # Define power spectral density function
        𝓟 = model(samples_𝓟[i, :]...)

        # Approximation of the PSD to form a covariance function
        𝓡 = approx(𝓟, f0, fM, n_components, samples_variance[i], basis_function=basis_function)

        # Build the GP
        f = ScalableGP(samples_μ[i], 𝓡)
        fx = f(t, σ²)

        # Condition the GP on the observed data
        fp = posterior(fx, y_obs)

        realisation = rand(fp, t_pred, 1)
        if with_log_transform
            push!(samples_pred, exp.(realisation .+ samples_c[i]))
        else
            push!(samples_pred, realisation)
        end
    end
    ts_array = mapreduce(permutedims, vcat, samples_pred)'

    return ts_array, t_pred
end

""" Plot the residuals and the autocorrelation function of the residuals 
    
    plot_residuals_diagnostics(t, mean_res, res_quantiles; confidence_intervals=[95, 99], path="")

"""
function plot_residuals_diagnostics(t, mean_res, res_quantiles; confidence_intervals=[95, 99], path="")

    sigs = [quantile(Normal(0, 1), (50 + ci / 2) / 100) for ci in confidence_intervals]
    fig = Figure(size=(1000, 600))
    gc = fig[1, :] = GridLayout()
    gd = fig[2, :] = GridLayout()

    ax1 = Axis(gc[1, 1],
        xlabel="Time (d)",
        ylabel="Residuals",
        xminorticks=IntervalsBetween(9),
        yminorticks=IntervalsBetween(9))
    lines!(ax1, t, vec(mean_res), color=:blue, label="mean")
    lines!(ax1, t, vec(res_quantiles[3]), label="median realisation", color=:black)
    band!(ax1, t, vec(res_quantiles[1]), vec(res_quantiles[5]), color=(:black, 0.1), label="95%")
    band!(ax1, t, vec(res_quantiles[2]), vec(res_quantiles[4]), color=(:black, 0.2), label="68%")
    ax2 = Axis(gc[1, 2],
        xminorticks=IntervalsBetween(9),
        yminorticks=IntervalsBetween(9))
    hist!(ax2, vec(res_quantiles[3]), bins=20, color=:black, alpha=0.25, label="Residuals", direction=:x)
    hist!(ax2, vec(mean_res), bins=20, color=:blue, alpha=0.01, label="Residuals", direction=:x)
    linkyaxes!(ax1, ax2)
    colsize!(gc, 1, Auto(length(0:0.1:5)))
    colsize!(gc, 2, Auto(length(0:0.1:2)))
    colgap!(gc, 1)
    ax3 = Axis(gd[1, 1],
        xlabel="Lag (indices)",
        ylabel="ACVF",
        xminorticks=IntervalsBetween(9),
        yminorticks=IntervalsBetween(9))


    lags = 0:Int(length(mean_res) // 10)
    acvf = autocor(mean_res, lags)
    acvf_median = autocor(vec(res_quantiles[3]), lags)


    stem!(ax3, lags, vec(acvf), color=:black, label="ACVF")
    stem!(ax3, lags, vec(acvf_median), color=:blue, label="ACVF median")

    band!(ax3, lags, sigs[1] * ones(length(lags)) / sqrt(length(t)), -sigs[1] * ones(length(lags)) / sqrt(length(t)), color=(:black, 0.1), label="95%")
    band!(ax3, lags, sigs[2] * ones(length(lags)) / sqrt(length(t)), -sigs[2] * ones(length(lags)) / sqrt(length(t)), color=(:black, 0.1), label="99%")
    fig
    save(path * "residuals_diagnostics.pdf", fig)
    # return fig
end

""" Plot the posterior predictive time series """
function plot_simu_ppc_timeseries(t_pred, ts_quantiles, t, y, yerr; path="")

    fig = Figure(size=(1000, 600))
    ax1 = Axis(fig[1, 1],
        xlabel="Time (d)",
        ylabel="Time series",
        xminorticks=IntervalsBetween(9),
        yminorticks=IntervalsBetween(9),
        title="Posterior predictive simulated time series")
    errorbars!(ax1, t, y, yerr)
    scatter!(ax1, t, y, marker=:circle, markersize=7.5, label="Data")
    lines!(ax1, t_pred, vec(ts_quantiles[3]), label="median realisation", color=:black, linealpha=0.5, linewidth=1)
    band!(ax1, t_pred, vec(ts_quantiles[1]), vec(ts_quantiles[5]), color=(:black, 0.1), label="95%")
    band!(ax1, t_pred, vec(ts_quantiles[2]), vec(ts_quantiles[4]), color=(:black, 0.2), label="68%")
    fig[2, 1] = Legend(fig, ax1, orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
        halign=:center, valign=:bottom,
        fontsize=10, nbanks=1,
        framevisible=false)
    fig
    save(path * "TS_ppc.pdf", fig)
    # return fig
end

""" 
    plot_ppc_timeseries(samples_𝓟, samples_variance, samples_ν, samples_μ, t, y, yerr, f0, fM, model, with_log_transform; t_pred=nothing, samples_c=nothing, n_samples=1000, n_components=20, basis_function="SHO", path="")

Plot the posterior predictive time series and the residuals 
"""
function plot_ppc_timeseries(samples_𝓟, samples_variance, samples_ν, samples_μ, t, y, yerr, f0, fM, model, with_log_transform; t_pred=nothing, samples_c=nothing, n_samples=1000, n_components=20, basis_function="SHO", path="")
    # get the posterior predictive time series and the prediction times
    ts_array, t_pred = get_ppc_timeseries(samples_𝓟, samples_variance, samples_ν, samples_μ, t, y, yerr, f0, fM, model, with_log_transform, t_pred=t_pred, samples_c=samples_c, n_samples=n_samples, n_components=n_components, basis_function=basis_function, path=path)
    # find the indexes of the observed data in the prediction times for the residuals
    indexes = [findall(t_pred -> t_pred == t_i, t_pred)[1] for t_i in t]

    ts_quantiles = vquantile!.(Ref(ts_array), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)
    res = (y .- ts_array[indexes, :]) ./ yerr
    res_quantiles = vquantile!.(Ref(res), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)
    mean_res = mean(res, dims=2)


    plot_simu_ppc_timeseries(t_pred, ts_quantiles, t, y, yerr, path=path)
    plot_residuals_diagnostics(t, mean_res, res_quantiles, path=path)

    ts_quantiles = mapreduce(permutedims, vcat, ts_quantiles)
    writedlm(path * "ppc_timeseries_quantiles.txt", ts_quantiles)
    res_quantiles = mapreduce(permutedims, vcat, res_quantiles)
    writedlm(path * "ppc_residuals_quantiles.txt", res_quantiles)
    writedlm(path * "ppc_residuals_mean.txt", mean_res)
    writedlm(path * "ppc_t_pred.txt", t_pred)

end

""" 

# """
# function plot_posterior_predictive_checks(samples_𝓟, samples_variance, samples_ν, samples_μ, t, y, yerr, f0, fM, model, with_log_transform; t_pred=nothing, samples_c=nothing, n_samples=1000, n_components=20, basis_function="SHO", path="")

#     plot_lsp_ppc(samples_𝓟, samples_variance, samples_ν, samples_μ, t, y, yerr, f0, fM, model; plot_f_P=false, n_frequencies=1000, n_samples=1000, n_components=20, bin_fact=10, path="", basis_function="SHO")
#     plot_ppc_timeseries(samples_𝓟, samples_variance, samples_ν, samples_μ, t, y, yerr, f0, fM, model, with_log_transform, t_pred=t_pred, samples_c=samples_c, n_samples=n_samples, n_components=n_components, basis_function=basis_function, path=path)

# end