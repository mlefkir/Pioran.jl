using CairoMakie
using VectorizedStatistics

""" Plot the boxplot of the residuals and ratios for the PSD approximation """
function plot_boxplot_psd_approx(residuals, ratios; path="")

    if ! ispath(path)
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
    if ! ispath(path)
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
    if ! ispath(path)
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
function sample_approx_model(samples, variance_samples, f0, fM, model, basis_function="SHO")
    P = size(samples, 2)
    f = collect(10 .^ range(log10(f0), log10(fM), 1000))

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

function run_diagnostics(prior_samples, variance_samples, f0, fM, model, f_min, f_max; path="")
    _, _, residuals, ratios, f = sample_approx_model(prior_samples, variance_samples, f0, fM, model)
    plot_diag(f, residuals, ratios, f_min, f_max, path=path)
end


"""
    psd_ppc(samples, variance_samples, f0, fM, model; path="")

Plot the posterior predictive power spectral density
    
    """
function psd_ppc(samples, variance_samples, f0, fM, model; path="")
    if !ispath(path)
        mkpath(path)
    end
    psd, psd_approx, _, _, f = sample_approx_model(samples, variance_samples, f0, fM, model)

    psd_quantiles = vquantile!.(Ref(psd), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)
    psd_approx_quantiles = vquantile!.(Ref(psd_approx), [0.025, 0.16, 0.5, 0.84, 0.975], dims=2)

    fig = Figure(size=(800, 600))
    ax1 = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel=L"Frequency (${d}^{-1}$)", ylabel="PSD",
        xminorticks=IntervalsBetween(9), yminorticks=IntervalsBetween(9), title="Posterior predictive power spectral density")

    lines!(ax1, f, vec(psd_quantiles[3]), label="Model Median", color=:blue)
    band!(ax1, f, vec(psd_quantiles[1]), vec(psd_quantiles[5]), color=(:blue, 0.2), label="95%")
    band!(ax1, f, vec(psd_quantiles[2]), vec(psd_quantiles[4]), color=(:blue, 0.4), label="68%")

    lines!(ax1, f, vec(psd_approx_quantiles[3]), label="Approx Median", color=:red)
    band!(ax1, f, vec(psd_approx_quantiles[1]), vec(psd_approx_quantiles[5]), color=(:red, 0.2), label="95%")
    band!(ax1, f, vec(psd_approx_quantiles[2]), vec(psd_approx_quantiles[4]), color=(:red, 0.4), label="68%")
    fig[2, 1] = Legend(fig, ax1, orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
        halign=:center, valign=:bottom,
        fontsize=10, nbanks=3,
        framevisible=false)
    save(path * "psd_ppc.pdf", fig)

end