using PkgBenchmark, BenchmarkTools, CairoMakie
using DelimitedFiles

results = readresults("results_celerite_terms.json")
suite = results.benchmarkgroup["inference"]
med = median(suite)

n_samples = 2 .^ (5:15)
n_components = 2 .^ (1:6)

ArrMediansTime = zeros(length(n_samples), length(n_components))
ArrMediansMemory = zeros(length(n_samples), length(n_components))

for (j, J) in enumerate(n_components)
    for (n, N) in enumerate(n_samples)
        ArrMediansTime[n, j] = med["$J"]["$N"].time / 1.0e9
        ArrMediansMemory[n, j] = med["$J"]["$N"].memory / 1024^2
    end
end

writedlm("median_time_celerite.txt", ArrMediansTime)
writedlm("median_memory_celerite.txt", ArrMediansMemory)

suite = results.benchmarkgroup["inference2"]
med = median(suite)


ArrMediansTime2 = zeros(length(n_samples), length(n_components))
ArrMediansMemory2 = zeros(length(n_samples), length(n_components))

for (j, J) in enumerate(n_components)
    for (n, N) in enumerate(n_samples)
        ArrMediansTime2[n, j] = med["$J"]["$N"].time / 1.0e9
        ArrMediansMemory2[n, j] = med["$J"]["$N"].memory / 1024^2
    end
end

writedlm("median_time_celerite2.txt", ArrMediansTime2)
writedlm("median_memory_celerite2.txt", ArrMediansMemory2)

fig = Figure(resolution = (700, 400), font = "sans", figure_padding = 3)

ax = Axis(
    fig[1, 1], xscale = log10, xticksmirrored = true,
    yticksmirrored = true,
    xminorticksvisible = true,
    yminorticksvisible = true,
    xminorgridvisible = false,
    xgridvisible = false,
    yminorgridvisible = false,
    ygridvisible = false,
    xminortickalign = 1, xminorticks = IntervalsBetween(9),
    yminorticks = IntervalsBetween(9),
    xticks = LogTicks(-5:-1),
    yminortickalign = 1, xtickalign = 1, ytickalign = 1, yscale = log10, xlabel = "N",
    ylabel = "Time (s)", title = "Likelihood evaluation time",
)

ax2 = Axis(
    fig[1, 2], xscale = log10, xticksmirrored = true,
    yticksmirrored = true,
    xminorticksvisible = true,
    yminorticksvisible = true,
    xminorgridvisible = false,
    xgridvisible = false,
    yminorgridvisible = false,
    ygridvisible = false,
    xminortickalign = 1, xminorticks = IntervalsBetween(9),
    yminorticks = IntervalsBetween(9),
    yminortickalign = 1, xtickalign = 1, ytickalign = 1, yscale = log10, xlabel = "N",
    ylabel = "Memory (MiB)", title = "Likelihood evaluation memory usage",
)
# ax.xticks = [100,  1_000, 10_000,  100_000]
for (j, J) in enumerate(n_components)
    lines!(ax, n_samples, ArrMediansTime[:, j])
    scatter!(ax, n_samples, ArrMediansTime[:, j], label = "$J")
    lines!(ax2, n_samples, ArrMediansMemory[:, j])
    scatter!(ax2, n_samples, ArrMediansMemory[:, j], label = "$J")

    lines!(ax, n_samples, ArrMediansTime2[:, j], linestyle = :dash)
    scatter!(ax, n_samples, ArrMediansTime2[:, j], marker = :rect, label = "$J sec")
    lines!(ax2, n_samples, ArrMediansMemory2[:, j], linestyle = :dash)
    scatter!(ax2, n_samples, ArrMediansMemory2[:, j], marker = :rect, label = "$J sec")

end
fig[2, 1:2] = Legend(
    fig[1, 1],
    ax,
    "Number of components",
    orientation = :horizontal,
    tellwidth = false,
    tellheight = true,
    halign = :center, valign = :bottom,
    fontsize = 10,
    framevisible = false
)
fig
save("Likelihood_benchmarks_celerite_all.pdf", fig, px_per_unit = 0.5)
save("Likelihood_benchmarks_celerite_all.png", fig)
