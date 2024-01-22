using PkgBenchmark, BenchmarkTools, CairoMakie
using DelimitedFiles

results = readresults("results_celerite.json")
suite = results.benchmarkgroup["inference"]
med = median(suite)

n_samples = [50, 100, 200, 500, 800, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
n_components = [10, 20, 25, 30, 40, 50]

ArrMediansTime = zeros(length(n_samples), length(n_components))
ArrMediansMemory = zeros(length(n_samples), length(n_components))

for (j, J) in enumerate(n_components)
    for (n, N) in enumerate(n_samples)
        ArrMediansTime[n, j] = med["$J"]["$N"].time / 1e9
        ArrMediansMemory[n, j] = med["$J"]["$N"].memory / 1024^2
    end
end

writedlm("median_time.txt", ArrMediansTime)
writedlm("median_memory.txt", ArrMediansMemory)

fig = Figure(resolution=(700, 400), font="sans", figure_padding=3)
ax = Axis(fig[1, 1], xscale=log10, xticksmirrored=true,
    yticksmirrored=true,
    xminorticksvisible=true,
    yminorticksvisible=true,
    xminorgridvisible=false,
    xgridvisible=false,
    yminorgridvisible=false,
    ygridvisible=false,
    xminortickalign=1, xminorticks=IntervalsBetween(9),
    yminorticks=IntervalsBetween(9),
    yminortickalign=1, xtickalign=1, ytickalign=1, yscale=log10, xlabel="N",
    ylabel="Time (s)", title="Likelihood evaluation time",)

ax2 = Axis(fig[1, 2], xscale=log10, xticksmirrored=true,
    yticksmirrored=true,
    xminorticksvisible=true,
    yminorticksvisible=true,
    xminorgridvisible=false,
    xgridvisible=false,
    yminorgridvisible=false,
    ygridvisible=false,
    xminortickalign=1, xminorticks=IntervalsBetween(9),
    yminorticks=IntervalsBetween(9),
    yminortickalign=1, xtickalign=1, ytickalign=1, yscale=log10, xlabel="N",
    ylabel="Memory (MiB)", title="Likelihood evaluation memory usage",)
# ax.xticks = [100,  1_000, 10_000,  100_000]
for (j, J) in enumerate(n_components)
    lines!(ax, n_samples, ArrMediansTime[:, j])
    scatter!(ax, n_samples, ArrMediansTime[:, j], label="$J")
    lines!(ax2, n_samples, ArrMediansMemory[:, j])
    scatter!(ax2, n_samples, ArrMediansMemory[:, j], label="$J")
end
fig[2, 1:2] = Legend(fig[1, 1],
    ax,
    "Number of components",
    orientation=:horizontal,
    tellwidth=false,
    tellheight=true,
    halign=:center, valign=:bottom,
    fontsize=10,
    framevisible=false)
fig
save("Likelihood_benchmarks_bis.pdf", fig, px_per_unit=0.5)
save("Likelihood_benchmarks_bis.png", fig)
