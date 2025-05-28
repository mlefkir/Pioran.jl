using CairoMakie
using HDF5

rev = ARGS[1]


fid = h5open("plots/bench_celerite_$(rev).h5", "r")

for data_label in ["celerite_likelihood", "SHO", "DRWCelerite"]

    data = read(fid[data_label])
    n_samples = read(fid[data_label*"_N"])
    n_components = read(fid[data_label*"_J"])

    println(size(data))
    ArrMediansTime = data[:, :, 1]'
    ArrMediansMemory = data[:, :, 5]'
    ArrPerc25 = data[:, :, 3]'
    ArrPerc75 = data[:, :, 4]'


    fig = Figure(resolution=(700, 400), font="sans", figure_padding=3)
    ax = Axis(
        fig[1, 1],
        yticks=LogTicks(WilkinsonTicks(6, k_min=5)),
        xscale=log10,
        xticksmirrored=true,
        yticksmirrored=true,
        xminorticksvisible=true,
        yminorticksvisible=true,
        xminorgridvisible=false,
        xgridvisible=false,
        yminorgridvisible=false,
        ygridvisible=false,
        xminortickalign=1,
        xminorticks=IntervalsBetween(9),
        yminorticks=IntervalsBetween(9),
        yminortickalign=1,
        xtickalign=1,
        ytickalign=1,
        yscale=log10,
        xlabel="N",
        ylabel="Time (s)",
        title="Likelihood evaluation time",
    )

    ax2 = Axis(
        fig[1, 2], xscale=log10, xticksmirrored=true,
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
        ylabel="Memory (MiB)", title="Likelihood evaluation memory usage",
    )
    for (j, J) in enumerate(n_components)
        lines!(ax, n_samples, ArrMediansTime[:, j])
        scatter!(ax, n_samples, ArrMediansTime[:, j], label="$J")
        lines!(ax2, n_samples, ArrMediansMemory[:, j])
        scatter!(ax2, n_samples, ArrMediansMemory[:, j], label="$J")
    end
    fig[2, 1:2] = Legend(
        fig[1, 1],
        ax,
        "Number of components",
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
        halign=:center, valign=:bottom,
        fontsize=10,
        framevisible=false
    )
    fig
    save("plots/"*data_label * "_likelihood_benchmarks_$(rev).pdf", fig, px_per_unit=0.5)
    save("plots/"*data_label * "_likelihood_benchmarks_$(rev).png", fig)
end
