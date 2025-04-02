using Plots, Pioran

f0, fM = 1.0e-4, 1.0e4

𝓟 = SingleBendingPowerLaw(0.4, 1.0e-1, 3.0)
f = 10 .^ range(log10(f0), stop = log10(fM), length = 1000)
psd_approx = Pioran.approximated_psd(f, 𝓟, f0, fM, n_components = 20, basis_function = "SHO", individual = true)
# psd_approxDR = Pioran.approximated_psd(f, 𝓟, f0, fM, n_components=20,basis_function="DRWCelerite",individual=true)


𝓓 = DoubleBendingPowerLaw(0.4, 1.0e-2, 1.5, 97.3, 3.1)
psd_approx_DBPL = Pioran.approximated_psd(f, 𝓓, f0, fM, n_components = 20, basis_function = "SHO", individual = true)
# psd_approxDR = Pioran.approximated_psd(f, 𝓓, f0, fM, n_components=20,basis_function="DRWCelerite",individual=true)

# l = @layout [a b]
p1 = plot(f, f .* 𝓟(f) / 𝓟(f0), xscale = :log10, yscale = :log10, label = "Model", lw = 2, color = :dodgerblue, title = "Single bending power-law")
p1 = plot!(f, f .* sum(psd_approx, dims = 2), xscale = :log10, yscale = :log10, label = "Approximation", xlabel = "Frequency", ylabel = "f x Power spectrum", lw = 1.5, color = :black, ls = :dash)

p1 = plot!(f, f .* psd_approx, xscale = :log10, yscale = :log10, label = nothing, lw = 1, ls = :dash, alpha = 0.5, color = :dodgerblue)
plot!([], [], label = "Basis function", lw = 1, ls = :dash, color = :dodgerblue, alpha = 0.5)

p2 = plot(f, f .* 𝓓(f) / 𝓓(f0), xscale = :log10, yscale = :log10, label = "Model", xlabel = "Frequency", lw = 2, color = :darkorchid, title = "Double bending power-law")
p2 = plot!(f, f .* sum(psd_approx_DBPL, dims = 2), xscale = :log10, yscale = :log10, label = "Approximation", xlabel = "Frequency", lw = 1.5, color = :pink, ls = :dash)

p2 = plot!(f, f .* psd_approx_DBPL, xscale = :log10, yscale = :log10, label = nothing, lw = 1, ls = :dash, color = :darkorchid, alpha = 0.5)
plot!([], [], label = "Basis function", lw = 1, ls = :dash, color = :darkorchid, alpha = 0.5, yformatter = Returns(""))
p = plot(p1, p2, layout = (1, 2), size = (800, 300), ylims = (1.0e-10, 0.1), framestyle = :box, legend_foreground_color = :white, grid = :off, minorticks = 10, link = :y, dpi = 300, left_margin = [5Plots.mm -3Plots.mm], bottom_margin = 5Plots.mm)
savefig("approximation.svg")
