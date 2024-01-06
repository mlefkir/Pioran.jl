module Pioran
using Random
using LinearAlgebra
using KernelFunctions
using Distances
using AbstractGPs
using LinearAlgebra
using Distributions
using Random
using Turing
using DelimitedFiles

export SHO, ScalableGP, posterior, SimpleBendingPowerLaw, DoubleBendingPowerLaw,DoubleBendingPowerLaw_Bis, approx, log_likelihood, plot_mean_approx, plot_quantiles_approx, plot_boxplot_psd_approx,run_diagnostics#,SemiSeparable,SumOfSemiSeparable
include("acvf.jl")
include("SHO.jl")
include("psd.jl")
include("celerite_solver.jl")
include("direct_solver.jl")
include("scalable_GP.jl")
include("plots.jl")
include("plots_diagnostics.jl")
# Write your package code here.

end
