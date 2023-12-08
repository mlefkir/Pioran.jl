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

export SHO, ScalableGP, posterior, SimpleBendingPowerLaw, DoubleBendingPowerLaw, approx, log_likelihood#,SemiSeparable,SumOfSemiSeparable
include("acvf.jl")
include("SHO.jl")
include("psd.jl")
include("celerite_solver.jl")
include("direct_solver.jl")
include("scalable_GP.jl")
include("plots.jl")
# Write your package code here.

end
