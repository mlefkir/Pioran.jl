module Pioran

export SHO, ScalableGP,SimpleBendingPowerLaw,approx,log_likelihood#,SemiSeparable,SumOfSemiSeparable
include("acvf.jl")
include("SHO.jl")
include("psd.jl")
include("scalable_GP.jl")
include("direct_solver.jl")
# Write your package code here.

end
