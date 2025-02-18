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
using StatsBase
using ProgressMeter
using Polynomials
using StructArrays
using LoopVectorization

export SHO, Celerite, Exp, CARMA
export SingleBendingPowerLaw, DoubleBendingPowerLaw, DoubleBendingPowerLaw_Bis, approx, TripleBendingPowerLaw
export ScalableGP, posterior, log_likelihood, mean, cov, std
export quad2roots, roots2coeffs
export run_diagnostics, run_posterior_predict_checks, extract_subset
export sample_approx_model
export TwoUniformDependent, TwoLogUniformDependent, ThreeUniformDependent

include("acvf.jl")
include("SHO.jl")
include("Celerite.jl")
include("Exp.jl")
include("psd.jl")
include("CARMA.jl")
include("celerite_solver.jl")
include("direct_solver.jl")
include("scalable_GP.jl")
include("utils.jl")
include("plots_diagnostics.jl")
include("priors.jl")
end
