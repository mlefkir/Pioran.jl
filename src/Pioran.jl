module Pioran

using Random
using LinearAlgebra
using KernelFunctions
using AbstractGPs
using Distributions
using Bijectors
using DelimitedFiles
using StatsBase
using ProgressMeter
using Polynomials
using StructArrays
using LoopVectorization
using Tonari
using CairoMakie
using VectorizedStatistics
using LombScargle

export SHO, Celerite, Exp, CARMA
export SingleBendingPowerLaw, DoubleBendingPowerLaw, PowerLaw, Lorentzian, QPO, approx
export get_covariance_from_psd, evaluate, CustomMean
export ScalableGP, posterior, log_likelihood, mean, cov, std
export quad2roots, roots2coeffs, run_diagnostics, run_posterior_predict_checks
export extract_subset, sample_approx_model
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
