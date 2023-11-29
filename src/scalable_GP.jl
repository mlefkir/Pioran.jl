using AbstractGPs
using LinearAlgebra
using Distributions
include("celerite_solver.jl")


struct ScalableGP{Typef<:GP{<:AbstractGPs.ConstMean},Tk<:SumOfSemiSeparable} <: AbstractGPs.AbstractGP
    f::Typef
    kernel::Tk
end

ScalableGP(f::GP) = ScalableGP(f, f.kernel)

ScalableGP(kernel::SumOfSemiSeparable) = ScalableGP(GP(0.0, kernel), kernel)
ScalableGP(mean::Real, kernel::SumOfSemiSeparable) = ScalableGP(GP(mean, kernel), kernel)

const FiniteScalableGP = AbstractGPs.FiniteGP{<:ScalableGP}

function Distributions.logpdf(f::FiniteScalableGP, Y::AbstractVecOrMat{<:Real})
    """
    logpdf(f::ScalableGP, Y::AbstractVecOrMat{<:Real})

    Compute the log-likelihood of the data Y given the GP f.
    """
    σ2_i = diag(f.Σy)
    y = Y .- f.f.f.mean.c
    return log_likelihood(f.f.kernel, f.x, y, σ2_i)
end