using AbstractGPs
using LinearAlgebra
using Distributions
using Random
include("celerite_solver.jl")


struct ScalableGP{Typef<:GP{<:AbstractGPs.ConstMean},Tk<:SumOfSemiSeparable} <: AbstractGPs.AbstractGP
    f::Typef
    kernel::Tk
end

ScalableGP(f::GP) = ScalableGP(f, f.kernel)

ScalableGP(kernel::SumOfSemiSeparable) = ScalableGP(GP(0.0, kernel), kernel)
ScalableGP(mean::Real, kernel::SumOfSemiSeparable) = ScalableGP(GP(mean, kernel), kernel)

const FiniteScalableGP = AbstractGPs.FiniteGP{<:ScalableGP}

function randScalableGP(rng::AbstractRNG, f::FiniteScalableGP)
    """
    rand(rng::AbstractRNG, f::ScalableGP, x::AbstractVector{<:Real})

    Sample a finite GP from the GP f at the points x.
    """
    σ2 = diag(f.Σy)

    return simulate(rng,f.f.kernel, f.x, σ2) .+ f.f.f.mean.c
end

function AbstractGPs.rand(rng::AbstractRNG, f::FiniteScalableGP)
    """
    rand(rng::AbstractRNG, f::ScalableGP, x::AbstractVector{<:Real})

    Sample a finite GP from the GP f at the points x.
    """
    randScalableGP(rng, f)
end


AbstractGPs.rand(f::FiniteScalableGP) = randScalableGP(Random.GLOBAL_RNG, f)
    

function Distributions.logpdf(f::FiniteScalableGP, Y::AbstractVecOrMat{<:Real})
    """
    logpdf(f::ScalableGP, Y::AbstractVecOrMat{<:Real})

    Compute the log-likelihood of the data Y given the GP f.
    """
    σ2 = diag(f.Σy)
    y = Y .- f.f.f.mean.c
    return log_likelihood(f.f.kernel, f.x, y, σ2)
end