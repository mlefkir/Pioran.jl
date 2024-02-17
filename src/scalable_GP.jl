
"""
Scalable Gaussian Processes

This module implements scalable Gaussian processes with a sum of semi-separable kernels.

The main type is `ScalableGP` which is a GP with a sum of semi-separable kernels.
"""
struct ScalableGP{Typef<:GP{<:AbstractGPs.ConstMean},Tk<:SumOfSemiSeparable} <: AbstractGPs.AbstractGP
    f::Typef
    kernel::Tk
end

# const FinitePosteriorGP = AbstractGPs.FiniteGP{<:ScalableGP}


ScalableGP(f::GP) = ScalableGP(f, f.kernel)
ScalableGP(kernel::SumOfSemiSeparable) = ScalableGP(GP(0.0, kernel), kernel)
ScalableGP(mean::Real, kernel::SumOfSemiSeparable) = ScalableGP(GP(mean, kernel), kernel)

const FiniteScalableGP = AbstractGPs.FiniteGP{<:ScalableGP}

struct PosteriorGP{Typef<:FiniteScalableGP,Ty<:AbstractVecOrMat{<:Real}} <: AbstractGPs.AbstractGP
    f::Typef
    y::Ty
end

""" posterior(f::ScalableGP, y::AbstractVecOrMat{<:Real})

Compute the posterior GP given the GP f and the data y.
"""
posterior(f::FiniteScalableGP, y::AbstractVecOrMat{<:Real}) = PosteriorGP(f, y)

""" 
    _predict_mean(fp::PosteriorGP, τ::AbstractVecOrMat{<:Real})

    Compute the Posterior mean of the GP at the points τ.
"""
function _predict_mean(fp::PosteriorGP, τ::AbstractVecOrMat{<:Real})
    K = fp.f.f.kernel
    x = fp.f.x
    y = fp.y
    σ2 = diag(fp.f.Σy)
    return predict(K, τ, x, y, σ2)
end

""" 
    _predict_cov(fp::PosteriorGP, τ::AbstractVecOrMat{<:Real})

    Compute the posterior covariance of the GP at the points τ.
"""
function _predict_cov(fp::PosteriorGP, τ::AbstractVecOrMat{<:Real})
    K = fp.f.f.kernel
    x = fp.f.x
    σ2 = diag(fp.f.Σy)
    return predict_cov(K, τ, x, σ2)
end

AbstractGPs.mean(fp::PosteriorGP, τ::AbstractVecOrMat{<:Real}) = _predict_mean(fp, τ)
AbstractGPs.mean(fp::PosteriorGP) = _predict_mean(fp, fp.f.x)
AbstractGPs.cov(fp::PosteriorGP, τ::AbstractVecOrMat{<:Real}) = _predict_cov(fp, τ)
AbstractGPs.cov(fp::PosteriorGP) = _predict_cov(fp, fp.f.x)

function AbstractGPs.rand(rng::AbstractRNG, fp::PosteriorGP,τ::AbstractVecOrMat{<:Real}, N::Int64=1)
    """
    rand(rng::AbstractRNG, f::PosteriorGP, τ::AbstractVecOrMat{<:Real}, N::Int=1)

    Sample N realisations from the posterior GP at the points x.
    """
    μ = mean(fp,τ)
    Σ = cov(fp,τ)
    post_dist = MvNormal(μ, Σ)

    return rand(rng, post_dist, N)
end

function AbstractGPs.rand(rng::AbstractRNG, fp::PosteriorGP, N::Int64=1)
    """
    rand(rng::AbstractRNG, f::PosteriorGP, N::Int=1)

    Sample N realisations from the posterior GP at the points x.
    """
    μ = mean(fp)
    Σ = cov(fp)
    post_dist = MvNormal(μ, Σ)

    return rand(rng, post_dist, N)
end

AbstractGPs.rand(fp::PosteriorGP, N::Int64) = AbstractGPs.rand(Random.GLOBAL_RNG, fp, N)
AbstractGPs.rand(fp::PosteriorGP,τ::AbstractVecOrMat{<:Real}, N::Int64) = AbstractGPs.rand(Random.GLOBAL_RNG, fp,τ, N)


function randScalableGP(rng::AbstractRNG, f::FiniteScalableGP)
    """
    rand(rng::AbstractRNG, f::ScalableGP)

    Sample a finite GP from the GP f at the points x.
    """
    σ2 = diag(f.Σy)

    return simulate(rng, f.f.kernel, f.x, σ2) .+ f.f.f.mean.c
end

function randScalableGP(rng::AbstractRNG, f::FiniteScalableGP,t::AbstractVecOrMat{<:Real})
    """
    rand(rng::AbstractRNG, f::ScalableGP)

    Sample a finite GP from the GP f at the points x.
    """
    σ2 = zeros(length(t))

    return simulate(rng, f.f.kernel, t, σ2) .+ f.f.f.mean.c
end

function AbstractGPs.rand(rng::AbstractRNG, f::FiniteScalableGP)
    """
    rand(rng::AbstractRNG, f::ScalableGP)

    Sample a finite GP from the GP f at the points x.
    """
    randScalableGP(rng, f)
end
function AbstractGPs.rand(rng::AbstractRNG, f::FiniteScalableGP,t::AbstractVecOrMat{<:Real})
    """
    rand(rng::AbstractRNG, f::ScalableGP)

    Sample a finite GP from the GP f at the points x.
    """
    randScalableGP(rng, f,t)
end

AbstractGPs.rand(f::FiniteScalableGP) = randScalableGP(Random.GLOBAL_RNG, f)
AbstractGPs.rand(f::FiniteScalableGP,t::AbstractVecOrMat{<:Real}) = randScalableGP(Random.GLOBAL_RNG, f,t)


function Distributions.logpdf(f::FiniteScalableGP, Y::AbstractVecOrMat{<:Real})
    """
    logpdf(f::ScalableGP, Y::AbstractVecOrMat{<:Real})

    Compute the log-likelihood of the data Y given the GP f.
    """
    σ2 = diag(f.Σy)
    y = Y .- f.f.f.mean.c
    return log_likelihood(f.f.kernel, f.x, y, σ2)
end