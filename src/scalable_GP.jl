
"""
A scalable Gaussian process has a covariance function formed of semi-separable kernels


# Example
```julia
using Pioran
𝓟 = SingleBendingPowerLaw(1.0, 1.0, 2.0)
𝓡 = approx(𝓟, 1e-4, 1e-1, 30, 2.31,basis_function="SHO")
μ = 1.2

f = ScalableGP(𝓡) # zero-mean GP
f = ScalableGP(μ, 𝓡) # with mean μ
```
See [Foreman-Mackey et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F) for more details.
"""
struct ScalableGP{Typef<:GP{<:AbstractGPs.ConstMean},Tk<:SemiSeparable} <: AbstractGPs.AbstractGP
    f::Typef
    kernel::Tk
end

# const FinitePosteriorGP = AbstractGPs.FiniteGP{<:ScalableGP}

ScalableGP(f::GP) = ScalableGP(f, f.kernel)
ScalableGP(kernel::SemiSeparable) = ScalableGP(GP(0.0, kernel), kernel)
ScalableGP(mean::Real, kernel::SemiSeparable) = ScalableGP(GP(mean, kernel), kernel)

const FiniteScalableGP = AbstractGPs.FiniteGP{<:ScalableGP}

struct PosteriorGP{Typef<:FiniteScalableGP,Ty<:AbstractVecOrMat{<:Real}} <: AbstractGPs.AbstractGP
    f::Typef
    y::Ty
end

"""
    posterior(f::ScalableGP, y::AbstractVecOrMat{<:Real})

Compute the posterior Gaussian process `fp` given the GP `f` and the data `y`.
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

"""
    mean(fp::PosteriorGP, τ::AbstractVecOrMat{<:Real})
    mean(fp::PosteriorGP)

Compute the mean of the posterior GP at the points τ.

"""
AbstractGPs.mean(fp::PosteriorGP, τ::AbstractVecOrMat{<:Real}) = _predict_mean(fp, τ)
AbstractGPs.mean(fp::PosteriorGP) = _predict_mean(fp, fp.f.x)
"""
    cov(fp::PosteriorGP, τ::AbstractVecOrMat{<:Real})
    cov(fp::PosteriorGP)

Compute the covariance of the posterior GP at the points τ.
"""
AbstractGPs.cov(fp::PosteriorGP, τ::AbstractVecOrMat{<:Real}) = _predict_cov(fp, τ)
AbstractGPs.cov(fp::PosteriorGP) = _predict_cov(fp, fp.f.x)
"""
    std(fp::PosteriorGP, τ::AbstractVecOrMat{<:Real})
    std(fp::PosteriorGP)

Compute the standard deviation of the posterior GP at the points τ.
"""
AbstractGPs.std(fp::PosteriorGP) = sqrt.(diag(_predict_cov(fp, fp.f.x)))
AbstractGPs.std(fp::PosteriorGP, τ::AbstractVecOrMat{<:Real}) = sqrt.(diag(_predict_cov(fp, τ)))

function AbstractGPs.rand(rng::AbstractRNG, fp::PosteriorGP, τ::AbstractVecOrMat{<:Real}, N::Int64=1)
    μ = mean(fp, τ)
    Σ = cov(fp, τ)
    post_dist = MvNormal(μ, Σ)

    return rand(rng, post_dist, N)
end

"""
    rand(rng::AbstractRNG, fp::PosteriorGP, N::Int=1)
    rand(rng::AbstractRNG, fp::PosteriorGP, τ::AbstractVecOrMat{<:Real}, N::Int=1)
    rand(fp::PosteriorGP, N::Int=1)

Sample `N` realisations from the posterior GP `fp` at the points `τ`.
"""
function AbstractGPs.rand(rng::AbstractRNG, fp::PosteriorGP, N::Int64=1)
    μ = mean(fp)
    Σ = cov(fp)
    post_dist = MvNormal(μ, Σ)

    return rand(rng, post_dist, N)
end

AbstractGPs.rand(fp::PosteriorGP, N::Int64) = AbstractGPs.rand(Random.GLOBAL_RNG, fp, N)
AbstractGPs.rand(fp::PosteriorGP, τ::AbstractVecOrMat{<:Real}, N::Int64) = AbstractGPs.rand(Random.GLOBAL_RNG, fp, τ, N)
AbstractGPs.rand(fp::PosteriorGP, τ::AbstractVecOrMat{<:Real}) = AbstractGPs.rand(Random.GLOBAL_RNG, fp, τ, 1)


function randScalableGP(rng::AbstractRNG, f::FiniteScalableGP)
    σ2 = diag(f.Σy)
    return simulate(rng, f.f.kernel, f.x, σ2) .+ f.f.f.mean.c
end

function randScalableGP(rng::AbstractRNG, f::FiniteScalableGP, t::AbstractVecOrMat{<:Real})
    σ2 = zeros(length(t))
    return simulate(rng, f.f.kernel, t, σ2) .+ f.f.f.mean.c
end

"""
    rand(rng::AbstractRNG, f::ScalableGP)
    rand(rng::AbstractRNG, f::ScalableGP, t::AbstractVecOrMat{<:Real})
    rand(f::ScalableGP)
    rand(f::ScalableGP, t::AbstractVecOrMat{<:Real})

Draw a realisation from the GP `f` at the points `t`.
"""
AbstractGPs.rand(rng::AbstractRNG, f::FiniteScalableGP) = randScalableGP(rng, f)
AbstractGPs.rand(rng::AbstractRNG, f::FiniteScalableGP, t::AbstractVecOrMat{<:Real}) = randScalableGP(rng, f, t)
AbstractGPs.rand(f::FiniteScalableGP) = randScalableGP(Random.GLOBAL_RNG, f)
AbstractGPs.rand(f::FiniteScalableGP, t::AbstractVecOrMat{<:Real}) = randScalableGP(Random.GLOBAL_RNG, f, t)

"""
    logpdf(f::ScalableGP, Y::AbstractVecOrMat{<:Real})

Compute the log-likelihood of the data Y given the GP f.
"""
function Distributions.logpdf(f::FiniteScalableGP, Y::AbstractVecOrMat{<:Real})
    σ2 = diag(f.Σy)
    y = Y .- f.f.f.mean.c
    return log_likelihood(f.f.kernel, f.x, y, σ2)
end