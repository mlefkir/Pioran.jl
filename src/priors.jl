using Bijectors

# COV_EXCL_START
@doc raw"""
    TwoUniformDependent(a, b, c, ϵ)
    TwoUniformDependent(a, b, c) (constructor with default ϵ = 1e-10)

Multivariate distribution to model two random variables  where the first one is given by U[a,b] and the second one is given by U[x,c],
where x is a random variable sampled from the first distribution.

- `a`: lower bound of the first distribution
- `b`: upper bound of the first distribution
- `c`: upper bound of the second distribution
- `ϵ`: small value to make sure that the lower and upper bounds of each distribution are different

This means that the lower bound of the second distribution is dependent on the value of the first distribution.This is implemented to overcome the limitations of the current Turing's implementation for dependent priors with dynamic support.
See the following issues for more details: [[1]](https://github.com/TuringLang/Turing.jl/issues/1558),[[2]](https://github.com/TuringLang/Turing.jl/issues/1708),[[3]](https://github.com/TuringLang/Turing.jl/issues/1270).

# Example
```jldoctest
julia> using Pioran, Distributions
julia> d = TwoUniformDependent(0, 1, 2)
TwoUniformDependent(0.0, 1.0, 2.0)

julia> rand(d)
2-element Array{Float64,1}:
 0.123
 1.234
```
"""
struct TwoUniformDependent <: ContinuousMultivariateDistribution
    a::Float64
    b::Float64
    c::Float64
    ϵ::Float64

    function TwoUniformDependent(a, b, c, ϵ)
        return if a >= b
            throw(ArgumentError("a must be less than b"))
        elseif b >= c
            throw(ArgumentError("b must be less than c"))
        else
            new(a, b, c, ϵ)
        end
    end
end
TwoUniformDependent(a, b, c) = TwoUniformDependent(a, b, c, 1.0e-10)

@doc raw"""
    ThreeUniformDependent(a, b, c, ϵ)
    ThreeUniformDependent(a, b, c) (constructor with default ϵ = 1e-10)

Multivariate distribution to model three random variables  where the first one x1 is given by U[a,b] and the second one x2 is given by U[x1,c] and the
third one x3 is given by U[x2,c]. where a<b<c.

- `a`: lower bound of the first distribution
- `b`: upper bound of the first distribution
- `c`: upper bound of the second and third distribution
- `ϵ`: small value to make sure that the lower and upper bounds of each distribution are different

This means that the lower bound of the second distribution is dependent on the value of the first distribution and so on... This is implemented to overcome the limitations of the current Turing's implementation for dependent priors with dynamic support.
See the following issues for more details: [[1]](https://github.com/TuringLang/Turing.jl/issues/1558),[[2]](https://github.com/TuringLang/Turing.jl/issues/1708),[[3]](https://github.com/TuringLang/Turing.jl/issues/1270).

"""
struct ThreeUniformDependent <: ContinuousMultivariateDistribution
    a::Float64
    b::Float64
    c::Float64
    ϵ::Float64
    function ThreeUniformDependent(a, b, c, ϵ)
        return if a >= b
            throw(ArgumentError("a must be less than b"))
        elseif b >= c
            throw(ArgumentError("b must be less than c"))
        else
            new(a, b, c, ϵ)
        end
    end
end
ThreeUniformDependent(a, b, c) = ThreeUniformDependent(a, b, c, 1.0e-10)

@doc raw"""
    TwoLogUniformDependent(a, b, ϵ)
    TwoLogUniformDependent(a, b) (constructor with default ϵ = 1e-10

Multivariate distribution to model three random variables  where the first one x1 is given by log-U[a,b] and the second one x2 is given by log-U[x1,b].

- `a`: lower bound of the first distribution
- `b`: upper bound of the first distribution
- `ϵ`: small value to make sure that the lower and upper bounds of each distribution are different

This means that the lower bound of the second distribution is dependent on the value of the first distribution. This is implemented to overcome the limitations of the current Turing's implementation for dependent priors with dynamic support.
See the following issues for more details: [[1]](https://github.com/TuringLang/Turing.jl/issues/1558),[[2]](https://github.com/TuringLang/Turing.jl/issues/1708),[[3]](https://github.com/TuringLang/Turing.jl/issues/1270).

"""
struct TwoLogUniformDependent <: ContinuousMultivariateDistribution
    a::Float64
    b::Float64
    ϵ::Float64
    function TwoLogUniformDependent(a, b, ϵ)
        return if a == 0 || b == 0
            throw(ArgumentError("a and b must be greater than 0"))
        elseif a >= b
            throw(ArgumentError("a must be less than b"))
        else
            new(a, b, ϵ)
        end

    end
end
TwoLogUniformDependent(a, b) = TwoLogUniformDependent(a, b, 1.0e-10)

function Distributions.rand(rng::Random.AbstractRNG, d::TwoUniformDependent)
    x = rand(rng, Uniform(d.a, d.b))
    y = rand(rng, Uniform(x, d.c))
    return [x, y]
end

function Distributions.rand(rng::Random.AbstractRNG, d::ThreeUniformDependent)
    x₁ = rand(rng, Uniform(d.a, d.b))
    x₂ = rand(rng, Uniform(x₁, d.c + d.ϵ))
    x₃ = rand(rng, Uniform(x₂, d.c + d.ϵ))
    return [x₁, x₂, x₃]
end

function Distributions.rand(rng::Random.AbstractRNG, d::TwoLogUniformDependent)
    x₁ = rand(rng, LogUniform(d.a, d.b))
    x₂ = rand(rng, LogUniform(x₁, d.b + d.ϵ))
    return [x₁, x₂]
end

function Distributions.logpdf(d::ThreeUniformDependent, x::AbstractVector{<:Real})
    return logpdf(Uniform(d.a, d.b), x[1]) + logpdf(Uniform(x[1], d.c + d.ϵ), x[2]) + logpdf(Uniform(x[2], d.c + d.ϵ), x[3])
end

function Distributions.logpdf(d::TwoUniformDependent, x::AbstractVector{<:Real})
    return logpdf(Uniform(d.a, d.b), x[1]) + logpdf(Uniform(x[1], d.c), x[2])
end

function Distributions.logpdf(d::TwoLogUniformDependent, x::AbstractVector{<:Real})
    return logpdf(LogUniform(d.a, d.b), x[1]) + logpdf(LogUniform(x[1], d.b + d.ϵ), x[2])
end

"""
    Bijectors.bijector(d::TwoUniformDependent)

Create a bijector for the TwoUniformDependent distribution. This is used to sample from the distribution using the Bijectors package.
Adapted from the following issues [[1]](https://github.com/TuringLang/Turing.jl/issues/1558),[[2]](https://github.com/TuringLang/Turing.jl/issues/1708),[[3]](https://github.com/TuringLang/Turing.jl/issues/1270).
"""
function Bijectors.bijector(d::TwoUniformDependent)
    b1 = Bijectors.Stacked((Bijectors.TruncatedBijector([d.a], [d.b]), identity))
    m = Bijectors.PartitionMask(2, [2], [1])
    b2 = Bijectors.Coupling(x -> Bijectors.TruncatedBijector(x, d.c), m)
    return b1 ∘ b2
end

function Bijectors.bijector(d::ThreeUniformDependent)
    b1 = Bijectors.Stacked((Bijectors.TruncatedBijector([d.a], [d.b]), identity, identity))
    m = Bijectors.PartitionMask(3, [2], [1])
    b2 = Bijectors.Coupling(x -> Bijectors.TruncatedBijector(x, d.c), m)
    m2 = Bijectors.PartitionMask(3, [3], [2])
    b3 = Bijectors.Coupling(x -> Bijectors.TruncatedBijector(x, d.c), m2)
    return b1 ∘ b2 ∘ b3
end

function Bijectors.bijector(d::TwoLogUniformDependent)
    b1 = Bijectors.Stacked((Bijectors.TruncatedBijector([d.a], [d.b]), identity))
    m = Bijectors.PartitionMask(2, [2], [1])
    b2 = Bijectors.Coupling(x -> Bijectors.TruncatedBijector(x, d.b), m)
    return b1 ∘ b2
end

function Base.length(d::TwoUniformDependent)
    return 2
end

function Base.length(d::ThreeUniformDependent)
    return 3
end

function Base.length(d::TwoLogUniformDependent)
    return 2
end
# COV_EXCL_STOP
