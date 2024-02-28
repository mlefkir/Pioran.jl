@doc raw"""
    UniformDependent(a, b, c)

Multivariate distribution to model two random variables  where the first one is given by U[a,b] and the second one is given by U[x,c], 
where x is a random variable sampled from the first distribution. 

- `a`: lower bound of the first distribution
- `b`: upper bound of the first distribution
- `c`: upper bound of the second distribution


This means that the lower bound of the second distribution is dependent on the value of the first distribution.This is implemented to overcome the limitations of the current Turing's implementation for dependent priors with dynamic support.
See the following issues for more details: [[1]](https://github.com/TuringLang/Turing.jl/issues/1558),[[2]](https://github.com/TuringLang/Turing.jl/issues/1708),[[3]](https://github.com/TuringLang/Turing.jl/issues/1270).
    
# Example
```jldoctest
julia> using Pioran, Distributions
julia> d = UniformDependent(0, 1, 2)
UniformDependent(0.0, 1.0, 2.0)

julia> rand(d)
2-element Array{Float64,1}:
 0.123
 1.234
```
"""
struct UniformDependent <: ContinuousMultivariateDistribution
    a::Float64
    b::Float64
    c::Float64
end

function Distributions.rand(rng::Random.AbstractRNG, d::UniformDependent)
    x = rand(rng, Uniform(d.a, d.b))
    y = rand(rng, Uniform(x, d.c))
    return [x, y]
end

function Distributions.logpdf(d::UniformDependent, x::AbstractVector{<:Real})
    return logpdf(Uniform(d.a, d.b), x[1]) + logpdf(Uniform(x[1], d.c), x[2])
end

"""
    Bijectors.bijector(d::UniformDependent)
    
Create a bijector for the UniformDependent distribution. This is used to sample from the distribution using the Bijectors package.
Adapted from the following issues [[1]](https://github.com/TuringLang/Turing.jl/issues/1558),[[2]](https://github.com/TuringLang/Turing.jl/issues/1708),[[3]](https://github.com/TuringLang/Turing.jl/issues/1270).
"""
function Bijectors.bijector(d::UniformDependent)
    b1 = Bijectors.Stacked((Bijectors.TruncatedBijector([d.a], [d.b]), identity))
    m = Bijectors.PartitionMask(2, [2], [1])
    b2 = Bijectors.Coupling(x -> Bijectors.TruncatedBijector(x, d.c), m)
    return b1 ∘ b2
end

function Base.length(d::UniformDependent)
    return 2
end