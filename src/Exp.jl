using Pioran: SemiSeparable

@doc raw"""
    Exp(A, α)

Exponential covariance Function

- `A`: the amplitude of the covariance function
- `α`: the decay rate of the covariance function

```math
k(τ) = A \exp(-α τ)
```

# Example
```julia
Exp(1.0, 0.25)
```
"""
struct Exp <: SemiSeparable
    A
    α
end

KernelFunctions.kappa(R::Exp, τ::Real) = Exp_covariance(τ, R.A, R.α)
KernelFunctions.metric(R::Exp) = Euclidean()
KernelFunctions.ScaledKernel(R::Exp, number::Real=1.0) = Exp(number * R.A, R.α)

function celerite_coefs(covariance::Exp)
    a = covariance.A
    c = covariance.α
    return [a, 0.0, c, 0.0]
end


function Exp_covariance(τ, A, α)
    return A * exp.(-α .* τ)
end