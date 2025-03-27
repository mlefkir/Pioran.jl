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


### There is still is a discrepancy between the PSD of EXP and Celerite!!!
### I need to fix this
function Exp_psd(f, A, α)
    return A * 2 * α ./ (α^2 .+4π^2 * f.^2)
end

""" calculate(f, C::Exp)
    
    This is the right formula but it disagrees with the Celerite implementation...

    Calculate the power spectral density at frequency f
"""
function calculate(f, R::Exp)
    return Exp_psd.(f, Ref(R.A), Ref(R.α))
end 