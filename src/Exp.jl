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
KernelFunctions.metric(R::Exp) = KernelFunctions.Euclidean()
KernelFunctions.ScaledKernel(R::Exp, number::Real = 1.0) = Exp(number * R.A, R.α)

function celerite_coefs(covariance::Exp)
    a = covariance.A/2 # divide by two to ensure that the variance of the process is the integral from 0 to infinity, otherwise when we simulate timeseries, we get a sample variance that is twice the value of a.
    c = covariance.α
    return [a, 0.0, c, 0.0]
end


function Exp_covariance(τ, A, α)
    return A/2 * exp.(-α .* τ)
end


function Exp_psd(f, A, α)
    return 2A * α ./ (α^2 .+ 4π^2 * f .^ 2) # factor 2 IS NEEDED because of simulations! See celerite_coefs function
end

""" evaluate(R::Exp, f)

    This is the right formula but it disagrees with the Celerite implementation...

    Evaluate the power spectral density at frequency f
"""
function evaluate(R::Exp, f)
    return Exp_psd.(f, Ref(R.A), Ref(R.α))
end
