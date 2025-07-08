using Pioran: SemiSeparable

@doc raw"""
     Celerite(a, b, c, d)

Celerite covariance Function

- `a`: the amplitude of the first term
- `b`: the amplitude of the second term
- `c`: the decay rate of the covariance function
- `d`: the `period` of the covariance function

```math
k(τ) = \exp(-c τ) (a \cos(d τ) + b \sin(d τ))
```

See [Foreman-Mackey et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F) for more details.

"""
struct Celerite <: SemiSeparable
    a
    b
    c
    d
end

# Define the kernel functions for the Celerite model
KernelFunctions.kappa(R::Celerite, τ::Real) = Celerite_covariance(τ, R.a, R.b, R.c, R.d)
KernelFunctions.metric(R::Celerite) = KernelFunctions.Euclidean()
KernelFunctions.ScaledKernel(R::Celerite, number::Real = 1.0) = Celerite(number * R.a, number * R.b, R.c, R.d)

# Get the celerite coefficients of a Celerite covariance function
function celerite_coefs(covariance::Celerite)
    a = covariance.a
    b = covariance.b
    c = covariance.c
    d = covariance.d
    return [a, b, c, d]
end

# Compute the covariance function for a Celerite with parameters a, b, c, d at time τ.
function Celerite_covariance(τ, a, b, c, d)
    return exp.(-c .* τ) .* (a * cos.(d .* τ) .+ b * sin.(d .* τ))
end

function Celerite_psd(f, a, b, c, d)
    ω = 2π * f
    num = (a * c + b * d) * (c^2 + d^2) .+ (a * c - b * d) * ω .^ 2
    den = ω .^ 4 + 2 * (c^2 - d^2) * ω .^ 2 .+ (c^2 + d^2)^2
    return num ./ den
end

""" evaluate(f, C::Celerite)

    evaluate the power spectral density at frequency f
"""
function evaluate(C::Celerite, f)
    return Celerite_psd.(f, Ref(C.a), Ref(C.b), Ref(C.c), Ref(C.d))
end

function Celerite_psd(cov::Celerite, f)
    return Celerite_psd(f, cov.a, cov.b, cov.c, cov.d)
end
