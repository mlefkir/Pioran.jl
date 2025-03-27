using Pioran: SemiSeparable

@doc raw"""
     SHO(A, ω₀, Q)

Simple Harmonic Oscillator covariance Function

- `A`: the amplitude of the covariance function
- `ω₀`: the angular frequency of the simple harmonic oscillator
- `Q`: the quality factor of the simple harmonic oscillator

```math
k(τ) = A \exp(-ω₀ τ / Q / 2) \left\{\begin{matrix} 2(1 + ω₀ τ) & Q = 1/2 \\ \cos(η ω₀ τ) + \frac{\sin(η ω₀ τ)}{2η Q} & Q < 1/2 \\ \cosh(η ω₀ τ) + \frac{\sinh(η ω₀ τ)}{2η Q} & Q \geq 1/2 \end{matrix}\right.\\
η = \sqrt{\left|1 - \frac{1}{4 Q^2}\right|}
```

See [Foreman-Mackey et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F) for more details.
"""
struct SHO <: SemiSeparable
    A
    ω₀
    Q
end

# Define the kernel functions for the SHO model
KernelFunctions.kappa(R::SHO, τ::Real) = SHO_covariance(τ, R.A, R.ω₀, R.Q)
KernelFunctions.metric(R::SHO) = Euclidean()
KernelFunctions.ScaledKernel(R::SHO, number::Real = 1.0) = SHO(number * R.A, R.ω₀, R.Q)

# Get the celerite coefficients of a SHO covariance function
function celerite_coefs(covariance::SHO)
    if covariance.Q == 1 / √2
        a = covariance.A
        b = a
        c = √2 / 2 * covariance.ω₀
        d = c
        return [a, b, c, d]
    else
        error("SHO with Q≠1/√2 not implemented yet")
    end
end

# Compute the covariance function for a SHO with parameters A, ω₀, Q at time τ.
function SHO_covariance(τ, A, ω₀, Q)
    term1 = A * exp.(-ω₀ .* τ / Q / 2)

    η = √(abs(1 - 1 / (4 * Q^2)))

    if Q == 1 / 2
        return term1 * 2 * (1 + ω₀ * τ)
    elseif Q >= 1 / 2
        return term1 * (cos.(η * ω₀ * τ) .+ sin.(η * ω₀ * τ) / (2η * Q))
    else
        return term1 * (cosh.(η * ω₀ * τ) .+ sinh.(η * ω₀ * τ) / (2η * Q))
    end
end
