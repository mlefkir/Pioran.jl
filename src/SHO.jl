
using Pioran: SemiSeparable
""" SHO model """

"""
    SHO(σ, ω₀, Q)

Construct a simple harmonic oscillator covariance function with parameters σ, ω₀, Q.
Where σ is the amplitude, ω₀ is the angular frequency and Q is the quality factor.

"""
struct SHO <: SemiSeparable
    σ
    ω₀
    Q
end

""" Define the kernel functions for the SHO model """
KernelFunctions.kappa(R::SHO, τ::Real) = SHO_covariance(τ, R.σ, R.ω₀, R.Q)
KernelFunctions.metric(R::SHO) = Euclidean()
KernelFunctions.ScaledKernel(R::SHO, number::Real=1.0) = SHO(number * R.σ, R.ω₀, R.Q)

"""
    Return the celerite coefficients for an SHO covariance function.
"""
function celerite_coefs(covariance::SHO)
    if covariance.Q == 1 / √2
        a = covariance.σ
        b = a
        c = √2 / 2 * covariance.ω₀
        d = √2 / 2 * covariance.ω₀
        return [a, b, c, d]
    end
end



"""
SHO_covariance(τ, σ, ω₀, Q)

Compute the model for a simple harmonic oscillator with parameters σ, ω₀, Q at time τ.
Where σ is the amplitude, ω₀ is the angular frequency and Q is the quality factor.

"""
function SHO_covariance(τ, σ, ω₀, Q)
    term1 = σ * exp.(-ω₀ .* τ / Q / 2)

    η = √(abs(1 - 1 / (4 * Q^2)))

    if Q == 1 / 2
        return term1 * 2 * (1 + ω₀ * τ)
    elseif Q >= 1 / 2
        return term1 * (cos.(η * ω₀ * τ) .+ sin.(η * ω₀ * τ) / (2η * Q))
    else
        return term1 * (cosh.(η * ω₀ * τ) .+ sinh.(η * ω₀ * τ) / (2η * Q))
    end
end

