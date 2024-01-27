
using Pioran: SemiSeparable
""" Exp model """

"""
    Exp(σ,α)

Construct a exponential covariance function with parameters σ, α.
Where σ is the amplitude and α is the decay rate.

"""
struct Exp <: SemiSeparable
    σ
    α
end

""" Define the kernel functions for the Exp model """
KernelFunctions.kappa(R::Exp, τ::Real) = Exp_covariance(τ, R.σ, R.α)
KernelFunctions.metric(R::Exp) = Euclidean()
KernelFunctions.ScaledKernel(R::Exp, number::Real=1.0) = Exp(number * R.σ, R.α)

"""
    Return the celerite coefficients for an Exp covariance function.
"""
function celerite_coefs(covariance::Exp)
    a = covariance.σ
    c = covariance.α
    return [a, 0.0, c, 0.0]
end


"""
Exp_covariance(τ, σ, α)

Compute the covariance function for an exponential with parameters σ, α at time τ.
Where σ is the amplitude and α is the decay rate.
"""
function Exp_covariance(τ, σ, α)
    return σ * exp.(-α .* τ)
end