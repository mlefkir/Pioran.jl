
using Pioran: SemiSeparable
""" Celerite model """

"""
    Celerite(a, b, c, d)

Construct a celerite covariance function with parameters a, b, c, d.

"""
struct Celerite <: SemiSeparable
    a
    b
    c
    d
end

""" Define the kernel functions for the Celerite model """
KernelFunctions.kappa(R::Celerite, τ::Real) = Celerite_covariance(τ, R.a, R.b, R.c, R.d)
KernelFunctions.metric(R::Celerite) = Euclidean()
KernelFunctions.ScaledKernel(R::Celerite, number::Real=1.0) = Celerite(number * R.a, number * R.b, R.c, R.d)

"""
    Return the celerite coefficients for an Celerite covariance function.
"""

"""
    Return the celerite coefficients for an SHO covariance function.
"""
function celerite_coefs(covariance::Celerite)
    a = covariance.a
    b = covariance.b
    c = covariance.c
    d = covariance.d
    return [a, b, c, d]
end



"""
Celerite_covariance(τ,a,b,c,d)

Compute the model for a celerite covariance function with parameters a, b, c, d at time τ.

"""
function Celerite_covariance(τ, a, b, c, d)
    return exp.(-c .* τ) .* (a * cos.(d .* τ) .+ b * sin.(d .* τ))
end

