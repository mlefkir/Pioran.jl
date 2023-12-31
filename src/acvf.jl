""" Covariance functions for Gaussian processes."""

using KernelFunctions
using Distances


"""
    SemiSeparable

Abstract type for semi-separable covariance functions.
"""
abstract type SemiSeparable <: KernelFunctions.SimpleKernel end

"""
    SumOfSemiSeparable

Abstract type for sum of semi-separable covariance functions.
It stores the individual covariance functions and the celerite coefficients (a,b,c,d).
"""
struct SumOfSemiSeparable{Tcov<:Vector{<:SemiSeparable},T<:Vector{<:Real}} <: SemiSeparable
    cov::Tcov
    a::T
    b::T
    c::T
    d::T

end

"""
    Sum of semi-separable covariance functions
"""
function Base.:+(cov1::SemiSeparable, cov2::SemiSeparable)
    a_1, b_1, c_1, d_1 = celerite_coefs(cov1)
    a_2, b_2, c_2, d_2 = celerite_coefs(cov2)
    if !(cov1 isa SumOfSemiSeparable) && !(cov2 isa SumOfSemiSeparable)
        cov = [cov1; cov2]
        a = [a_1; a_2]
        b = [b_1; b_2]
        c = [c_1; c_2]
        d = [d_1; d_2]
    elseif (cov1 isa SumOfSemiSeparable && cov2 isa SumOfSemiSeparable)
        cov = [cov1.cov; cov2.cov]
        a = vcat(a_1, a_2)
        b = vcat(b_1, b_2)
        c = vcat(c_1, c_2)
        d = vcat(d_1, d_2)

    else
        if (cov1 isa SumOfSemiSeparable)
            cov = [cov1.cov; cov2]

            a, b, c, d = cov1.a, cov1.b, cov1.c, cov1.d
            append!(a, a_2)
            append!(b, b_2)
            append!(c, c_2)
            append!(d, d_2)
        else
            cov = [cov1; cov2.cov]

            a, b, c, d = cov2.a, cov2.b, cov2.c, cov2.d
            append!(a, a_1)
            append!(b, b_1)
            append!(c, c_1)
            append!(d, d_1)
        end
    end

    return SumOfSemiSeparable(cov, a, b, c, d)
end

"""
    Return the celerite coefficients for a semi-separable covariance function.
"""
function celerite_coefs(covariance::SumOfSemiSeparable)
    J = length(covariance.cov)
    a_1, b_1, c_1, d_1 = celerite_coefs(covariance.cov[1])
    T = eltype(a_1)
    a, b, c, d = zeros(T, J), zeros(T, J), zeros(T, J), zeros(T, J)

    @inbounds for j in 1:J
        a[j], b[j], c[j], d[j] = celerite_coefs(covariance.cov[j])
    end
    return a, b, c, d
end

"""Define the metric for the semi-separable covariance function."""
KernelFunctions.metric(R::SumOfSemiSeparable) = Euclidean()

"""Define the total covariance function for the semi-separable covariance function."""
function KernelFunctions.kappa(R::SumOfSemiSeparable, τ::Real)
    J = length(R.cov)
    K = KernelFunctions.kappa(R.cov[1], τ)
    for j in 2:J
        K += KernelFunctions.kappa(R.cov[j], τ)
    end
    return K
end

"""Define the scaled semi-separable covariance function."""
function KernelFunctions.ScaledKernel(R::SumOfSemiSeparable, number::Real=1.0)

    J = length(R.cov)
    for j in 1:J
        R.cov[j] = ScaledKernel(R.cov[j], number)
    end

    return SumOfSemiSeparable(R.cov, number * R.a, number * R.b, R.c, R.d)
end