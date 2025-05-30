""" Covariance functions for Gaussian processes."""

"""
    SemiSeparable

Abstract type for semi-separable covariance functions.
"""
abstract type SemiSeparable <: KernelFunctions.SimpleKernel end
abstract type SumOfTerms <: SemiSeparable end
"""
    SumOfSemiSeparable

Abstract type for sum of semi-separable covariance functions.
It stores the individual covariance functions and the celerite coefficients (a,b,c,d).
"""
struct SumOfSemiSeparable{Tcov <: Vector{<:SemiSeparable}} <: SumOfTerms
    cov::Tcov
    a::AbstractVector
    b::AbstractVector
    c::AbstractVector
    d::AbstractVector
end

"""
    SumOfCelerite{Tcov <: Vector{<:Celerite},T<:Real} <: SemiSeparable

Represents the sum of celerite covariance functions.
It appears to be faster than the SumOfSemiSeparable model but more restrictive as
the covariance functions must all be celerite.

Constructor:
    SumOfCelerite(cov::StructArray{Celerite}(a,b,c,d))
    SumOfCelerite(a, b, c, d)
"""
struct SumOfCelerite{Tcov <: StructArray{<:SemiSeparable}} <: SumOfTerms
    cov::Tcov
    a::AbstractVector
    b::AbstractVector
    c::AbstractVector
    d::AbstractVector

    function SumOfCelerite(cov::Tcov) where {Tcov <: StructArray{<:SemiSeparable}}
        return new{Tcov}(cov, cov.a, cov.b, cov.c, cov.d)
    end

    function SumOfCelerite(a::AbstractVector, b::AbstractVector, c::AbstractVector, d::AbstractVector)
        return new{StructArray{<:SemiSeparable}}(StructArray{Celerite}((a, b, c, d)), a, b, c, d)
    end
end

function celerite_coefs(covariance::SumOfCelerite)
    return covariance.a, covariance.b, covariance.c, covariance.d
end

"""
     +(::SemiSeparable, ::SemiSeparable)

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

            a_, b_, c_, d_ = cov1.a, cov1.b, cov1.c, cov1.d
            a, b, c, d = similar(a_, length(a_) + length(a_2)), similar(b_, length(b_) + length(b_2)), similar(c_, length(c_) + length(c_2)), similar(d_, length(d_) + length(d_2))
            for i in range(1, length(a_1))
                a[i] = a_1[i]
                b[i] = b_1[i]
                c[i] = c_1[i]
                d[i] = d_1[i]
            end

            for i in range(1, length(a_2))
                a[i + length(a_1)] = a_2[i]
                b[i + length(b_1)] = b_2[i]
                c[i + length(c_1)] = c_2[i]
                d[i + length(d_1)] = d_2[i]
            end
            # append!(a, a_2)
            # append!(b, b_2)
            # append!(c, c_2)
            # append!(d, d_2)
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
     celerite_coefs(covariance)

    Get the celerite coefficients
"""
function celerite_coefs(covariance::SumOfSemiSeparable)
    J = length(covariance.cov)
    a_1, _, _, _ = celerite_coefs(covariance.cov[1])
    T = eltype(a_1)
    a, b, c, d = zeros(T, J), zeros(T, J), zeros(T, J), zeros(T, J)

    @inbounds for j in 1:J
        a[j], b[j], c[j], d[j] = celerite_coefs(covariance.cov[j])
    end
    return a, b, c, d
end

# Define the kernel functions for the SumOfSemiSeparable model
KernelFunctions.metric(R::SumOfSemiSeparable) = KernelFunctions.Euclidean()

## For the SumOfCelerite model
# Define the kernel functions for the SumOfCelerite model
KernelFunctions.metric(R::SumOfTerms) = KernelFunctions.Euclidean()

# Define the kernel functions for the SumOfCelerite model
function KernelFunctions.kappa(R::SumOfTerms, τ::Real)
    return sum(map(x -> KernelFunctions.kappa(x, τ), R.cov))
end

# Define the kernel functions for the SumOfSemiSeparable model
function KernelFunctions.ScaledKernel(R::SumOfSemiSeparable, number::Real = 1.0)

    J = length(R.cov)
    for j in 1:J
        R.cov[j] = ScaledKernel(R.cov[j], number)
    end

    return SumOfSemiSeparable(R.cov, number * R.a, number * R.b, R.c, R.d)
end

# Define the kernel functions for the SumOfCelerite model
function KernelFunctions.ScaledKernel(R::SumOfCelerite, number::Real = 1.0)
    return SumOfCelerite(map(x -> ScaledKernel(x, number), R.cov))
end
