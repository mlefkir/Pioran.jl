@doc raw""" 
    CARMA(p, q, rα, β, σ²)

Continuous-time AutoRegressive Moving Average (CARMA) model for the power spectral density

- `p`: the order of the autoregressive polynomial
- `q`: the order of the moving average polynomial
- `rα`: roots of the autoregressive polynomial length p+1
- `β`: the moving average coefficients length q+1
- `σ²`: the variance of the process

```math
\mathcal{P}(f) = \sigma^2 \left|\dfrac{\sum\limits_{k=0}^q \beta_k \left(2\pi\mathrm{i}f\right)^k }{\sum\limits_{l=0}^p \alpha_l \left(2\pi\mathrm{i}f\right)^l}\right|^2
```

"""
struct CARMA{Tp<:Int64,Trα<:Complex,Tβ<:Real,T<:Real} <: SemiSeparable
    p::Tp
    q::Tp
    rα::Vector{Trα} # roots of AR polynomial length p
    β::Vector{Tβ} # moving average coefficients length q+1
    σ²::T

    function CARMA(p::Tp, q::Tp, rα::Vector{Trα}, β::Vector{Tβ}, σ²::T) where {Tp<:Int64,Trα<:Complex,Tβ<:Real,T<:Real}
        if p < 1 || q < 0
            throw(ArgumentError("The order of the autoregressive and moving average polynomials must be positive"))
        elseif q > p
            throw(ArgumentError("The order of the moving average polynomial must be less than or equal to the order of the autoregressive polynomial"))
        elseif length(rα) != p
            throw(ArgumentError("The length of the roots of the autoregressive polynomial must be equal to the order of the autoregressive polynomial"))
        elseif length(β) != q + 1
            throw(ArgumentError("The length of the moving average coefficients must be equal to q + 1"))
        end
        new{Tp,Trα,Tβ,T}(p, q, rα, β, σ²)
    end

end
CARMA(p::Int64, q::Int64, rα, β) = CARMA(p, q, rα, β, 1.0)

# Define the kernel functions for the CARMA model
KernelFunctions.kappa(R::CARMA, τ::Real) = CARMA_covariance(τ, R)
KernelFunctions.metric(R::CARMA) = Euclidean()
KernelFunctions.ScaledKernel(R::CARMA, number::Real=1.0) = CARMA(R.p, R.q, R.rα, R.β, R.σ² * number)

"""
    celerite_repr(cov::CARMA)

Convert a CARMA model to a Celerite model.

"""
function celerite_repr(cov::CARMA)
    a, b, c, d = celerite_coefs(cov)

    cov = Celerite(a[1], b[1], c[1], d[1])
    for i in 2:length(a)
        cov += Celerite(a[i], b[i], c[i], d[i])
    end
    return cov
end



function celerite_coefs(covariance::CARMA)
    return CARMA_celerite_coefs(covariance.p, covariance.rα, covariance.β, covariance.σ²)
end

""" 
    CARMA_celerite_coefs(p, rα, β, σ²)

Convert the CARMA coefficients to Celerite coefficients.

# Arguments
- `p::Int`: the order of the autoregressive polynomial
- `rα::Vector{Complex}`: roots of the autoregressive polynomial
- `β::Vector{Real}`: moving average coefficients
- `σ²::Real`: the variance of the process   
"""
function CARMA_celerite_coefs(p, rα, β, σ²)

    T = eltype(β)
    a, b, c, d = zeros(T, p), zeros(T, p), zeros(T, p), zeros(T, p)

    for (k, rₖ) in enumerate(rα)
        num_1, num_2 = 0, 0
        for (l, βₗ) in enumerate(β)
            num_1 += βₗ * rₖ^(l - 1)
            num_2 += βₗ * (-rₖ)^(l - 1)
        end
        num = num_1 * num_2
        den = -2 * real(rₖ)
        r_ = filter(x -> x != rₖ, rα)
        for rⱼ in r_
            den *= (rⱼ - rₖ) * (conj(rⱼ) + rₖ)
        end

        Frac = num / den
        a[k] = 2 * real(Frac)
        b[k] = 2 * imag(Frac)
        c[k] = -real(rₖ)
        d[k] = -imag(rₖ)
    end
    variance = sum(a)
    va = σ² / variance
    return a .* va, b .* va, c, d
end


""" 
    get_normalisation(model::CARMA)

Compute the normalisation constant of the CARMA model.

# Arguments
- `model::CARMA`: the CARMA model
"""
function get_normalisation(model::CARMA)
    variance = 0.0
    for rₖ in model.rα
        num_1, num_2 = 0, 0
        for (l, βₗ) in enumerate(model.β)
            num_1 += βₗ * rₖ^(l - 1)
            num_2 += βₗ * (-rₖ)^(l - 1)
        end
        num = num_1 * num_2

        den = -2 * real(rₖ)
        r_ = filter(x -> x != rₖ, model.rα)
        for rⱼ in r_
            den *= (rⱼ - rₖ) * (conj(rⱼ) + rₖ)
        end
        variance += num / den
    end
    return real(variance)
end

""" 
    calculate(f, model::CARMA)

Calculate the power spectral density of the CARMA model at frequency f.
"""
function calculate(f, model::CARMA)
    num = zeros(length(f))
    den = zeros(length(f))

    p = model.p
    q = model.q
    ωi = 2π * f * im
    β = model.β
    rα = model.rα
    α = roots2coeffs(rα)

    for i in 1:q+1
        num += β[i] * ωi .^ (i - 1)
    end
    for j in 1:p+1
        den += α[j] * ωi .^ (j - 1)
    end


    return abs.(num ./ den) .^ 2 / get_normalisation(model) * model.σ² / 2
end

"""
    roots2coeffs(r)

Convert the roots of a polynomial to its coefficients.

# Arguments
- `r::Vector{Complex}`: Roots of the polynomial.

# Returns
- `c::Vector{Complex}`: Coefficients of the polynomial.
"""
function roots2coeffs(r)
    P = fromroots(r)
    return P.coeffs
end

"""
    quad2roots(quad)

Convert the coefficients of a quadratic polynomial to its roots.

# Arguments
- `quad::Vector{Real}`: Coefficients of the quadratic polynomial.

# Returns
- `r::Vector{Complex}`: Roots of the polynomial.
"""
function quad2roots(quad)
    n = length(quad)
    r = zeros(Complex, n)
    if n % 2 == 1
        r[end] = -quad[end]
        n_ = n - 1
    else
        n_ = n
    end

    for k in 1:2:n_
        Δ = quad[k]^2 - 4 * quad[k+1]
        if Δ < 0
            r[k] = (-quad[k] + im * √ - Δ) / 2
            r[k+1] = conj(r[k])
        else
            r[k] = (-quad[k] + √Δ) / 2
            r[k+1] = (-quad[k] - √Δ) / 2
        end
    end
    return r
end

"""
    CARMA_covariance(τ, covariance::CARMA)

Compute the covariance function of the CARMA model at time τ.
"""
function CARMA_covariance(τ, covariance::CARMA)

    # compute the first term
    rₖ = covariance.rα[1]
    num_1, num_2 = 0, 0
    for (l, βₗ) in enumerate(covariance.β)
        num_1 += βₗ * rₖ^(l - 1)
        num_2 += βₗ * (-rₖ)^(l - 1)
    end
    num = num_1 * num_2

    den = -2 * real(rₖ)
    r_ = filter(x -> x != rₖ, covariance.rα)
    for rⱼ in r_
        den *= (rⱼ - rₖ) * (conj(rⱼ) + rₖ)
    end
    R = num .* exp.(rₖ .* abs.(τ)) / den
    variance = num / den

    # compute the remaining terms
    for rₖ in covariance.rα[2:end]
        num_1, num_2 = 0, 0
        for (l, βₗ) in enumerate(covariance.β)
            num_1 += βₗ * rₖ^(l - 1)
            num_2 += βₗ * (-rₖ)^(l - 1)
        end
        num = num_1 * num_2

        den = -2 * real(rₖ)
        r_ = filter(x -> x != rₖ, covariance.rα)
        for rⱼ in r_
            den *= (rⱼ - rₖ) * (conj(rⱼ) + rₖ)
        end
        R += num .* exp.(rₖ .* abs.(τ)) / den
        variance += num / den
    end
    return real.(R) ./ real(variance) * covariance.σ²
end