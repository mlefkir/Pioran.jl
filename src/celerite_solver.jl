using Pioran: SumOfSemiSeparable
using LinearAlgebra
using LoopVectorization

"""
     init_semi_separable(a, b, c, d, τ, σ2)

Initialise the matrices and vectors needed for the celerite algorithm.
U,V are the rank-R matrices, D is the diagonal matrix and ϕ is the matrix of the exponential terms.

See [Foreman-Mackey et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F) for more details.
"""

@inline function init_semi_separable!(
        a::AbstractVector, b::AbstractVector, c::AbstractVector,
        d::AbstractVector, τ::AbstractVector, σ2::AbstractVector, V::AbstractMatrix,
        D::AbstractVector, U::AbstractMatrix, ϕ::Matrix, S_n::AbstractMatrix
    )

    J::Int64 = length(a)
    R::Int64 = 2 * J
    # sum of a coefficients
    suma = sum(a)
    # number of data points
    N::Int64 = length(τ)

    # initialise matrices and vectors
    # it is faster to access the columns
    D[1] = suma + σ2[1]
    dn = D[1]
    buff = 1.0 / dn
    τ1 = τ[1]

    # initialise first row
    @inbounds for j in 1:J
        co = cos(d[j] * τ1)
        si = sin(d[j] * τ1)

        V[2j, 1] = si * buff
        V[2j - 1, 1] = co * buff

        U[2j, 1] = a[j] * si - b[j] * co
        U[2j - 1, 1] = a[j] * co + b[j] * si
    end

    return @inbounds for n in 2:N

        s = 0.0
        τn = τ[n]
        dτ = τn - τ[n - 1]

        # initialise the U,V and ϕ matrices
        @inbounds for j in 1:J
            co = cos(d[j] * τn)
            si = sin(d[j] * τn)
            ec = exp(-c[j] * dτ)

            ϕ[2j, n - 1] = ec
            ϕ[2j - 1, n - 1] = ec

            U[2j, n] = a[j] * si - b[j] * co
            U[2j - 1, n] = a[j] * co + b[j] * si

            V[2j, n] = si
            V[2j - 1, n] = co
        end

        # use the property that S_n is symmetric to fill only the lower triangle
        # compute the triple product U*S*U and the sum for D at the same time
        # no Float64 order is needed for the computation
        @inbounds for j in 1:R
            uj = U[j, n]
            ϕnj = ϕ[j, n - 1]
            vn = V[j, n - 1]
            dn = D[n - 1] * vn
            vnj = V[j, n]

            @inbounds for k in 1:(j - 1)
                uk = U[k, n]
                r = ϕnj * ϕ[k, n - 1] * (S_n[j, k] + dn * V[k, n - 1])
                S_n[j, k] = r
                v = uj * r
                V[k, n] -= v
                vnj -= uk * r
                s += 2 * v * uk # 2 times because of symmetry
            end
            S_n[j, j] = ϕnj^2 * (S_n[j, j] + dn * vn)
            r = S_n[j, j] * uj

            s += r * uj
            V[j, n] = vnj - r
        end
        # compute the diagonal element
        dn = suma + σ2[n] - s
        D[n] = dn
        # update V ( which is W in the paper )
        @inbounds for j in 1:R
            V[j, n] /= dn

        end
    end
end

init_semi_separable!(
    a::Float64, b::Float64, c::Float64,
    d::Float64, τ::AbstractVector, σ2::AbstractVector, V::AbstractMatrix,
    D::AbstractVector, U::AbstractMatrix, ϕ::Matrix, S_n::AbstractMatrix
) = init_semi_separable!([a], [b], [c], [d], τ, σ2, V, D, U, ϕ, S_n)

"""
    solve_prec!(z, y, U, W, D, ϕ)

Forward and backward substitution of the celerite algorithm.

See [Foreman-Mackey et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F) for more details.
"""
@inline function solve_prec!(
        z::AbstractVector, y::AbstractVector,
        U::AbstractMatrix, W::AbstractMatrix, D::AbstractVector, ϕ::AbstractMatrix
    )
    T = eltype(U)
    N = length(y)
    R = size(U, 1)
    f = Vector{T}(undef, R)
    fp = zeros(T, R)
    g = Vector{T}(undef, R)
    gp = zeros(T, R)
    logdetD = log.(D[1])
    # because ϕ[j,1] = 0
    z[1] = y[1]


    # forward substitution
    @inbounds for n in 2:N
        s = 0.0
        z_p = z[n - 1]
        @inbounds for j in 1:R
            f[j] = (gp[j] + W[j, n - 1] * z_p) * ϕ[j, n - 1]
            s += U[j, n] * f[j]
        end
        gp = f
        logdetD += log(abs(D[n]))
        z[n] = y[n] - s
    end

    # backward substitution
    z[N] /= D[N]
    @inbounds for n in (N - 1):-1:1
        s = 0.0
        zn = z[n + 1]
        @inbounds for j in 1:R
            g[j] = (fp[j] + U[j, n + 1] * zn) * ϕ[j, n]
            s += W[j, n] * g[j]
        end
        fp = g
        z[n] = z[n] / D[n] - s
    end

    return logdetD
end

"""
    get_values!(a, b, c, d, zp, U, V, P, D, t)

Compute the values of the matrices and vectors needed for the celerite algorithm.
This is a vectorised version of the `init_semi_separable!` and `solve_prec!` functions.
This function appears to be faster than the two previous functions when J > 16 but it also uses more memory.

More study of this implementation is needed.

"""
@inline function get_values!(
        a::AbstractVector, b::AbstractVector, c::AbstractVector, d::AbstractVector
        , zp::AbstractVector, U::AbstractMatrix, V::AbstractMatrix, P::AbstractMatrix, D::AbstractVector, t::AbstractVector
    )
    R, N = size(U)
    T = eltype(U)
    S_n = zeros(T, R, R)
    f = zeros(T, R)
    return @views  begin
        odd = 1:2:R
        even = 2:2:R
        dτ = diff(t)
        td = @turbo @. t' * d
        co = @turbo @. cos(td)
        si = @turbo @. sin(td)
        ec = @turbo @. exp(-dτ' * c)
        U[odd, :] = @turbo @. a * co + b * si
        U[even, :] = @turbo @. a * si - b * co
        V[odd, :] = co
        V[even, :] = si
        P[even, :] = ec
        P[odd, :] = ec
        V[:, 1] /= D[1]
        @inbounds for n in 2:N
            un = U[:, n]
            vn1 = V[:, n - 1]
            pn1 = P[:, n - 1]
            @turbo @. S_n += D[n - 1] * vn1 * vn1'
            @turbo @. S_n *= pn1 * pn1'
            buff = S_n * un
            D[n] -= (un)' * buff
            V[:, n] -= buff
            V[:, n] /= D[n]
            @turbo @. f += (vn1) * zp[n - 1]
            @turbo @. f *= pn1
            zp[n] -= (un)' * f
        end
        f = zeros(T, R)
        zp[N] /= D[N]
        @inbounds for n in (N - 1):-1:1
            @turbo  @. f += (U[:, n + 1]) * zp[n + 1]
            @turbo @. f *= (P[:, n])
            zp[n] /= D[n]
            zp[n] -= (V[:, n])' * f
        end
    end
end

"""
    compute_nll(t, y, σ², a, b, c, d)

    Compute the likelihood using the vectorised implementation in `get_values!`.

    Still experimental.
"""
@inline function compute_nll(t, y, σ², a, b, c, d)
    N = length(y)
    J = length(a)
    T = eltype(a)
    R = 2 * J
    U = Matrix{T}(undef, R, N)
    V = Matrix{T}(undef, R, N)
    P = Matrix{T}(undef, R, N - 1)
    suma = sum(a)

    D = @turbo @. σ² + suma
    zp = y[:]

    get_values!(
        a, b, c, d, zp,
        U,
        V,
        P,
        D,
        t
    )
    𝓛 = @turbo -sum(log.(D)) / 2 - N * log(2π) / 2 - y'zp / 2
    return 𝓛
end

"""
    log_likelihood(cov, τ, y, σ2)

Compute the log-likelihood of a semi-separable covariance function using the celerite algorithm.

# Arguments
- `cov::SumOfSemiSeparable` or `cov::CARMA` or `cov::SemiSeparable`: the covariance function
- `τ::Vector`: the time points
- `y::Vector`: the data
- `σ2::Vector`: the measurement variances

"""
@inline function log_likelihood(cov::SumOfCelerite, τ, y, σ2)
    return logl(cov.a, cov.b, cov.c, cov.d, τ, y, σ2)
    #return compute_nll(τ, y, σ2, cov.cov.a, cov.cov.b, cov.cov.c, cov.cov.d)
end

@inline function log_likelihood(cov::CARMA, τ, y, σ2)
    a, b, c, d = celerite_coefs(cov)
    return real(logl(a, b, c, d, τ, y, σ2))
end

@inline function log_likelihood(cov::SemiSeparable, τ, y, σ2)
    a, b, c, d = celerite_coefs(cov)
    return logl(a, b, c, d, τ, y, σ2)
end

"""
    logl(a, b, c, d, τ, y, σ2)

Compute the log-likelihood of a GP with a semi-separable covariance function using the celerite algorithm.

# Arguments
- `a::Vector`
- `b::Vector`
- `c::Vector`
- `d::Vector`
- `τ::Vector`: the time points
- `y::Vector`: the data
- `σ2::Vector`: the measurement variances

See [Foreman-Mackey et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F) for more details.
"""
function logl(a, b, c, d, τ, y, σ2)
    N::Int64 = length(y)

    # initialise the matrices and vectors
    T = eltype(a)
    # number of terms
    J::Int64 = length(a)
    # number of rows in U and V, twice the number of terms
    R::Int64 = 2 * J

    S_n = zeros(T, R, R)
    ϕ = Matrix{T}(undef, R, N - 1)
    U = Matrix{T}(undef, R, N)
    V = Matrix{T}(undef, R, N)
    D = Vector{T}(undef, N)

    init_semi_separable!(a, b, c, d, τ, σ2, V, D, U, ϕ, S_n)

    z = Vector{T}(undef, N)

    logdetD = solve_prec!(z, y, U, V, D, ϕ)
    return -logdetD / 2 - N * log(2π) / 2 - y'z / 2
end

"""
    predict(cov, τ, t, y, σ2)

Compute the posterior mean of the GP at the points τ given the data y and time t.

# Arguments
- `cov::SumOfSemiSeparable` or `cov::CARMA` or `cov::SemiSeparable`: the covariance function
- `τ::Vector`: the time points
- `t::Vector`: the data time points
- `y::Vector`: the data
- `σ2::Vector`: the measurement variances
"""
function predict(cov::SumOfSemiSeparable, τ::AbstractVector, t::AbstractVector, y::AbstractVector, σ²::AbstractVector)
    a, b, c, d = cov.a, cov.b, cov.c, cov.d
    return pred(a, b, c, d, τ, t, y, σ²)
end

function predict(cov::CARMA, τ::AbstractVector, t::AbstractVector, y::AbstractVector, σ²::AbstractVector)
    a, b, c, d = celerite_coefs(cov)
    return pred(a, b, c, d, τ, t, y, σ²)
end

function predict(cov::SemiSeparable, τ::AbstractVector, t::AbstractVector, y::AbstractVector, σ²::AbstractVector)
    a, b, c, d = celerite_coefs(cov)
    return pred(a, b, c, d, τ, t, y, σ²)
end

function pred(a, b, c, d, τ::AbstractVector, t::AbstractVector, y::AbstractVector, σ²::AbstractVector)

    M = length(τ)
    N = length(t)
    # initialise the matrices and vectors
    # get the coefficients
    T = eltype(a)
    # number of terms
    J::Int64 = length(a)
    # number of rows in U and V, twice the number of terms
    R::Int64 = 2 * J

    S_n = zeros(T, R, R)
    ϕ = Matrix{T}(undef, R, N - 1)
    U = Matrix{T}(undef, R, N)
    V = Matrix{T}(undef, R, N)
    D = Vector{T}(undef, N)
    init_semi_separable!(a, b, c, d, t, σ², V, D, U, ϕ, S_n)
    # get z
    z = Vector{T}(undef, N)

    _ = solve_prec!(z, y, U, V, D, ϕ)

    Q = zeros(T, R) # same as in the paper
    μₘ = zeros(T, M)
    n₀L = searchsortedfirst.(Ref(t), τ) .- 1
    S = zeros(T, R) # ia  Q' * X⁻

    start = 1
    ### forward pass ###
    for (m, n₀) in enumerate(n₀L)
        τm = τ[m]
        # compute Q, S
        #-----------------#
        for n in start:(n₀ - 1)
            start += 1
            tn = t[n]
            tn₊₁ = t[n + 1]
            zn = z[n]

            Q[1:2:end] = (Q[1:2:end] .+ zn .* cos.(d .* tn)) .* exp.(-c .* (tn₊₁ .- tn))
            Q[2:2:end] = (Q[2:2:end] .+ zn .* sin.(d .* tn)) .* exp.(-c .* (tn₊₁ .- tn))
        end
        # compute S = X⁻' * Q and then update Q
        if start >= n₀ && n₀ != 0
            tn = t[n₀]
            zn = z[n₀]
            if n₀ == N
                tn₊₁ = t[n₀]
            else
                tn₊₁ = t[n₀ + 1]
            end

            S[2:2:end] = (Q[2:2:end] .+ zn .* sin.(d .* tn)) .* exp.(-c .* (τm .- tn)) .* (a .* sin.(d .* τm) .- b .* cos.(d .* τm))
            S[1:2:end] = (Q[1:2:end] .+ zn .* cos.(d .* tn)) .* exp.(-c .* (τm - tn)) .* (a .* cos.(d .* τm) .+ b .* sin.(d .* τm))
            # update Q for the next iteration
            if start + 1 == n₀
                start += 1

                Q[1:2:end] = (Q[1:2:end] .+ zn .* cos.(d .* tn)) .* exp.(-c .* (tn₊₁ .- tn))
                Q[2:2:end] = (Q[2:2:end] .+ zn .* sin.(d .* tn)) .* exp.(-c .* (tn₊₁ .- tn))
            end
        end
        #-----------------#
        μₘ[m] = sum(S)
    end
    # reset Q
    fill!(Q, 0.0)

    ### backward pass ###
    stop = N
    for (m, n₀) in Iterators.reverse(enumerate(n₀L))
        if n₀ != N # if n₀ == N, then we already have the value for μₘ[m]
            τm = τ[m]
            stop_cur = stop
            for n in stop_cur:-1:(n₀ + 2)
                stop -= 1
                tn = t[n]
                tn₋₁ = t[n - 1]
                zn = z[n]

                Q[1:2:end] = (Q[1:2:end] .+ zn .* (a .* cos.(d .* tn) .+ b .* sin.(d .* tn))) .* exp.(-c .* (tn - tn₋₁))
                Q[2:2:end] = (Q[2:2:end] .+ zn .* (a .* sin.(d .* tn) .- b .* cos.(d .* tn))) .* exp.(-c .* (tn - tn₋₁))

            end

            # compute S = X⁺' * Q and then update Q
            n = n₀ + 1
            zn = z[n]
            tn = t[n]
            if n == 1
                tn₋₁ = t[1]
            else
                tn₋₁ = t[n - 1]
            end

            S[1:2:end] = (Q[1:2:end] .+ zn .* (a .* cos.(d .* tn) .+ b .* sin.(d .* tn))) .* exp.(-c .* (tn .- τm)) .* cos.(d .* τm)
            S[2:2:end] = (Q[2:2:end] .+ zn .* (a .* sin.(d .* tn) .- b .* cos.(d .* tn))) .* exp.(-c .* (tn .- τm)) .* sin.(d .* τm)
            # update Q for the next iteration
            if m != 1
                k = m
            else
                k = 2
            end

            if (stop_cur == n₀ + 1) && (n₀L[k] != n₀L[k - 1]) # if n₀L[k] == n₀L[k-1], then we do not need to update Q yet
                stop -= 1

                Q[1:2:end] = (Q[1:2:end] .+ zn .* (a .* cos.(d .* tn) .+ b .* sin.(d .* tn))) .* exp.(-c .* (tn - tn₋₁))
                Q[2:2:end] = (Q[2:2:end] .+ zn .* (a .* sin.(d .* tn) .- b .* cos.(d .* tn))) .* exp.(-c .* (tn - tn₋₁))

            end

            μₘ[m] += sum(S)
        end

    end

    return μₘ

end

"""
    simulate(rng, cov, τ, σ2)
    simulate(cov, τ, σ2)

Draw a realisation from the  GP with the covariance function cov at the points τ with the variances σ2.

# Arguments
- `rng::AbstractRNG`: the random number generator
- `cov::SumOfSemiSeparable` or `cov::CARMA` or `cov::SemiSeparable`: the covariance function
- `τ::Vector`: the time points
- `σ2::Vector`: the measurement variances
"""
function simulate(rng::AbstractRNG, cov::SumOfSemiSeparable, τ::AbstractVector, σ2::AbstractVector)
    a, b, c, d = cov.a, cov.b, cov.c, cov.d
    return sim(rng, a, b, c, d, τ, σ2)
end

function simulate(rng::AbstractRNG, cov::SemiSeparable, τ::AbstractVector, σ2::AbstractVector)
    a, b, c, d = celerite_coefs(cov)
    return sim(rng, a, b, c, d, τ, σ2)
end

function simulate(rng::AbstractRNG, cov::CARMA, τ::AbstractVector, σ2::AbstractVector)
    a, b, c, d = celerite_coefs(cov)
    return sim(rng, a, b, c, d, τ, σ2)
end

simulate(cov::SumOfSemiSeparable, τ::AbstractVector, σ2::AbstractVector) = simulate(Random.GLOBAL_RNG, cov::SumOfSemiSeparable, τ::AbstractVector, σ2::AbstractVector)


function sim(rng::AbstractRNG, a, b, c, d, τ::AbstractVector, σ2::AbstractVector)
    N::Int64 = length(τ)

    q = randn(rng, N)
    # initialise the matrices and vectors
    T = eltype(a)
    # number of terms
    J::Int64 = length(a)
    # number of rows in U and V, twice the number of terms
    R::Int64 = 2 * J

    S_n = zeros(T, R, R)
    ϕ = zeros(T, R, N - 1)
    U = zeros(T, R, N)
    V = zeros(T, R, N)
    D::Vector = zeros(T, N)::Union{Vector, Matrix{Float64}}

    init_semi_separable!(a::Vector, b::Vector, c::Vector, d::Vector, τ, σ2, V, D::Vector, U, ϕ, S_n)

    y_sim = zeros(N)
    y_sim[1] = sqrt(D[1]) * q[1]
    f = zeros(R)
    g = zeros(R)

    for n in 2:N
        for j in 1:R
            f[j] = ϕ[j, n - 1] * (g[j] + V[j, n - 1] * sqrt(D[n - 1]) * q[n - 1])
            y_sim[n] += U[j, n] * f[j]
        end
        g = f
        y_sim[n] += sqrt(D[n]) * q[n]
    end

    return y_sim
end
