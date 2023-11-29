using Pioran: SumOfSemiSeparable

"""
init_semi_separable(a, b, c, d, τ, σ2)"
Initialise the matrices and vectors needed for the celerite algorithm.
U,V are the rank-R matrices, D is the diagonal matrix and ϕ is the matrix of the exponential terms.

See Foreman-Mackey et al. 2017 for more details.
"""
function init_semi_separable!(J::Int64, a::Vector, b::Vector, c::Vector, d::Vector, τ::Vector, σ2::Vector, V::Matrix, D::Vector, U::Matrix, ϕ::Matrix, S_n::Matrix)

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
    for j in 1:J
        co = cos(d[j] * τ1)
        si = sin(d[j] * τ1)

        V[2j, 1, 1] = si * buff
        V[2j-1, 1] = co * buff

        U[2j, 1, 1] = a[j] * si - b[j] * co
        U[2j-1, 1] = a[j] * co + b[j] * si
    end

    @inbounds for n in 2:N

        s = 0.0
        τn = τ[n]
        dτ = τn - τ[n-1]

        # initialise the U,V and ϕ matrices
        @inbounds for j in 1:J
            co = cos(d[j] * τn)
            si = sin(d[j] * τn)
            ec = exp(-c[j] * dτ)

            ϕ[2j, n-1] = ec
            ϕ[2j-1, n-1] = ec

            U[2j, n] = a[j] * si - b[j] * co
            U[2j-1, n] = a[j] * co + b[j] * si

            V[2j, n] = si
            V[2j-1, n] = co
        end

        # use the property that S_n is symmetric to fill only the lower triangle
        # compute the triple product U*S*U and the sum for D at the same time
        # no Float64 order is needed for the computation
        @inbounds for j in 1:R
            uj = U[j, n]
            ϕnj = ϕ[j, n-1]
            vn = V[j, n-1]
            dn = D[n-1] * vn
            vnj = V[j, n]

            @inbounds for k in 1:j-1
                uk = U[k, n]
                r = ϕnj * ϕ[k, n-1] * (S_n[j, k] + dn * V[k, n-1])
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
        for j in 1:R
            V[j, n] /= dn

        end
    end
end

""" Solve the linear system using the celerite algorithm

See Foreman-Mackey et al. 2017 for more details.
"""
function solve_prec(y::Vector, U::Matrix, W::Matrix, D::Vector, ϕ::Matrix)
    T = eltype(U)
    N = length(y)
    R = size(U, 1)
    zp = zeros(T, N)
    z = zeros(T, N)
    f = zeros(T, R, N)
    g = zeros(T, R, N)
    logdetD = log.(D[1])
    # because ϕ[j,1] = 0
    zp[1] = y[1]


    # forward substitution
    @inbounds for n in 2:N
        s = 0.0
        z_p = zp[n-1]
        for j in 1:R
            f[j, n] = (f[j, n-1] + W[j, n-1] * z_p) * ϕ[j, n-1]
            s += U[j, n] * f[j, n]
        end
        logdetD += log(abs(D[n]))
        zp[n] = y[n] - s
    end

    # backward substitution
    z[N] = zp[N] / D[N]
    @inbounds for n = N-1:-1:1
        s = 0.0
        zn = z[n+1]
        for j in 1:R
            g[j, n] = (g[j, n+1] + U[j, n+1] * zn) * ϕ[j, n]
            s += W[j, n] * g[j, n]
        end
        z[n] = zp[n] / D[n] - s
    end

    return z, logdetD
end

"""

Compute the log-likelihood of the data y given the covariance function cov
"""
function log_likelihood(cov::SumOfSemiSeparable, τ::Vector, y::Vector, σ2::Vector)

    N::Int64 = length(y)

    # initialise the matrices and vectors
    a::Vector, b::Vector, c::Vector, d::Vector = cov.a::Vector, cov.b::Vector, cov.c::Vector, cov.d::Vector
    T = eltype(a)
    # number of terms
    J::Int64 = length(a)
    # number of rows in U and V, twice the number of terms
    R::Int64 = 2 * J

    S_n::Union{Array{Float64,3},Matrix} = zeros(T, R, R)::Union{Array{Float64,3},Matrix}
    ϕ::Union{Array{Float64,3},Matrix} = zeros(T, R, N - 1)::Union{Array{Float64,3},Matrix}
    U::Union{Array{Float64,3},Matrix} = zeros(T, R, N)::Union{Array{Float64,3},Matrix}
    V::Union{Array{Float64,3},Matrix} = zeros(T, R, N)::Union{Array{Float64,3},Matrix}
    D::Vector = zeros(T, N)::Union{Vector,Matrix{Float64}}

    init_semi_separable!(J::Int64, a::Vector, b::Vector, c::Vector, d::Vector, τ, σ2, V::Union{Array{Float64,3},Matrix}, D::Vector, U::Union{Array{Float64,3},Matrix}, ϕ::Union{Array{Float64,3},Matrix}, S_n::Union{Array{Float64,3},Matrix})

    z::Union{Vector,Matrix{Float64}}, logdetD::Any = solve_prec(y, U, V, D, ϕ)
    # println("logdetD = ", logdetD)
    # println("z = ", z)
    # println("chi2 = ", y'z *0.5 )
    return -logdetD / 2 - N * log(2π) / 2 - y'z / 2
end
