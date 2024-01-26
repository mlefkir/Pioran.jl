
function log_likelihood_direct(cov::KernelFunctions.SimpleKernel, t::Vector, y::Vector, σ²::Vector)

    N::Int64 = length(y)
    K = zeros(N, N)
    @inbounds for i in 1:N
        @inbounds for j in 1:N
            K[i, j] = cov.(t[i], t[j])
        end
    end
    K = K + Diagonal(σ²)
    L = cholesky(K)
    z = L.U' \ y

    return logdet(L.U) + 0.5 * z' * z + 0.5 * N * log(2 * pi)

end

function predict_direct(cov::KernelFunctions.SimpleKernel, τ::AbstractVector, t::AbstractVector, y::AbstractVector, σ²::AbstractVector, with_covariance::Bool=false)

    N = length(t)
    M = length(τ)

    K0 = zeros(N, N)
    Kτ = zeros(M, M)
    Kτ0 = zeros(M, N)

    # fill the matrices
    @inbounds for i in 1:N
        @inbounds for j in 1:N
            K0[i, j] = cov.(t[i], t[j])
        end
    end

    @inbounds for i in 1:M
        @inbounds for j in 1:M
            Kτ[i, j] = cov.(τ[i], τ[j])
        end
    end

    @inbounds for i in 1:M
        @inbounds for j in 1:N
            Kτ0[i, j] = cov.(τ[i], t[j])
        end
    end

    # add the noise
    K0 += Diagonal(σ²)

    # Cholesky decomposition of the covariance matrix
    L = cholesky(K0).U'
    # solve the linear system
    w = L \ Kτ0'

    # compute the posterior mean and covariance
    y_p = w' * (L \ I) * y
    if with_covariance
        K_p = Kτ - w'w
        return y_p, K_p
    end
    return y_p

end