
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
