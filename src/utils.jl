"""
extract_subset(rng, t, y, yerr)

Extract a subset of the data and log transform it.

Parameters
----------
rng : Random.MersenneTwister
    Random number generator.
t : Array{Float64,1}
    Time array.
y : Array{Float64,1}
    Time series array.
yerr : Array{Float64,1}
    Time series error array.

Returns
-------
t_subset
    Time array of the subset.
y_subset
    Time series array of the subset.
yerr_subset 
    Time series error array of the subset.
μμ
    Mean of the log-normal distribution for μ.
σ²μ
    Standard deviation  of the log-normal distribution for μ.
μσ²
    Mean of the log-normal distribution for σ².
σσ²
    Standard deviation of the log-normal distribution for σ².
"""
function extract_subset(rng, t, y, yerr)

    # total number of points
    n_points = length(t)
    # number of points to remove
    n_samples = Int(round(n_points * 0.02))
    # println("n_samples = ", n_samples, " n_points = ", n_points)
    # indexes of points to remove
    subset = sample(rng, range(1, n_points), n_samples, replace=false)
    # indexes of points to keep
    x = range(1, n_points)
    subset_indexes = findall(x -> x ∉ subset, x)

    t_subset = t[subset_indexes]
    y_subset = y[subset_indexes]
    yerr_subset = yerr[subset_indexes]

    # log transform
    x = log.(y_subset)

    # initial guess
    va = var(x)
    x̄ = mean(x)
    σ²ₓ = x̄
    # println("x̄ = ", x̄, " σ²ₓ = ", σ²ₓ)

    # μ
    σ²μ = (log(1 + (σ²ₓ / x̄^2)))
    μμ = log(x̄) - σ²μ / 2
    # println("μμ = ", μμ, " σ²μ = ", σ²μ)

    # F_var   
    μᵥ = -1.5
    σᵥ = 1.0

    # σ²
    μσ² = 2 * μᵥ + 2 * μμ
    σσ² = sqrt(2 * σᵥ + 2 * σ²μ)
    # println("μσ² = ", μσ², " σσ² = ", σσ²)
    return t_subset, y_subset, yerr_subset, x̄, va, μμ, sqrt(σ²μ), μσ², σσ²
end