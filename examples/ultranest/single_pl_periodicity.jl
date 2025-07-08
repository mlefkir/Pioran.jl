#= Example script for the inference of a periodic signal embedded in red noise modelled by a single bending power spectral density.

using Pioran and ultranest (python package).

run with:

julia script_name.jl data.txt

where `data.txt` is a file containing the time series data. The file should have three columns: time, flux, flux error.
The script will create a directory `inference` containing the results of the inference.

If you have MPI installed, you may want to run the script in parallel, using the following command:

mpirun -n 4 julia script.jl data.txt

where `-n 4` is the number of processes to use.

=#

# load MPI and initialise

using MPI
MPI.Init()
comm = MPI.COMM_WORLD

using Pioran
using Distributions
using Statistics
using Random
using DelimitedFiles
using PyCall

# load the python package ultranest
ultranest = pyimport("ultranest")
warnings = pyimport("warnings")
warnings.filterwarnings("ignore")

# define the random number generator
seed = 123
rng = MersenneTwister(123)

# get the filename from the command line
filename = ARGS[1]
fname = replace(split(filename, "/")[end], ".txt" => "_periodic_rednoise")
dir = "inference/" * fname * "_$(seed)_factor"
plot_path = dir * "/plots/"


# work on one process
if MPI.Comm_rank(comm) == 0
    if !ispath(dir)
        mkpath(dir)
    end

    # Load the data and extract a subset for the analysis
    A = readdlm(filename, comments = true, comment_char = '#')
    t_all, y_all, yerr_all = A[:, 1], A[:, 2], A[:, 3]

    t_all = t_all .- t_all[1]
    t, y, yerr, x̄, va = extract_subset(seed, dir * "/" * fname, t_all, y_all, yerr_all, take_log = false)
end

# wait
MPI.Barrier(comm)
# do it again but on the other ones

if MPI.Comm_rank(comm) != 0
    # Load the data and extract a subset for the analysis
    A = readdlm(filename, comments = true, comment_char = '#')
    t_all, y_all, yerr_all = A[:, 1], A[:, 2], A[:, 3]

    t_all = t_all .- t_all[1]
    t, y, yerr, x̄, va = extract_subset(seed, dir * "/" * fname, t_all, y_all, yerr_all, take_log = false)
end


T = (t[end] - t[1]) # duration of the time series
Δt = minimum(diff(t)) # min time separation


# Frequency range for the approx and the prior
f_min, f_max = 1 / T, 1 / Δt / 2
f0, fM = f_min / 20.0, f_max * 20.0
min_f_b, max_f_b = f0 * 4.0, fM / 4.0

# F var^2 is distributed as a log-normal
μᵥ, σᵥ = -1.5, 1.0;
μₙ, σₙ² = 2μᵥ, 2(σᵥ)^2;
σₙ = sqrt(σₙ²)

# options for the approximation
basis_function = "SHO" # α₂ cannot be steeper than 4! with SHO
n_components = 20
model = SingleBendingPowerLaw
posterior_checks = true
prior_checks = false

""" function to build the Gaussian process
"""
function GP_model(t, y, σ, params, basis_function = basis_function, n_components = n_components, model = model)

    T = (t[end] - t[1]) # duration of the time series
    Δt = minimum(diff(t)) # min time separation

    f_min, f_max = 1 / T, 1 / Δt / 2

    α₁, f₁, α₂, variance, ν, μ, A, ϕ, T₀ = params

    # Rescale the measurement variance
    σ² = ν .* σ .^ 2

    # Define the mean
    mean_function(x, A = A, ϕ = ϕ, T₀ = T₀, μ = μ) = @. A * sin(2π * x / T₀ + ϕ) + μ

    μ_fun = CustomMean(mean_function)
    # Define power spectral density function
    𝓟 = model(α₁, f₁, α₂)

    # Approximation of the PSD to form a covariance function
    𝓡 = approx(𝓟, f_min, f_max, n_components, variance, basis_function = basis_function)

    # Build the GP
    f = ScalableGP(μ_fun, 𝓡)

    # Condition on the times and errors
    fx = f(t, σ²)
    return fx
end

function loglikelihood(t, y, σ, params)
    fx = GP_model(t, y, σ, params)
    return logpdf(fx, y)
end
logl(pars) = loglikelihood(t, y, yerr, pars)

# Priors should in the same order as in the likelihood
function prior_transform(cube)
    α₁ = quantile(Uniform(0.0, 1.5), cube[1])
    f₁ = quantile(LogUniform(min_f_b, max_f_b), cube[2])
    α₂ = quantile(Uniform(α₁, 4.0), cube[3])
    variance = quantile(LogNormal(μₙ, σₙ), cube[4])
    ν = quantile(Gamma(2, 0.5), cube[5])
    μ = quantile(Normal(x̄, 5 * sqrt(va)), cube[6])
    A = quantile(LogNormal(0.0, 1.0), cube[7])
    ϕ = quantile(Uniform(0.0, 2π), cube[8])
    T₀ = quantile(Uniform(0, T), cube[9])
    return [α₁, f₁, α₂, variance, ν, μ, A, ϕ, T₀]
end
paramnames = ["α₁", "f₁", "α₂", "variance", "ν", "μ", "A", "ϕ", "T₀"]

if MPI.Comm_rank(comm) == 0 && prior_checks
    unif = rand(rng, 9, 3000) # uniform samples from the unit hypercube
    priors = mapreduce(permutedims, hcat, [prior_transform(unif[:, i]) for i in 1:3000]') # transform the uniform samples to the prior
    run_diagnostics(priors[1:3, :], priors[4, :], f_min, f_max, model, path = plot_path, basis_function = basis_function, n_components = n_components)
end

println("Hello world, I am $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))")

println("Running sampler...")
sampler = ultranest.ReactiveNestedSampler(paramnames, logl, resume = true, transform = prior_transform, log_dir = dir, vectorized = false)
results = sampler.run()
sampler.print_results()
sampler.plot()


if MPI.Comm_rank(comm) == 0 && posterior_checks
    samples = readdlm(dir * "/chains/equal_weighted_post.txt", skipstart = 1)

    # this dict helps splitting the samples in the various components
    paramnames_split = Dict(
        "psd" => ["α₁", "f₁", "α₂"],
        "norm" => "variance",
        "scale_err" => "ν",
        "mean" => ["A", "ϕ", "T₀", "μ"]
    )

    run_posterior_predict_checks(samples, paramnames, paramnames_split, t, y, yerr, model, GP_model, false; plots = ["psd", "lsp", "timeseries"], plot_f_P = false, path = plot_path, basis_function = basis_function, n_components = n_components)
end
