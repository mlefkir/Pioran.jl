#= Example script for the inference of a single bending power spectral density using Pioran and ultranest (python package).

run with:

julia single.jl data.txt

where `data.txt` is a file containing the time series data. The file should have three columns: time, flux, flux error.
The script will create a directory `inference` containing the results of the inference.

If you have MPI installed, you may want to run the script in parallel, using the following command:

mpirun -n 4 julia single.jl data.txt

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
fname = replace(split(filename, "/")[end], ".txt" => "_single")

dir = "../inference/" * fname * "_$(seed)_factor"

plot_path = dir * "/plots/"

# work on one process
if MPI.Comm_rank(comm) == 0
    if !ispath(dir)
        mkpath(dir)
    end

    # Load the data and extract a subset for the analysis
    A = readdlm(filename, comments = true, comment_char = '#')
    t, y, yerr = A[:, 1], A[:, 2], A[:, 3]
    t = t .- t[1]
    x̄ = mean(y)
    va = var(y)
end
# wait
MPI.Barrier(comm)
# do it again but on the other ones
if MPI.Comm_rank(comm) != 0
    # Load the data and extract a subset for the analysis
    A = readdlm(filename, comments = true, comment_char = '#')
    t, y, yerr = A[:, 1], A[:, 2], A[:, 3]
    t = t .- t[1]
    x̄ = mean(y)
    va = var(y)
end

# Frequency range for the approx and the prior
f_min, f_max = 1 / (t[end] - t[1]), 1 / minimum(diff(t)) / 2
S_low = S_high = 20.0
min_f_b, max_f_b = f_min / 5, f_max * 5

# F var^2 is distributed as a log-normal
μᵥ, σᵥ = -1.5, 1.0 / sqrt(2);
μₙ, σₙ² = 2μᵥ, 4(σᵥ)^2;
σₙ = sqrt(σₙ²)

# options for the approximation
basis_function = "DRWCelerite"
n_components = 30
model = SingleBendingPowerLaw

posterior_checks = true
prior_checks = false

@inline function loglikelihood(y, t, σ2, params, S_low = S_low, S_high = S_high, model = model, basis_function = basis_function, n_components = n_components)
    α₁, f₁, α₂, variance, ν, μ = params

    # Rescale the measurement variance
    σ² = ν * σ2

    # Define power spectral density function
    𝓟 = model(α₁, f₁, α₂)

    # Approximation of the PSD to form a covariance function
    𝓡 = approx(𝓟, f_min, f_max, n_components, variance, S_low, S_high, basis_function = basis_function, is_integrated_power = true)

    # Build the GP
    f = ScalableGP(μ, 𝓡)

    # sample the conditioned distribution
    return logpdf(f(t, σ²), y)
end
@inline logl(pars) = loglikelihood(log.(y), t, (yerr ./ y) .^ 2, pars)

# Priors
@inline function prior_transform(cube, μₙ = μₙ, σₙ = σₙ, x̄ = x̄, va = va, min_f_b = min_f_b, max_f_b = max_f_b)
    α₁ = quantile(Uniform(0.0, 1.5), cube[1])
    f₁ = quantile(LogUniform(min_f_b, max_f_b), cube[2])
    α₂ = quantile(Uniform(α₁, 6.0), cube[3])
    variance = quantile(LogNormal(μₙ, σₙ), cube[4])
    ν = quantile(Gamma(2, 0.5), cube[5])
    μ = quantile(Normal(x̄, 5 * sqrt(va)), cube[6])
    return [α₁, f₁, α₂, variance, ν, μ]
end
paramnames = ["α₁", "f₁", "α₂", "variance", "ν", "μ"]

println("Hello world, I am $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))")

println("Running sampler...")
sampler = ultranest.ReactiveNestedSampler(paramnames, logl, resume = true, transform = prior_transform, log_dir = dir, vectorized = false)
results = sampler.run(min_num_live_points = 400)
sampler.print_results()
sampler.plot()
