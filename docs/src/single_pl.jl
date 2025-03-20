""" Example script for the inference of a single bending power spectral density using Pioran and ultranest (python package).

run with:
```bash
julia simple.jl data.txt
```
where `data.txt` is a file containing the time series data. The file should have three columns: time, flux, flux error.
The script will create a directory `inference` containing the results of the inference.

If you have MPI installed, you may want to run the script in parallel, using the following command:
```bash
mpirun -n 4 julia simple.jl data.txt
```
where `-n 4` is the number of processes to use.

"""
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
# define the random number generator
rng = MersenneTwister(123)

# get the filename from the command line
filename = ARGS[1]
fname = replace(split(filename, "/")[end], ".txt" => "_single")
dir = "inference/" * fname
plot_path = dir * "/plots/"

# Load the data and extract a subset for the analysis
A = readdlm(filename, comments = true, comment_char = '#')
t_all, y_all, yerr_all = A[:, 1], A[:, 2], A[:, 3]
t, y, yerr, x̄, va = extract_subset(rng, fname, t_all, y_all, yerr_all);

# Frequency range for the approx and the prior
f_min, f_max = 1 / (t[end] - t[1]), 1 / minimum(diff(t)) / 2
f0, fM = f_min / 20.0, f_max * 20.0
min_f_b, max_f_b = f0 * 4.0, fM / 4.0

# F var^2 is distributed as a log-normal
μᵥ, σᵥ = -1.5, 1.0;
μₙ, σₙ² = 2μᵥ, 2(σᵥ)^2;
σₙ = sqrt(σₙ²)

# options for the approximation
basis_function = "SHO"
n_components = 20
model = SingleBendingPowerLaw
posterior_checks = true
prior_checks = true

function loglikelihood(y, t, σ, params)

    α₁, f₁, α₂, variance, ν, μ, c = params

    # Rescale the measurement variance
    σ² = ν .* σ .^ 2 ./ (y .- c) .^ 2

    # Make the flux Gaussian
    yn = log.(y .- c)

    # Define power spectral density function
    𝓟 = model(α₁, f₁, α₂)

    # Approximation of the PSD to form a covariance function
    𝓡 = approx(𝓟, f0, fM, n_components, variance, basis_function = basis_function)

    # Build the GP
    f = ScalableGP(μ, 𝓡)

    # sample the conditioned distribution
    return logpdf(f(t, σ²), yn)
end
logl(pars) = loglikelihood(y, t, yerr, pars)

# Priors
function prior_transform(cube)
    α₁ = quantile(Uniform(0.0, 1.25), cube[1])
    f₁ = quantile(LogUniform(min_f_b, max_f_b), cube[2])
    α₂ = quantile(Uniform(α₁, 4.0), cube[3])
    variance = quantile(LogNormal(μₙ, σₙ), cube[4])
    ν = quantile(Gamma(2, 0.5), cube[5])
    μ = quantile(Normal(x̄, 5 * sqrt(va)), cube[6])
    c = quantile(LogUniform(1.0e-6, minimum(y) * 0.99), cube[7])
    return [α₁, f₁, α₂, variance, ν, μ, c]
end
paramnames = ["α₁", "f₁", "α₂", "variance", "ν", "μ", "c"]

if MPI.Comm_rank(comm) == 0 && prior_checks
    unif = rand(rng, 7, 3000) # uniform samples from the unit hypercube
    priors = mapreduce(permutedims, vcat, [prior_transform(unif[:, i]) for i in 1:3000]') # transform the uniform samples to the prior
    run_diagnostics(priors[1:3, :], priors[4, :], f0, fM, model, f_min, f_max, path = plot_path, basis_function = basis_function, n_components = n_components)
end

println("Hello world, I am $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))")

println("Running sampler...")
sampler = ultranest.ReactiveNestedSampler(paramnames, logl, resume = true, transform = prior_transform, log_dir = dir, vectorized = false)
results = sampler.run()
sampler.print_results()
sampler.plot()

if MPI.Comm_rank(comm) == 0 && posterior_checks
    samples = readdlm(dir * "/chains/equal_weighted_post.txt", skipstart = 1)
    run_posterior_predict_checks(samples, paramnames, t, y, yerr, f0, fM, model, true; path = plot_path, basis_function = basis_function, n_components = n_components)
end
