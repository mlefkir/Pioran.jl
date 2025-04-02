# Nested sampling with ultranest

In this example, we show how to use the Python package [`ultranest`](https://johannesbuchner.github.io/UltraNest/index.html) to perform inference on a simple model with nested sampling. We also show how to use the `ultranest`[^1] package with MPI to parallelise the sampling on multiple processes.

## Installation

#### Install the Python environment
It is recommended to use a virtual environment to install the required Python packages. To create a new virtual environment, run the following command in your terminal. If you are using `conda`, you can create a new environment with the following command:
```bash
conda create -n julia_ultranest python=3.10
```
Then, activate the environment:
```bash
conda activate julia_ultranest
```
You can then install `ultranest` with the following command:
```bash
conda install -c conda-forge ultranest
```

#### Install the Julia environment

To use `ultranest` with Julia, you need to install the `PyCall` package.  In the Julia REPL, set the `PYTHON` environment variable to the path of the Python executable in the virtual environment you created earlier. For example:

```julia
ENV["PYTHON"] = "/opt/miniconda3/envs/julia_ultranest/bin/python"
```
You can do this by running the following commands in the Julia REPL:
```julia
using Pkg; Pkg.add("PyCall")
Pkg.build("PyCall")
```

We can check that the Python environment has been set correctly by running the following commands in the Julia REPL:
```julia
using PyCall
PyCall.python
```

Finally, `ultranest` can be loaded in Julia with the following commands:
```julia
ultranest = pyimport("ultranest")
```

#### Install MPI.jl (optional)

If you want to parallelise the sampling with MPI, you first have to install MPI on your system. On Ubuntu, you can install the `openmpi` package with the following command:
```bash
sudo apt-get install openmpi-bin libopenmpi-dev
```
You can then install `MPI.jl` and `MPIPreferences.jl` in Julia with the following commands:
```julia
using Pkg; Pkg.add("MPI")
Pkg.add("MPIPreferences")
```
As detailed in the official documentation of `MPI.jl`[^2], the system MPI binary can be linked to Julia with the following command:
```bash
julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
```
More information on how to configure MPI on can be found in the official documentation of `MPI.jl`.

The following lines of code can be used to initialise MPI in a Julia script:
```julia
using MPI
MPI.Init()
```

## Modelling with ultranest

We assume the reader is familiar with nested sampling and the `ultranest` package.

### Priors

Priors are defined using the `quantile` function and probability distributions from the [`Distributions`](https://juliastats.org/Distributions.jl/stable/) package. The `prior_transform` function is then used to transform the unit cube to the prior space as shown in the following example:

```julia
using Distributions

function prior_transform(cube)
    α₁ = quantile(Uniform(0.0, 1.25), cube[1])
    f₁ = quantile(LogUniform(1e-3, 1e3), cube[2])
    α₂ = quantile(Uniform(α₁, 4.0), cube[3])
    variance = quantile(LogNormal(-2., 1.2), cube[4])
    ν = quantile(Gamma(2, 0.5), cube[5])
    μ = quantile(Normal(0., 2.), cube[6])
    return [α₁, f₁, α₂, variance, ν, μ]
end
```
### Model and likelihood

The `loglikelihood` function is then defined to compute the likelihood of the model given the data and the parameters. The various instructions in the function are detailed in previous sections or examples.

```julia
using Pioran # hide
function loglikelihood(y, t, σ, params)
    α₁, f₁, α₂, variance, ν, μ = params
    σ² = ν .* σ .^ 2
    𝓟 = model(α₁, f₁, α₂)
    𝓡 = approx(𝓟, f0, fM, 20, variance)
    f = ScalableGP(μ, 𝓡)
    return logpdf(f(t, σ²), y)
end
paramnames = ["α₁", "f₁", "α₂", "variance", "ν", "μ"]
logl(params) = loglikelihood(y, t, yerr, params)
```

We can then initialise the `ultranest` sampler with the following command:

```julia
sampler = ultranest.ReactiveNestedSampler(paramnames, logl, resume=true, transform=prior_transform, log_dir="inference_dir", vectorized=false)
```
The inference can be started with the following command:

```julia
results = sampler.run()
```
Finally, the results can be printed and plotted with the following commands:

```julia
sampler.print_results()
sampler.plot()
```

### Parallel sampling with MPI

If you want to parallelise the sampling with MPI to speed-up the computation, follow the steps presented before and run the script with the following command:
```bash
mpiexec -n 4 julia script.jl
```
where `4` is the number of processes to use.

## Full example

```julia
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
A = readdlm(filename, comments=true, comment_char='#')
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
    𝓡 = approx(𝓟, f0, fM, n_components, variance, basis_function=basis_function)

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
    c = quantile(LogUniform(1e-6, minimum(y) * 0.99), cube[7])
    return [α₁, f₁, α₂, variance, ν, μ, c]
end
paramnames = ["α₁", "f₁", "α₂", "variance", "ν", "μ", "c"]

if MPI.Comm_rank(comm) == 0 && prior_checks
    unif = rand(rng, 7, 3000) # uniform samples from the unit hypercube
    priors = mapreduce(permutedims, hcat, [prior_transform(unif[:, i]) for i in 1:3000]')# transform the uniform samples to the prior
    run_diagnostics(priors[1:3, :], priors[4, :], f0, fM, model, f_min, f_max, path=plot_path, basis_function=basis_function, n_components=n_components)
end

println("Hello world, I am $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))")

println("Running sampler...")
sampler = ultranest.ReactiveNestedSampler(paramnames, logl, resume=true, transform=prior_transform, log_dir=dir, vectorized=false)
results = sampler.run()
sampler.print_results()
sampler.plot()

if MPI.Comm_rank(comm) == 0 && posterior_checks
    samples = readdlm(dir * "/chains/equal_weighted_post.txt", skipstart=1)
    run_posterior_predict_checks(samples, paramnames, t, y, yerr, f0, fM, model, true; path=plot_path, basis_function=basis_function, n_components=n_components)
end
```

## References


[^1]: [https://johannesbuchner.github.io/UltraNest/index.html](https://johannesbuchner.github.io/UltraNest/index.html)

[^2]: [https://juliaparallel.org/MPI.jl/stable/configuration/](https://juliaparallel.org/MPI.jl/stable/configuration/)