# Nested sampling with ultranest

In this example, we show how to use the Python package `ultranest`[^1] to perform inference on a simple model with nested sampling. We also show how to use the `ultranest` package with MPI to parallelise the sampling on multiple processes.

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
Then the code can be run with the following command:
```bash
mpiexec -n 4 julia script.jl
```
where `4` is the number of processes to use.

## Modelling with ultranest

We assume the reader is familiar with nested sampling and the `ultranest` package. 

### Priors

Priors are defined using the `quantile` function and probability distributions from the `Distributions` package. The `prior_transform` function is then used to transform the unit cube to the prior space as shown in the following example:

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

```bash
mpiexec -n 4 julia script.jl
```

## Full example

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../examples/single_pl.jl",String))
```
""")
````

## References


[^1]: [https://johannesbuchner.github.io/UltraNest/index.html](https://johannesbuchner.github.io/UltraNest/index.html)

[^2]: [https://juliaparallel.org/MPI.jl/stable/configuration/](https://juliaparallel.org/MPI.jl/stable/configuration/)