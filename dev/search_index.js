var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API Reference","title":"API","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"SHO\nCelerite\nExp\nposterior\napprox\nlog_likelihood\nplot_mean_approx\nplot_quantiles_approx\nplot_boxplot_psd_approx\nrun_diagnostics\nextract_subset","category":"page"},{"location":"api/#Pioran.SHO","page":"API Reference","title":"Pioran.SHO","text":"SHO(σ, ω₀, Q)\n\nConstruct a simple harmonic oscillator covariance function with parameters σ, ω₀, Q. Where σ is the amplitude, ω₀ is the angular frequency and Q is the quality factor.\n\n\n\n\n\n","category":"type"},{"location":"api/#Pioran.Celerite","page":"API Reference","title":"Pioran.Celerite","text":"Celerite(a, b, c, d)\n\nConstruct a celerite covariance function with parameters a, b, c, d.\n\n\n\n\n\n","category":"type"},{"location":"api/#Pioran.Exp","page":"API Reference","title":"Pioran.Exp","text":"Exp(σ,α)\n\nConstruct a exponential covariance function with parameters σ, α. Where σ is the amplitude and α is the decay rate.\n\n\n\n\n\n","category":"type"},{"location":"api/#Pioran.posterior","page":"API Reference","title":"Pioran.posterior","text":"posterior(f::ScalableGP, y::AbstractVecOrMat{<:Real})\n\nCompute the posterior GP given the GP f and the data y.\n\n\n\n\n\n","category":"function"},{"location":"api/#Pioran.approx","page":"API Reference","title":"Pioran.approx","text":"approx(psdmodel, f0, fM, ncomponents=20, var=1.0; basis_function=\"SHO\")\n\nApproximate the PSD with a sum of SHO functions\n\npsd_model: the model for the power spectral density\nf0: the lowest frequency\nfM: the highest frequency\nn_components: the number of components to use\nvar: the variance of the process\nbasis_function: the basis function to use\n\n\n\n\n\n","category":"function"},{"location":"api/#Pioran.log_likelihood","page":"API Reference","title":"Pioran.log_likelihood","text":"Compute the log likelihood of the GP at the points τ given the data y and time t.\n\n\n\n\n\n","category":"function"},{"location":"api/#Pioran.plot_mean_approx","page":"API Reference","title":"Pioran.plot_mean_approx","text":"Plot the frequency-averaged residuals and ratios \n\n\n\n\n\n","category":"function"},{"location":"api/#Pioran.plot_quantiles_approx","page":"API Reference","title":"Pioran.plot_quantiles_approx","text":"plot_quantiles_approx(f, f_min, f_max, residuals, ratios; path=\"\")\n\nPlot the quantiles of the residuals and ratios (with respect to the approximated PSD) of the PSD \n\n\n\n\n\n","category":"function"},{"location":"api/#Pioran.plot_boxplot_psd_approx","page":"API Reference","title":"Pioran.plot_boxplot_psd_approx","text":"Plot the boxplot of the residuals and ratios for the PSD approximation \n\n\n\n\n\n","category":"function"},{"location":"api/#Pioran.run_diagnostics","page":"API Reference","title":"Pioran.run_diagnostics","text":"Run the diagnostics for the PSD approximation\n\n\n\n\n\n","category":"function"},{"location":"api/#Pioran.extract_subset","page":"API Reference","title":"Pioran.extract_subset","text":"extractsubset(rng, prefix, t, y, yerr; nperc=0.03, take_log=true)\n\nExtract a subset of the data for the analysis and return initial guesses for the mean and variance.\n\nParameters\n\nrng : Random.MersenneTwister     Random number generator. prefix : String     Prefix for the output files. t : Array{Float64,1}     Time array. y : Array{Float64,1}       Time series array. yerr : Array{Float64,1}     Time series error array. nperc : Float64         Percentage of the time series to extract. takelog : Bool     If true, log transform the time series for the estimation of the mean and variance.\n\nReturns\n\ntsubset : Array{Float64,1}     Time array of the subset. ysubset : Array{Float64,1}     Time series array of the subset. yerr_subset : Array{Float64,1}     Time series error array of the subset. x̄ : Float64     Mean of the normal distribution for μ. va : Float64         Variance of the normal distribution for μ.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API Reference","title":"API Reference","text":"<!– @autodocs Modules = [Pioran] Order   = [:function, :type] –>","category":"page"},{"location":"getting_started/#Getting-Started","page":"Getting Started","title":"Getting Started","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"As Pioran.jl is written in Julia, you need to install Julia first. Please refer to the official website for the installation.","category":"page"},{"location":"getting_started/#Installation","page":"Getting Started","title":"Installation","text":"","category":"section"},{"location":"#Pioran.jl-Documentation","page":"Home","title":"Pioran.jl Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}