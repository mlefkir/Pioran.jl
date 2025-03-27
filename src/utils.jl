# COV_EXCL_START
"""
    extract_subset(rng, prefix, t, y, yerr; n_perc=0.03, take_log=true, suffix="")
    extract_subset(seed, prefix, t, y, yerr; n_perc=0.03, take_log=true)

Extract a subset of the data for the analysis and return initial guesses for the mean and variance.
Either a random number generator or a seed can be provided.

# Arguments
- `seed::Int64` : Seed for the random number generator.
- `rng::AbstractRNG` : Random number generator.
- `prefix::String` : Prefix for the output files.
- `t::Array{Float64,1}` : Time array.
- `y::Array{Float64,1}` : Time series array.
- `yerr::Array{Float64,1}` : Time series error array.
- `n_perc::Float64` : Percentage of the time series to extract.
- `take_log::Bool` : If true, log transform the time series for the estimation of the mean and variance.
- `suffix::String` : Suffix for the output files.

# Return
- `t_subset::Array{Float64,1}` : Time array of the subset.
- `y_subset::Array{Float64,1}` : Time series array of the subset.
- `yerr_subset::Array{Float64,1}` : Time series error array of the subset.
- `x̄::Float64` : Mean of the normal distribution for μ.
- `va::Float64` : Variance of the normal distribution for μ.
"""
function extract_subset(rng::AbstractRNG, prefix, t, y, yerr; n_perc=0.03, take_log=true, suffix="")

    filename = prefix * "_subset_time_series" *suffix*".txt"
    println("Filename: ", filename)
    if !isfile(filename)
        println("Extracting subset time series")
        # total number of points
        n_points = length(t)
        # number of points to remove
        n_samples = Int(round(n_points * n_perc))
        if n_samples <= 1
            n_samples = 3
        end
        # indexes of points to remove
        subset = sample(rng, range(1, n_points), n_samples, replace=false)
        # indexes of points to keep
        x = range(1, n_points)
        subset_indexes = findall(x -> x ∈ subset, x)
        extract_subset = findall(x -> x ∉ subset, x)

        t_subset = t[extract_subset]
        y_subset = y[extract_subset]
        yerr_subset = yerr[extract_subset]
        # log transform
        if take_log
            x = log.(y[subset_indexes])
            info = "#Estimates computed on the log of the subset time series\n"
        else
            x = y[subset_indexes]
            info = ""
        end
        # initial guess
        va = var(x)
        x̄ = mean(x)

        open(filename, "w") do io
            write(io, "#Extracted time series for the analysis (97% of the OG time series)\n# t y yerr\n#Initial guess for the mean and variance from the discarded subset\n#mean: $x̄ va: $va\n$info")
            writedlm(io, hcat(t_subset, y_subset, yerr_subset))
        end
    else
        println("Subset time series file already exists, reading from file")
        for line in eachline(filename)
            if "#mean: " == line[1:7]
                x̄, va = parse.(Float64, split(line[8:end], " va: "))
                break
            end
        end
        A = readdlm(filename, comments=true, comment_char='#')
        t_subset, y_subset, yerr_subset = A[:, 1], A[:, 2], A[:, 3]
    end
    return t_subset, y_subset, yerr_subset, x̄, va
end

function extract_subset(seed::Int64, prefix, t, y, yerr; n_perc=0.03, take_log=true, suffix="")
    rng = MersenneTwister(seed)
    return extract_subset(rng, prefix, t, y, yerr, n_perc=n_perc, take_log=take_log,suffix=suffix)

end

"""
    separate_samples(samples, paramnames, with_log_transform::Bool)

Separate the samples into the parameters of the model and the parameters of the power spectral density.

"""
function separate_samples(samples,paramnames,with_log_transform::Bool)
    
    # try to find all the parameters except the PSD parameters
    # gamma
    n_samples = size(samples,1)
    collected_pars = []
    gamma_index = findall(name->name=="γ", paramnames)
    if isempty(gamma_index)
        samples_γ = ones(n_samples)
    else
        samples_γ = samples[:,gamma_index[1]]
        push!(collected_pars,gamma_index[1])
    end

    # nu
    nu_index = findall(name->name=="ν", paramnames)
    if isempty(nu_index)
        samples_ν = ones(n_samples)
    else
        samples_ν = samples[:,nu_index[1]]
        push!(collected_pars,nu_index[1])
    end
    # const
    if with_log_transform
        c_index = findall(name->name=="c", paramnames)
        if isempty(c_index)
            samples_c = zeros(n_samples)
        else
            samples_c = samples[:,c_index[1]]
            push!(collected_pars,c_index[1])
        end
    else
        samples_c = nothing
    end
    # mu
    mu_index = findall(name->name=="μ", paramnames)
    if isempty(mu_index)
        samples_μ = zeros(n_samples)
    else
        samples_μ = samples[:,mu_index[1]]
        push!(collected_pars,mu_index[1])
    end
    # var 
    variance_index = findall(name->name=="variance", paramnames)
    if isempty(variance_index)
        error("The 'variance' parameter is not found in the parameter names")
    else
        samples_variance = samples[:,variance_index[1]]
        push!(collected_pars,variance_index[1])
    end
    # PSD parameters
    allpars = collect(1:length(paramnames))
    remaining = setdiff(allpars, collected_pars)
    println("Deducing that the PSD parameter are: ", paramnames[remaining])
    println("Deducing that the hyperparameter are: ", paramnames[collected_pars])

    samples_𝓟 = samples[:,remaining]
    return samples_𝓟, samples_variance, samples_ν, samples_μ, samples_c
end

"""
    check_conjugate_pair(r::Vector{Complex})

Check if the roots are complex conjugate pairs and negative real parts
Returns true if the roots are complex conjugate pairs and false otherwise
"""
function check_conjugate_pair(r::Vector{Complex})
    if any(real.(r).>0)
        return false
    end
    n = length(r)
    if n%2==0
        for i in 1:2:n
            if r[i] != conj(r[i+1])
                return false
            end
        end
    else
        for i in 1:2:n-1
            if r[i] != conj(r[i+1])
                return false
            end
        end

    end
    return true
end

"""
    check_roots_bounds(r::Vector{Complex},f_min::Float64,f_max::Float64)

Check if the roots are within the bounds of the frequency range
"""
function check_roots_bounds(r::Vector{Complex},f_min::Float64,f_max::Float64)
    if all(-f_max.<real.(r).<-f_min) && all(-f_max.<imag.(r).<f_max)
        return true
    end
    return false
end

"""
    check_order_imag_roots(r::Vector{Complex})

Check if the imaginary parts of the roots are in ascending order
"""
function check_order_imag_roots(r)
	n = length(r)
	if n % 2 == 0
		perm = sortperm((imag.(r[1:2:n])), rev = false)
	else
		perm = sortperm((imag.(r[1:2:n-1])), rev = false)
	end
	if perm != range(1, length(perm))
		return false
	end
	return true
end
# COV_EXCL_STOP
