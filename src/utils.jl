"""
extract_subset(rng, prefix, t, y, yerr; n_perc=0.03, take_log=true)

Extract a subset of the data for the analysis and return initial guesses for the mean and variance.

Parameters
----------
rng : Random.MersenneTwister
    Random number generator.
prefix : String
    Prefix for the output files.
t : Array{Float64,1}
    Time array.
y : Array{Float64,1}  
    Time series array.
yerr : Array{Float64,1}
    Time series error array.
n_perc : Float64    
    Percentage of the time series to extract.
take_log : Bool
    If true, log transform the time series for the estimation of the mean and variance.

Returns
-------
t_subset : Array{Float64,1}
    Time array of the subset.
y_subset : Array{Float64,1}
    Time series array of the subset.
yerr_subset : Array{Float64,1}
    Time series error array of the subset.
x̄ : Float64
    Mean of the normal distribution for μ.
va : Float64    
    Variance of the normal distribution for μ.
"""
function extract_subset(rng, prefix, t, y, yerr; n_perc=0.03, take_log=true)

    if !isfile(prefix * "_subset_time_series.txt")
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
            info = "#Estimates computed on the log of the subsets time series\n"
        else
            x = y[subset_indexes]
            info = ""
        end
        # initial guess
        va = var(x)
        x̄ = mean(x)

        open(prefix * "_subset_time_series.txt", "w") do io
            write(io, "#Extracted time series for the analysis (97% of the OG time series)\n# t y yerr\n#Initial guess for the mean and variance from the discarded subset\n#mean: $x̄ va: $va\n$info")
            writedlm(io, hcat(t_subset, y_subset, yerr_subset))
        end
    else
        println("Subset time series file already exists, reading from file")
        for line in eachline(prefix * "_subset_time_series.txt")
            if "#mean: " == line[1:7]
                x̄, va = parse.(Float64, split(line[8:end], " va: "))
                break
            end
        end
        A = readdlm(prefix * "_subset_time_series.txt", comments=true, comment_char='#')
        t_subset, y_subset, yerr_subset = A[:, 1], A[:, 2], A[:, 3]
    end
    return t_subset, y_subset, yerr_subset, x̄, va
end