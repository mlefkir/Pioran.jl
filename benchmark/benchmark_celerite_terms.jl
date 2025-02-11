
using BenchmarkTools
using DelimitedFiles
using Pioran
using Distributions
using Random

const SUITE = BenchmarkGroup()

SUITE["inference"] = BenchmarkGroup([])
SUITE["inference2"] = BenchmarkGroup([])

n_samples = 2 .^ (5:15)
n_components = 2 .^ (1:6)


# load data
A = readdlm("simulate_long.txt")
t, y, yerr = collect.(eachcol(A))
σ² = yerr .^ 2

rng = MersenneTwister(1234)

function loglikelihood(a, b, c, d, t, y, σ²)
	return Pioran.logl(a, b, c, d, t, y, σ²)
end

function loglikelihood2(a, b, c, d, t, y, σ²)
	return Pioran.logl2(a, b, c, d, t, y, σ²)
end
a, b, c, d = collect.(eachcol(rand(rng, Float64, (maximum(n_components), 4))))

for J in n_components
	SUITE["inference"][string(J)] = BenchmarkGroup()
    SUITE["inference2"][string(J)] = BenchmarkGroup()

	for N in n_samples
		SUITE["inference"][string(J)][N] = @benchmarkable (loglikelihood($a[1:$J],
                                                            $b[1:$J],
                                                            $c[1:$J],
                                                            $d[1:$J],
                                                            t[1:$N],
                                                            y[1:$N],
                                                            yerr[1:$N]))
        SUITE["inference2"][string(J)][N] = @benchmarkable (loglikelihood2($a[1:$J],
                                                            $b[1:$J],
                                                            $c[1:$J],
                                                            $d[1:$J],
                                                            t[1:$N],
                                                            y[1:$N],
                                                            yerr[1:$N]))
	end
end
tune!(SUITE)

results = run(SUITE, verbose = true, seconds = 1)
