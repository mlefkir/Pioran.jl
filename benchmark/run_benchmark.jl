using PkgBenchmark
using Pkg
# Pkg.activate("..")
using Pioran

#benchmarkpkg(Pioran,script="benchmark/benchmarks_celerite_DRWCelerite.jl",resultfile="results_celerite_DRWCelerite.json")

benchmarkpkg(Pioran, script = "benchmark/benchmarks_celerite.jl", resultfile = "results_celerite_new.json")
