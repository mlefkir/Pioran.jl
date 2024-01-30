using PkgBenchmark
using Pkg
Pkg.activate("..")
import Pioran

benchmarkpkg(Pioran,script="benchmark/benchmarks_celerite_DRWCelerite.jl",resultfile="results_celerite_DRWCelerite.json") 

#benchmarkpkg(Pioran,script="benchmark/benchmarks_celerite.jl",resultfile="results_celerite.json") 