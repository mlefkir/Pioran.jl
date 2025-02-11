using PkgBenchmark
import Pioran

benchmarkpkg(Pioran,script="benchmark/benchmark_celerite_terms.jl",resultfile="results_celerite_terms.json") 
#benchmarkpkg(Pioran,script="benchmark/benchmarks_celerite_DRWCelerite.jl",resultfile="results_celerite_DRWCelerite.json") 

