using PkgBenchmark
import Pioran

benchmarkpkg(Pioran, script = "benchmark/benchmark_celerite.jl", resultfile = "results_celerite_new.json")
#benchmarkpkg(Pioran,script="benchmark/benchmarks_celerite_DRWCelerite.jl",resultfile="results_celerite_DRWCelerite.json")
