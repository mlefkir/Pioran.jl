using PkgBenchmark
using Pkg
Pkg.activate(".")
import Pioran

benchmarkpkg(Pioran,script="benchmarks_celerite.jl",resultfile="results_celerite.json") 