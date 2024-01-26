using Pioran
using Test

@testset "Pioran.jl" begin
    # Write your tests here.
    include("test_likelihood.jl")
    include("test_prediction.jl")
end

