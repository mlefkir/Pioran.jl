using Pioran
using Test

@testset "Pioran.jl" begin
    # Write your tests here.
    include("test_likelihood.jl")
    include("test_prediction.jl")
    include("test_covariancefunctions.jl")
    include("test_acvf.jl")
    include("test_psd.jl")
    include("test_scalablegp.jl")
    include("test_predict_celerite.jl")
    include("test_carma.jl")
    include("test_plots.jl")
    include("test_mean.jl")

end
