push!(LOAD_PATH, "../src/")

using Documenter, Pioran

BASIC_PAGES = ["explanation.md", "modelling.md","simulations.md", "timeseries.md", "diagnostics.md"]
ADVANCED_PAGES = ["turing.md", "ultranest.md"]

makedocs(sitename="Pioran.jl",
    pages=["Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Basic usage" => BASIC_PAGES,
        "Inference" => ADVANCED_PAGES,
        "API Reference" => "api.md"],
    format=Documenter.HTML(description="Pioran.jl: A Julia package for power spectral density estimation using scalable Gaussian processes.",
        prettyurls=true,#get(ENV, "CI", nothing) == "true",
#  collapselevel=1
    ))#, format=:html)

deploydocs(
    repo="github.com/mlefkir/Pioran.jl.git",
)