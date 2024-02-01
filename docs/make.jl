push!(LOAD_PATH, "../src/")

using Documenter, Pioran

makedocs(sitename="Pioran.jl",
    pages=["Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => "api.md"],
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true"
    ))#, format=:html)
