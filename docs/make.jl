using Documenter
using Capse

ENV["GKSwstype"] = "100"

push!(LOAD_PATH,"../src/")

makedocs(
    modules = [Capse],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
    sidebar_sitename=true),
    sitename = "Capse.jl",
    authors  = "Marco Bonici",
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/CosmologicalEmulators/Capse.jl.git",
    devbranch = "develop"
)
