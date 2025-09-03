using Documenter
using Capse

# Set environment for headless plotting
ENV["GKSwstype"] = "100"

# Add source directory to load path
push!(LOAD_PATH, "../src/")

# Build documentation
makedocs(
    modules = [Capse],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        sidebar_sitename = false
    ),
    sitename = "Capse.jl",
    authors = "Marco Bonici, Federico Bianchini, Jaime Ruiz-Zapatero, and contributors",
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md"
    ]
)

# Deploy documentation
deploydocs(
    repo = "github.com/CosmologicalEmulators/Capse.jl.git",
    devbranch = "develop"
)