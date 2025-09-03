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
        "API Reference" => "api.md",
        "Installation" => [
            "Quick Start" => "index.md#quick-start",
            "Requirements" => "index.md#installation"
        ],
        "Usage Guide" => [
            "Basic Usage" => "index.md#basic-usage",
            "Advanced Features" => "index.md#advanced-usage",
            "Performance Tips" => "index.md#performance"
        ],
        "Python Integration" => "index.md#python-integration",
        "Troubleshooting" => "index.md#troubleshooting",
        "Contributing" => "index.md#contributing"
    ]
)

deploydocs(
    repo = "github.com/CosmologicalEmulators/Capse.jl.git",
    devbranch = "develop"
