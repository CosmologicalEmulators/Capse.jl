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
        sidebar_sitename = false,
        canonical = "https://cosmologicalemulators.github.io/Capse.jl/stable",
        assets = String[],
        analytics = "UA-XXXXXXXXX-X",  # Add Google Analytics ID if available
        collapselevel = 2,
        footer = "Capse.jl v$(pkgversion(Capse)) | [GitHub](https://github.com/CosmologicalEmulators/Capse.jl)"
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
    ],
    repo = "https://github.com/CosmologicalEmulators/Capse.jl/blob/{commit}{path}#{line}",
    strict = false,  # Set to true in CI to catch documentation errors
    checkdocs = :exports,
    linkcheck = :true,
    linkcheck_ignore = [
        "https://zenodo.org/record/8187935"  # Ignore Zenodo links that may be slow
    ]
)

# Deploy documentation
deploydocs(
    repo = "github.com/CosmologicalEmulators/Capse.jl.git",
    devbranch = "develop",
    push_preview = true,
    forcepush = true,
    versions = ["stable" => "v^", "v#.#", "dev" => "develop"]
)
