"""
    get_emulator_description(Cℓemu::AbstractCℓEmulators) -> Nothing

Display detailed information about the emulator configuration.

Prints to stdout:
- Cosmological parameter names and ordering
- Network architecture details
- Training configuration
- Accuracy metrics (if available)
- Version information

# Arguments
- `Cℓemu::AbstractCℓEmulators`: The emulator instance

# Returns
- `nothing` (information is printed to stdout)
"""
function AbstractCosmologicalEmulators.get_emulator_description(Cℓemu::AbstractCℓEmulators)
    println(Cℓemu.TrainedEmulator.Description)
end

"""
    load_emulator(path::String; kwargs...) -> CℓEmulator

Load a pre-trained `CℓEmulator` from disk.

# Arguments
- `path::String`: Directory path containing the emulator files (must end with '/')

# Keyword Arguments
- `emu::Type = SimpleChainsEmulator`: Backend to use
  - `SimpleChainsEmulator`: CPU-optimized (default)
  - `LuxEmulator`: GPU-capable
- `ℓ_file::String = "l.npy"`: Filename for ℓ-grid
- `weights_file::String = "weights.npy"`: Filename for network weights
- `inminmax_file::String = "inminmax.npy"`: Filename for input normalization
- `outminmax_file::String = "outminmax.npy"`: Filename for output normalization
- `nn_setup_file::String = "nn_setup.json"`: Filename for network architecture definition
- `postprocessing_file::String = "postprocessing.jl"`: Filename for postprocessing script (falls back to Python default logic if `.jl` missing)

# Example
```julia
using Capse

# Load default configuration
emulator = load_emulator("path/to/weights/")

# Load with specific backend and custom files
emulator = load_emulator("path/to/weights/", 
    emu = LuxEmulator,
    ℓ_file = "multipoles.npy"
)

# Check what was loaded
get_emulator_description(emulator)
```

# Errors
- `SystemError`: If path doesn't exist or files are missing
- `LoadError`: If files are corrupted or incompatible

!!! tip
    Pre-trained emulators are available on [Zenodo](https://zenodo.org/record/8187935).
    Download and extract the weights folder, then load with this function.

See also: [`get_Cℓ`](@ref), [`get_emulator_description`](@ref), [`get_ℓgrid`](@ref)
"""
function load_emulator(path::String; emu = SimpleChainsEmulator,
    ℓ_file = "l.npy", weights_file = "weights.npy", inminmax_file = "inminmax.npy",
    outminmax_file = "outminmax.npy", nn_setup_file = "nn_setup.json",
    postprocessing_file = "postprocessing.jl")
    
    # Ensure path ends with /
    path = endswith(path, "/") ? path : path * "/"
    
    NN_dict = parsefile(path*nn_setup_file)
    ℓ = npzread(path*ℓ_file)

    weights = npzread(path*weights_file)
    trained_emu = Capse.init_emulator(NN_dict, weights, emu)
    
    postproc_path = path*postprocessing_file
    postproc_obj = if isfile(postproc_path)
        include(postproc_path)
    else
        # Fallback to the known py version if no .jl is present in the archive
        (input, output, Cℓemu) -> output .* exp(input[1] - 3.0)
    end
    
    Cℓ_emu = Capse.CℓEmulator(
        TrainedEmulator = trained_emu, 
        ℓgrid = ℓ,
        InMinMax = npzread(path*inminmax_file),
        OutMinMax = npzread(path*outminmax_file),
        Postprocessing = postproc_obj
    )
    return Cℓ_emu
end
