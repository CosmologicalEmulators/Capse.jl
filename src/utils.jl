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

!!! warning
    Always check parameter ordering before using an emulator, as different
    training configurations may expect parameters in different orders.

See also: [`load_emulator`](@ref), [`get_Cℓ`](@ref)
"""
function get_emulator_description(Cℓemu::AbstractCℓEmulators)
    if haskey(Cℓemu.TrainedEmulator.Description, "emulator_description")
        get_emulator_description(Cℓemu.TrainedEmulator)
    else
        @warn "No emulator description found!"
    end
    return nothing
end

"""
    load_emulator(path::String; kwargs...) -> CℓEmulator

Load a pre-trained CMB power spectrum emulator from disk.

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
- `nn_setup_file::String = "nn_setup.json"`: Filename for network architecture
- `postprocessing_file::String = "postprocessing.jl"`: Filename for post-processing function
- `postprocessing_name`: Registered postprocessing name. Overrides artifact metadata;
  when neither is provided, load `postprocessing_file` as before.
- `ln10As_index`, `tau_index`: One-based input parameter positions for a named
  postprocessor. Explicit keywords override values in `nn_setup.json`.

# Returns
- `CℓEmulator`: Loaded emulator ready for inference

# Examples
```julia
# Basic loading
emulator = load_emulator("/path/to/weights/")

# Use GPU backend
using Lux
emulator = load_emulator("/path/to/weights/", emu=LuxEmulator)
```

See also: [`get_Cℓ`](@ref), [`get_emulator_description`](@ref), [`CℓEmulator`](@ref)
"""
function _postprocessing_index(NN_dict, name::String, explicit_index)
    description = get(NN_dict, "emulator_description", Dict())
    index = isnothing(explicit_index) ?
        get(NN_dict, name, get(description, name, nothing)) : explicit_index
    (index isa Integer && !(index isa Bool) && 1 <= index <= NN_dict["n_input_features"]) ||
        throw(ArgumentError("$(name) must be a one-based input index between 1 and $(NN_dict["n_input_features"])"))
    return Int(index)
end

function load_emulator(path::String; emu = SimpleChainsEmulator,
    ℓ_file = "l.npy", weights_file = "weights.npy", inminmax_file = "inminmax.npy",
    outminmax_file = "outminmax.npy", nn_setup_file = "nn_setup.json",
    postprocessing_file = "postprocessing.jl", postprocessing_name = nothing,
    ln10As_index = nothing, tau_index = nothing, interpolation = :auto,
    max_spline_knots::Integer = 2048, endpoint_tolerance::Real = 0.1)
    
    # Ensure path ends with /
    path = endswith(path, "/") ? path : path * "/"
    
    NN_dict = parsefile(path*nn_setup_file)
    ℓ = npzread(path*ℓ_file)

    weights = npzread(path*weights_file)
    trained_emu = Capse.init_emulator(NN_dict, weights, emu)
    name = if !isnothing(postprocessing_name)
        postprocessing_name
    elseif haskey(NN_dict, "postprocessing_name")
        NN_dict["postprocessing_name"]
    else
        get(get(NN_dict, "emulator_description", Dict()), "postprocessing_name", nothing)
    end
    postprocessing = if isnothing(name)
        include(path*postprocessing_file)
    else
        name isa Union{Symbol, AbstractString} ||
            throw(ArgumentError("postprocessing_name must be a string or symbol"))
        key = String(name)
        haskey(BUILTIN_POSTPROCESSING, key) ||
            throw(ArgumentError("Unknown postprocessing_name: $(key)"))
        as_index = _postprocessing_index(NN_dict, "ln10As_index", ln10As_index)
        optical_depth_index = key == "as_log" ? 0 :
            _postprocessing_index(NN_dict, "tau_index", tau_index)
        BUILTIN_POSTPROCESSING[key](as_index, optical_depth_index)
    end
    Cℓ_emu = Capse.CℓEmulator(
        TrainedEmulator = trained_emu, 
        ℓgrid = ℓ,
        InMinMax = npzread(path*inminmax_file),
        OutMinMax = npzread(path*outminmax_file),
        Postprocessing = postprocessing,
        interpolation = interpolation,
        max_spline_knots = max_spline_knots,
        endpoint_tolerance = endpoint_tolerance,
    )
    return Cℓ_emu
end
