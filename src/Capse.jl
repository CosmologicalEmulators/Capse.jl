module Capse

using Base: @kwdef
using Adapt
using AbstractCosmologicalEmulators
import AbstractCosmologicalEmulators: get_emulator_description
import JSON: parsefile
import NPZ: npzread

export get_Cℓ
export ChebyshevInterpolPlan, prepare_Cℓ_interpolation, interp_Cℓ

abstract type AbstractCℓEmulators end

"""
    CℓEmulator(; TrainedEmulator, ℓgrid, InMinMax, OutMinMax, Postprocessing)

Main struct for CMB angular power spectrum emulation.

# Fields
- `TrainedEmulator::AbstractTrainedEmulators`: Trained neural network model
- `ℓgrid::AbstractVector`: Multipole moments (ℓ values) on which the emulator was trained
- `InMinMax::AbstractMatrix`: Min-max normalization parameters for inputs (2×n_params)
- `OutMinMax::AbstractMatrix`: Min-max normalization parameters for outputs (2×n_ℓ)
- `Postprocessing::Function`: Post-processing function with signature `f(input, output, emulator)`

# Example
```julia
# Typically created via load_emulator, but can be constructed manually:
emulator = CℓEmulator(
    TrainedEmulator = trained_nn,
    ℓgrid = collect(2:2500),
    InMinMax = [mins max_vals],  # 2×n_params matrix
    OutMinMax = [mins max_vals],  # 2×n_ℓ matrix
    Postprocessing = (input, output, emu) -> output
)
```

See also: [`load_emulator`](@ref), [`get_Cℓ`](@ref)
"""
@kwdef struct CℓEmulator <: AbstractCℓEmulators
    TrainedEmulator::AbstractTrainedEmulators
    ℓgrid::AbstractVector
    InMinMax::AbstractMatrix
    OutMinMax::AbstractMatrix
    Postprocessing::Function
end

Adapt.@adapt_structure CℓEmulator

"""
    get_Cℓ(input_params, Cℓemu::AbstractCℓEmulators) -> Vector{Float64} or Matrix{Float64}

Compute CMB angular power spectrum ``C_ℓ`` for given cosmological parameters.

# Arguments
- `input_params`: Cosmological parameters
  - `Vector{<:Real}`: Single set of parameters (length must match emulator's expectation)
  - `Matrix{<:Real}`: Multiple parameter sets (size n_params × n_samples)
- `Cℓemu::AbstractCℓEmulators`: The emulator instance

# Returns
- `Vector{Float64}`: Power spectrum values on the emulator's ℓ-grid (single evaluation)
- `Matrix{Float64}`: Power spectra where each column is one spectrum (batch evaluation)

# Throws
- `ArgumentError`: If input dimensions don't match emulator requirements
- `ArgumentError`: If input contains NaN or Inf values
- `AssertionError`: If parameter count doesn't match expected dimensions

# Examples
```julia
# Single evaluation
params = [0.02237, 0.1200, 0.6736, 0.9649, 0.0544, 2.042e-9]
Cℓ = get_Cℓ(params, emulator)

# Batch evaluation (100 cosmologies)
params_batch = rand(6, 100)
Cℓ_batch = get_Cℓ(params_batch, emulator)

# Access specific multipole
ℓ_grid = get_ℓgrid(emulator)
idx_ℓ100 = findfirst(==(100), ℓ_grid)
Cℓ_at_100 = Cℓ[idx_ℓ100]
```

See also: [`load_emulator`](@ref), [`get_ℓgrid`](@ref), [`get_emulator_description`](@ref)
"""
function get_Cℓ(input_params, Cℓemu::AbstractCℓEmulators)
    # Validate input dimensions
    ndims(input_params) > 2 && throw(ArgumentError("Input must be 1D vector or 2D matrix, got $(ndims(input_params))D array"))
    
    # Check if input is a vector or matrix and validate dimensions accordingly
    if ndims(input_params) == 1
        @assert length(input_params) == size(Cℓemu.InMinMax, 1) "Input dimension mismatch: expected $(size(Cℓemu.InMinMax, 1)) parameters, got $(length(input_params))"
    else
        @assert size(input_params, 1) == size(Cℓemu.InMinMax, 1) "Input dimension mismatch: expected $(size(Cℓemu.InMinMax, 1)) parameters per sample, got $(size(input_params, 1))"
    end
    
    # Check for NaN or Inf values
    any(x -> isnan(x) || isinf(x), input_params) && throw(ArgumentError("Input contains NaN or Inf values"))
    
    norm_input = maximin(input_params, Cℓemu.InMinMax)
    output = Array(run_emulator(norm_input, Cℓemu.TrainedEmulator))
    norm_output = inv_maximin(output, Cℓemu.OutMinMax)
    return Cℓemu.Postprocessing(input_params, norm_output, Cℓemu)
end

"""
    ChebyshevInterpolPlan{P, T}

Pre-computed plan for interpolating Cℓ spectra from a Chebyshev ℓ-grid onto a user-defined ℓ-grid.
Construct with `prepare_Cℓ_interpolation(emulator, ℓgrid_new)`.
"""
struct ChebyshevInterpolPlan{P, T}
    cheb_plan::ChebyshevPlan{1, P, T}  # FFT plan for decomposition
    T_mat::Matrix{T}                    # Chebyshev basis at ℓgrid_new, shape (n_new, K+1)
    ℓ_min::T
    ℓ_max::T
    K::Int
    ascending::Bool                     # true if the emulator ℓgrid was stored ascending
end

"""
    prepare_Cℓ_interpolation(Cℓemu, ℓgrid_new; tol=1e-6) -> ChebyshevInterpolPlan

Prepare a reusable interpolation plan for `interp_Cℓ`.

# Arguments
- `Cℓemu::AbstractCℓEmulators`: The emulator whose `ℓgrid` defines the Chebyshev nodes.
- `ℓgrid_new::AbstractVector`: Target ℓ-values for evaluation.
- `tol::Real=1e-6`: Tolerance for Chebyshev grid validation.

# Grid-orientation logic
1. Retrieve `ℓgrid` from the emulator.
2. If ascending, set `ascending=true` and reverse internally (Chebyshev nodes are descending).
3. Compute `ℓ_min`, `ℓ_max`, `K = length(ℓgrid) - 1`.
4. Generate the *expected* Chebyshev grid via `chebpoints(K, ℓ_min, ℓ_max)` and compare
   element-wise to the (possibly reversed) emulator grid.
   - If `norm(expected - observed) / norm(expected) > tol`, emit a `@warn`
     informing the user that the ℓ-grid may not be a Chebyshev grid and accuracy
     could be degraded.
5. Prepare the FFT plan and precompute `T_mat = chebyshev_polynomials(ℓgrid_new, ℓ_min, ℓ_max, K)`.
"""
function prepare_Cℓ_interpolation(Cℓemu::AbstractCℓEmulators,
                                   ℓgrid_new::AbstractVector;
                                   tol::Real = 1e-6)
    ℓgrid = get_ℓgrid(Cℓemu)
    ascending = issorted(ℓgrid)          # ascending  ↔ needs reversal for FFTW
    ℓgrid_desc = ascending ? reverse(ℓgrid) : ℓgrid

    ℓ_min = Float64(last(ℓgrid_desc))   # smallest ℓ (tail of descending vector)
    ℓ_max = Float64(first(ℓgrid_desc))  # largest  ℓ (head of descending vector)
    K = length(ℓgrid) - 1

    # Validate against theoretical Chebyshev nodes
    expected = chebpoints(K, ℓ_min, ℓ_max)  # descending, K+1 points
    
    # We use LinearAlgebra.norm so we need to make sure LinearAlgebra is available
    # but to avoid adding dependencies we can just use maximum absolute difference
    rel_err = maximum(abs.(expected .- ℓgrid_desc)) / maximum(abs.(expected))
    
    if rel_err > tol
        @warn "The emulator ℓ-grid does not appear to be a Chebyshev grid " *
              "(relative max deviation = $(round(rel_err; sigdigits=3))). " *
              "Interpolation accuracy may be degraded."
    end

    cheb_plan = prepare_chebyshev_plan(ℓ_min, ℓ_max, K)
    T_mat     = chebyshev_polynomials(Float64.(ℓgrid_new), ℓ_min, ℓ_max, K)

    return ChebyshevInterpolPlan(cheb_plan, T_mat, ℓ_min, ℓ_max, K, ascending)
end

"""
    interp_Cℓ(Cℓ_vals, plan) -> Vector

Interpolate a single Cℓ spectrum from its Chebyshev ℓ-grid onto the target ℓ-grid
baked into `plan`.

# Arguments
- `Cℓ_vals::AbstractVector`: Spectrum values on the emulator's Chebyshev ℓ-grid (length K+1).
  Must be in the *same orientation* as the emulator's stored ℓ-grid (ascending or descending).
- `plan::ChebyshevInterpolPlan`: Prepared interpolation plan.

# Returns
- `Vector`: Spectrum evaluated at the target ℓ-grid.
"""
function interp_Cℓ(Cℓ_vals::AbstractVector, plan::ChebyshevInterpolPlan)
    # If emulator stored ascending, reverse so that values match descending Chebyshev nodes
    vals_desc = plan.ascending ? reverse(Cℓ_vals) : Cℓ_vals
    coeffs    = chebyshev_decomposition(plan.cheb_plan, vals_desc)
    return plan.T_mat * coeffs
end

"""
    interp_Cℓ(Cℓ_mat, plan) -> Matrix

Interpolate multiple Cℓ spectra (columns of a matrix) onto the target ℓ-grid
baked into `plan`.

# Arguments
- `Cℓ_mat::AbstractMatrix`: Shape `(n_ℓ, n_spectra)`. Each column is a spectrum on the
  emulator's Chebyshev ℓ-grid, in the same orientation as the stored ℓ-grid.
- `plan::ChebyshevInterpolPlan`

# Returns
- `Matrix`: Shape `(length(ℓgrid_new), n_spectra)`.
"""
function interp_Cℓ(Cℓ_mat::AbstractMatrix, plan::ChebyshevInterpolPlan)
    # Reverse rows if the emulator's ℓ-grid was ascending
    mat_desc  = plan.ascending ? reverse(Cℓ_mat; dims=1) : Cℓ_mat
    # chebyshev_decomposition already handles batched (matrix) input column-wise
    coeffs    = chebyshev_decomposition(plan.cheb_plan, mat_desc)  # (K+1, n_spectra)
    return plan.T_mat * coeffs                                       # (n_new, n_spectra)
end

"""
    get_ℓgrid(CℓEmulator::AbstractCℓEmulators) -> AbstractVector

Return the multipole moments (ℓ values) on which the emulator was trained.

# Arguments
- `CℓEmulator::AbstractCℓEmulators`: The emulator instance

# Returns
- `AbstractVector`: Array of ℓ values (typically ranges from 2 to ~2500)

# Example
```julia
ℓ_values = get_ℓgrid(emulator)
println("ℓ range: ", first(ℓ_values), " to ", last(ℓ_values))
println("Number of multipoles: ", length(ℓ_values))

# Find specific multipole
ℓ_target = 100
idx = findfirst(==(ℓ_target), ℓ_values)
```

See also: [`get_Cℓ`](@ref), [`CℓEmulator`](@ref)
"""
function get_ℓgrid(CℓEmulator::AbstractCℓEmulators)
    return CℓEmulator.ℓgrid
end

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

# Example
```julia
get_emulator_description(emulator)
# Output:
# ===================================
# Emulator Description
# ===================================
# Parameters: [ωb, ωc, h, ns, τ, As]
# Architecture: 6 → 64 → 64 → 64 → 64 → 2500
# Training samples: 100,000
# Validation accuracy: 0.1%
# ...
```

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

# Returns
- `CℓEmulator`: Loaded emulator ready for inference

# Required Files
The directory must contain:
1. `l.npy`: Multipole grid
2. `weights.npy`: Neural network weights
3. `inminmax.npy`: Input normalization (2×n_params)
4. `outminmax.npy`: Output normalization (2×n_ℓ)
5. `nn_setup.json`: Architecture description
6. `postprocessing.jl`: Post-processing function

# Examples
```julia
# Basic loading
emulator = load_emulator("/path/to/weights/")

# Use GPU backend
using Lux
emulator = load_emulator("/path/to/weights/", emu=LuxEmulator)

# Custom filenames
emulator = load_emulator(
    "/path/to/weights/",
    weights_file = "model_weights.npy",
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

See also: [`get_Cℓ`](@ref), [`get_emulator_description`](@ref), [`CℓEmulator`](@ref)
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
    Cℓ_emu = Capse.CℓEmulator(
        TrainedEmulator = trained_emu, 
        ℓgrid = ℓ,
        InMinMax = npzread(path*inminmax_file),
        OutMinMax = npzread(path*outminmax_file),
        Postprocessing = include(path*postprocessing_file)
    )
    return Cℓ_emu
end

end # module