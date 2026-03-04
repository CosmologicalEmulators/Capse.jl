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

# Internal helper: run the neural network and invert normalisation, but skip postprocessing.
# Used by tests and by the one-shot get_Cℓ(input_params, emu, plan) method.
function get_emulator_output(input_params, Cℓemu::AbstractCℓEmulators)
    norm_input  = maximin(input_params, Cℓemu.InMinMax)
    output      = Array(run_emulator(norm_input, Cℓemu.TrainedEmulator))
    return inv_maximin(output, Cℓemu.OutMinMax)
end

"""
    get_Cℓ(input_params, Cℓemu::AbstractCℓEmulators, plan::ChebyshevInterpolPlan)

One-shot convenience method: evaluate the emulator **and** interpolate onto the
target ℓ-grid baked into `plan`.

# Returns
- `Vector` (or `Matrix` for batched `input_params`) on the target ℓ-grid.
"""
function get_Cℓ(input_params, Cℓemu::AbstractCℓEmulators, plan::ChebyshevInterpolPlan)
    norm_output = get_emulator_output(input_params, Cℓemu)
    Cℓ_pp = Cℓemu.Postprocessing(input_params, norm_output, Cℓemu)
    return interp_Cℓ(Cℓ_pp, plan)
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
```

See also: [`get_Cℓ`](@ref), [`CℓEmulator`](@ref)
"""
function get_ℓgrid(CℓEmulator::AbstractCℓEmulators)
    return CℓEmulator.ℓgrid
end
