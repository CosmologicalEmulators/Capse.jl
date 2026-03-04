"""
    get_Cℓ(input_params, Cℓemu::AbstractCℓEmulators) -> AbstractVector

Evaluate the emulator to predict the Cℓ spectrum for the given cosmological parameters.

# Arguments
- `input_params::AbstractVector` or `AbstractMatrix`: The cosmological parameters
  (e.g., `[ωb, ωc, H0, ns, ln10^{10}As, τ]`). If a matrix is provided, it must be
  of size `(n_params, n_samples)`.
- `Cℓemu::AbstractCℓEmulators`: The loaded emulator instance.

# Returns
- `AbstractVector` or `AbstractMatrix`: The predicted Cℓ spectrum. Output shape matches
  the number of samples in `input_params`.

# Example
```julia
# Single cosmology evaluation
params = [0.022, 0.12, 67.0, 0.96, 3.0, 0.05]
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
    input = @. (input_params - Cℓemu.InMinMax[:,1]) / (Cℓemu.InMinMax[:,2] - Cℓemu.InMinMax[:,1])
    norm_output = Cℓemu.TrainedEmulator(input)
    return Cℓemu.Postprocessing(input_params, norm_output, Cℓemu)
end

"""
    get_Cℓ(input_params, Cℓemu::AbstractCℓEmulators, plan::ChebyshevInterpolPlan)

Evaluate the emulator and interpolate the output in one shot onto the target ℓ-grid defined by `plan`.
"""
function get_Cℓ(input_params, Cℓemu::AbstractCℓEmulators, plan::ChebyshevInterpolPlan)
    Cℓ_eval = get_Cℓ(input_params, Cℓemu)
    return interp_Cℓ(Cℓ_eval, plan)
end
