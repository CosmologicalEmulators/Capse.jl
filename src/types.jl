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
