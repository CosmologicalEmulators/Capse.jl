abstract type AbstractCℓEmulators end

@kwdef struct CℓEmulator <: AbstractCℓEmulators
    TrainedEmulator::AbstractTrainedEmulators
    ℓgrid::AbstractVector
    InMinMax::AbstractMatrix
    OutMinMax::AbstractMatrix
    Postprocessing::Function
end

Adapt.@adapt_structure CℓEmulator

"""
    get_ℓgrid(Cℓemu::AbstractCℓEmulators) -> AbstractVector

Return the ℓ-grid associated with the given emulator.

# Arguments
- `Cℓemu::AbstractCℓEmulators`: The emulator instance

# Returns
- `AbstractVector`: The ℓ-grid array

# Example
```julia
ℓ_grid = get_ℓgrid(emulator)
```
"""
function get_ℓgrid(Cℓemu::AbstractCℓEmulators)
    return Cℓemu.ℓgrid
end
