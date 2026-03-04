@kwdef struct CℓEmulator <: AbstractCosmologicalEmulators.AbstractCℓEmulators
    TrainedEmulator::AbstractCosmologicalEmulators.AbstractTrainedEmulators
    ℓgrid::AbstractVector
    InMinMax::AbstractMatrix
    OutMinMax::AbstractMatrix
    Postprocessing::Function
end

Adapt.@adapt_structure CℓEmulator

"""
    get_ℓgrid(Cℓemu::AbstractCosmologicalEmulators.AbstractCℓEmulators) -> AbstractVector

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
function AbstractCosmologicalEmulators.get_ℓgrid(Cℓemu::AbstractCosmologicalEmulators.AbstractCℓEmulators)
    return Cℓemu.ℓgrid
end
