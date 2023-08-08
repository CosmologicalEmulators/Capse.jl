module Capse

using Base: @kwdef
using AbstractCosmologicalEmulators

abstract type AbstractCℓEmulators end

"""
    w0waCDMCosmology(TrainedEmulator::AbstractTrainedEmulators, ℓgrid::Array,
    InMinMax::Matrix, OutMinMax::Matrix)

- TrainedEmulator::AbstractTrainedEmulators, the trained emulator

- ℓgrid::Array, the ``\\ell``-grid the emulator has been trained on.

- InMinMax::Matrix, the `Matrix` containing the numbers to perform the MinMax normalization of the input features

- OutMinMax::Matrix, the `Matrix` containing the numbers to perform the MinMax normalization
of the input features
"""
@kwdef mutable struct CℓEmulator <: AbstractCℓEmulators
    TrainedEmulator::AbstractTrainedEmulators
    ℓgrid::Array
    InMinMax::Matrix
    OutMinMax::Matrix
end

"""
    get_Cℓ(CℓEmulator::AbstractCℓEmulators)
Compute the ``C_\ell``on the ``\\ell``-grid the emulator has been trained on.
"""
function get_Cℓ(input_params, CℓEmulator::AbstractCℓEmulators)
    input = deepcopy(input_params)
    maximin_input!(input, CℓEmulator.InMinMax)
    output = Array(run_emulator(input, CℓEmulator.TrainedEmulator))
    inv_maximin_output!(output, CℓEmulator.OutMinMax)
    return output .* exp(input_params[1]-3.)
end

"""
    get_ℓgrid(CℓEmulator::AbstractCℓEmulators)
Return the ``\\ell``-grid the emulator has been trained on.
"""
function get_ℓgrid(CℓEmulator::AbstractCℓEmulators)
    return CℓEmulator.ℓgrid
end

end # module
