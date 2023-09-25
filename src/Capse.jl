module Capse

using Base: @kwdef
using Adapt
using AbstractCosmologicalEmulators
import AbstractCosmologicalEmulators.get_emulator_description

export get_Cℓ, get_emulator_description

abstract type AbstractCℓEmulators end

"""
    CℓEmulator(TrainedEmulator::AbstractTrainedEmulators, ℓgrid::Array,
    InMinMax::Matrix, OutMinMax::Matrix)

This is the fundamental struct used to obtain the ``C_\\ell``'s from an emulator.
It contains:

- TrainedEmulator::AbstractTrainedEmulators, the trained emulator

- ℓgrid::AbstractVector, the ``\\ell``-grid the emulator has been trained on.

- InMinMax::AbstractMatrix, the `Matrix` used for the MinMax normalization of the input features

- OutMinMax::AbstractMatrix, the `Matrix` used for the MinMax normalization of the output features
"""
@kwdef mutable struct CℓEmulator <: AbstractCℓEmulators
    TrainedEmulator::AbstractTrainedEmulators
    ℓgrid::AbstractVector
    InMinMax::AbstractMatrix
    OutMinMax::AbstractMatrix
end

Adapt.@adapt_structure CℓEmulator

"""
    get_Cℓ(CℓEmulator::AbstractCℓEmulators)
Computes and returns the ``C_\\ell``on the ``\\ell``-grid the emulator has been trained on.
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
Returns the ``\\ell``-grid the emulator has been trained on.
"""
function get_ℓgrid(CℓEmulator::AbstractCℓEmulators)
    return CℓEmulator.ℓgrid
end

function get_emulator_description(Clemu::AbstractCℓEmulators)
    get_emulator_description(Clemu.TrainedEmulator)
end

function load_emulator(path::String, emu = SimpleChainsEmulator,
    ℓ_file = "l.npy", weights_file = "weights.npy", inminmax_file = "inminmax.npy",
    outminmax_file = "outminmax.npy", nn_setup_file = "nn_setup.json")
    NN_dict = JSON.parsefile(path*nn_setup_file)
    ℓ = npzread(path*ℓ_file)

    weights = npzread(path*weights_file)
    trained_emu = Capse.init_emulator(NN_dict, weights, emu)
    Cℓ_emu = Capse.CℓEmulator(TrainedEmulator = trained_emu, ℓgrid = ℓ,
                             InMinMax = npzread(path*inminmax_file),
                             OutMinMax = npzread(path*outminmax_file))
    return Cℓ_emu
end

end # module
