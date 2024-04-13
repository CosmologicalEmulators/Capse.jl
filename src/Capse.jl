module Capse

using Base: @kwdef
using Adapt
using AbstractCosmologicalEmulators
using IRTools
import AbstractCosmologicalEmulators.get_emulator_description
import JSON.parsefile
import NPZ.npzread

export get_Cℓ

abstract type AbstractCℓEmulators end

"""
    CℓEmulator(TrainedEmulator::AbstractTrainedEmulators, ℓgrid::Array,
    InMinMax::Matrix, OutMinMax::Matrix)

This is the fundamental struct used to obtain the ``C_\\ell``'s from an emulator.
It contains:

- `TrainedEmulator::AbstractTrainedEmulators`, the trained emulator

- `ℓgrid::AbstractVector`, the ``\\ell``-grid the emulator has been trained on.

- `InMinMax::AbstractMatrix`, the `Matrix` used for the MinMax normalization of the input features

- `OutMinMax::AbstractMatrix`, the `Matrix` used for the MinMax normalization of the output features

- `Postprocessing::Function`, the `Function` used for the postprocessing of the NN output
"""
@kwdef mutable struct CℓEmulator <: AbstractCℓEmulators
    TrainedEmulator::AbstractTrainedEmulators
    ℓgrid::AbstractVector
    InMinMax::AbstractMatrix
    OutMinMax::AbstractMatrix
    Postprocessing::Function
end

Adapt.@adapt_structure CℓEmulator

"""
    get_Cℓ(input_params, Cℓemu::AbstractCℓEmulators)
Computes and returns the ``C_\\ell``'s on the ``\\ell``-grid the emulator has been trained on given input array `input_params`.

"""
function get_Cℓ(input_params, Cℓemu::AbstractCℓEmulators)
    norm_input = maximin_input(input_params, Cℓemu.InMinMax)
    output = Array(run_emulator(norm_input, Cℓemu.TrainedEmulator))
    norm_output = inv_maximin_output(output, Cℓemu.OutMinMax)
    return Cℓemu.Postprocessing(input_params, norm_output, Cℓemu)
end

"""
    get_ℓgrid(CℓEmulator::AbstractCℓEmulators)
Returns the ``\\ell``-grid the emulator has been trained on.
"""
function get_ℓgrid(CℓEmulator::AbstractCℓEmulators)
    return CℓEmulator.ℓgrid
end

"""
    get_emulator_description(Cℓemu::AbstractCℓEmulators)
Print on screen the emulator description.
"""
function get_emulator_description(Cℓemu::AbstractCℓEmulators)
    get_emulator_description(Cℓemu.TrainedEmulator)
end

"""
    load_emulator(path::String, emu_backend::AbstractTrainedEmulators)
Load the emulator with the files in the folder `path`, using the backend defined by `emu_backend`.
The following keyword arguments are used to specify the name of the files used to load the emulator:
- `ℓ_file`, default `l.npy`
- `weights_file`, default `weights.npy`
- `inminmax_file`, default `inminmax.npy`
- `outminmax_file`, default `outminmax.npy`
- `nn_setup_file`, default `nn_setup.json`
- `postprocessing_file`, default `postprocessing.jl`
If the corresponding file in the folder you are trying to load have different names,
 change the default values accordingly.
"""
function load_emulator(path::String; emu = SimpleChainsEmulator,
    ℓ_file = "l.npy", weights_file = "weights.npy", inminmax_file = "inminmax.npy",
    outminmax_file = "outminmax.npy", nn_setup_file = "nn_setup.json",
    postprocessing_file = "postprocessing.jl")
    NN_dict = parsefile(path*nn_setup_file)
    ℓ = npzread(path*ℓ_file)
    include(path*postprocessing_file)
    #we assume there is a postprocessing() function in the postprocessing_file
    #TODO not exactly elegant. Maybe ask to more proficient people?
    _postprocessing = @code_ir postprocessing(1,2,3)
    Postprocessing = IRTools.func(_postprocessing)
    postprocess(a,b,c) = Postprocessing(_postprocessing, a,b,c)

    weights = npzread(path*weights_file)
    trained_emu = Capse.init_emulator(NN_dict, weights, emu)
    Cℓ_emu = Capse.CℓEmulator(TrainedEmulator = trained_emu, ℓgrid = ℓ,
                             InMinMax = npzread(path*inminmax_file),
                             OutMinMax = npzread(path*outminmax_file),
                             Postprocessing = postprocess)
    return Cℓ_emu
end

end # module
