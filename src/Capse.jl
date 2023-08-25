module Capse

using Base: @kwdef
using AbstractCosmologicalEmulators
import AbstractCosmologicalEmulators.get_emulator_description
using FastChebInterp

abstract type AbstractCℓEmulators end

"""
    CℓEmulator(TrainedEmulator::AbstractTrainedEmulators, ℓgrid::Array,
    InMinMax::Matrix, OutMinMax::Matrix)

This is the fundamental struct used to obtain the ``C_\\ell``'s from an emulator.
It contains:

- TrainedEmulator::AbstractTrainedEmulators, the trained emulator

- ℓgrid::Array, the ``\\ell``-grid used to train the emulator

- InMinMax::Matrix, the `Matrix` used for the MinMax normalization of the input features

- OutMinMax::Matrix, the `Matrix` used for the MinMax normalization of the output features

- PolyGrid::Matrix, the `Matrix` which contains the Chebyshev polynomials evaluated on the requested ``\\ell``-grid
"""
@kwdef mutable struct CℓEmulator <: AbstractCℓEmulators
    TrainedEmulator::AbstractTrainedEmulators
    ℓgrid::Array
    InMinMax::Matrix
    OutMinMax::Matrix
    PolyGrid::Matrix = zeros(200,200)
end

"""
    get_Cℓ(CℓEmulator::AbstractCℓEmulators)
Computes and returns the ``C_\\ell``on the ``\\ell``-grid that was previously passed to `eval_polygrid!`.
"""
function get_Cℓ(input_params, CℓEmulator::AbstractCℓEmulators)
    chebcoefs = get_chebcoefs(input_params, CℓEmulator)
    return CℓEmulator.PolyGrid * chebcoefs
end

"""
    get_Cℓ(CℓEmulator::AbstractCℓEmulators)
Computes and returns the Chebyshev expansion coefficients ``a_{\\ell m}``.
"""
function get_chebcoefs(input_params, Clemulator::AbstractCℓEmulators)
    Clgrid = _get_Cℓ(input_params, Clemulator)
    return FastChebInterp.chebcoefs(Clgrid)
end

function _get_Cℓ(input_params, CℓEmulator::AbstractCℓEmulators)
    input = deepcopy(input_params)
    maximin_input!(input, CℓEmulator.InMinMax)
    output = Array(run_emulator(input, CℓEmulator.TrainedEmulator))
    inv_maximin_output!(output, CℓEmulator.OutMinMax)
    return output .* exp(input_params[1]-3.)
end

function eval_polygrid!(Cl::CℓEmulator, myℓgrid::Array)
    cb = FastChebInterp.ChebPoly(zeros(length(Cl.ℓgrid)),
                                 FastChebInterp.SVector{1}([Cl.ℓgrid[begin]]),
                                 FastChebInterp.SVector{1}([Cl.ℓgrid[end]]))
    Cl.PolyGrid = _eval_polygrid(cb, myℓgrid)
    return nothing
end

function eval_polygrid!(Cl::CℓEmulator)
    eval_polygrid!(Cl, Cl.ℓgrid)
    return nothing
end

function _eval_polygrid(cb::FastChebInterp.ChebPoly, l)
    n = length(cb.coefs)
    grid = zeros(length(l), n)

    for i in 1:n
        base_coefs = zero(cb)
        base_coefs.coefs[i] = 1
        grid[:,i] = base_coefs.(l)
    end
    return grid
end

"""
    get_ℓgrid(CℓEmulator::AbstractCℓEmulators)
Returns the ``\\ell``-grid the emulator has been trained on.
"""
function get_ℓgrid(CℓEmulator::AbstractCℓEmulators)
    return CℓEmulator.ℓgrid
end

"""
    get_emulator_description(CℓEmulator::AbstractCℓEmulators)
Prints on screen the data contained in the emulator.
"""
function get_emulator_description(CℓEmulator::AbstractCℓEmulators)
    get_emulator_description(CℓEmulator.TrainedEmulator)
    return nothing
end

end # module
