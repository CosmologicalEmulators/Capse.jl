module Capse

using Base: @kwdef
using SimpleChains
using BSON

function maximin_input!(x, in_MinMax)
    for i in 1:length(x)
        x[i] -= in_MinMax[i,1]
        x[i] /= (in_MinMax[i,2]-in_MinMax[i,1])
    end
end

function inv_maximin_output!(x, out_MinMax)
    for i in 1:length(x)
        x[i] *= (out_MinMax[i,2]-out_MinMax[i,1])
        x[i] += out_MinMax[i,1]
    end
end

abstract type AbstractTrainedEmulators end

@kwdef mutable struct SimpleChainsEmulator <: AbstractTrainedEmulators
    Architecture
    Weights
end

abstract type AbstractCℓEmulators end

@kwdef mutable struct CℓEmulators <: AbstractCℓEmulators
    TrainedEmulator::AbstractTrainedEmulators
    InMinMax::Matrix{Float64} = zeros(8,2)
    OutMinMax::Array{Float64} = zeros(2499,2)
end

function compute_Cℓ(input_params, CℓEmulator::AbstractCℓEmulators)
    input = deepcopy(input_params)
    maximin_input!(input, CℓEmulator.InMinMax)
    output = Array(run_emulator(input, CℓEmulator.TrainedEmulator))
    inv_maximin_output!(output, CℓEmulator.OutMinMax)
    return output .* exp(input_params[1]-3.)
end

function run_emulator(input, trained_emulator::SimpleChainsEmulator)
    return trained_emulator.Architecture(input, trained_emulator.Weights)
end

function load_emulators(path)
    loaded = BSON.load(path)

    Cltt = loaded[:CℓTT]
    Clee = loaded[:CℓEE]
    Clte = loaded[:CℓTE]
    Clpp = loaded[:CℓPP]

    return Cltt, Clee, Clte, Clpp

end

end # module
