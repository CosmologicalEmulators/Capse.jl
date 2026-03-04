module Capse

using Base: @kwdef
using Adapt
using AbstractCosmologicalEmulators
import AbstractCosmologicalEmulators: get_emulator_description
import JSON: parsefile
import NPZ: npzread
using Artifacts

export get_Cℓ
export ChebyshevInterpolPlan, prepare_Cℓ_interpolation, interp_Cℓ

include("types.jl")
include("interpolation.jl")
include("predict.jl")
include("utils.jl")

function __init__()
    global trained_emulators = Dict()
    trained_emulators["CAMB_LCDM"] = Dict()
    trained_emulators["CAMB_LCDM"]["TT"] = load_emulator(joinpath(artifact"CAMB_LCDM", "TT/"))
    trained_emulators["CAMB_LCDM"]["TE"] = load_emulator(joinpath(artifact"CAMB_LCDM", "TE/"))
    trained_emulators["CAMB_LCDM"]["EE"] = load_emulator(joinpath(artifact"CAMB_LCDM", "EE/"))
    trained_emulators["CAMB_LCDM"]["PP"] = load_emulator(joinpath(artifact"CAMB_LCDM", "PP/"))
end

end # module