module Capse

using Adapt
using AbstractCosmologicalEmulators
import AbstractCosmologicalEmulators: get_emulator_description
import JSON: parsefile
import NPZ: npzread
using Artifacts

export get_Cℓ, get_ℓgrid, get_training_ℓgrid
export SplinePlan, prepare_interpolation_method, interp_Cℓ

include("types.jl")
include("interpolation.jl")
include("predict.jl")
include("postprocessing.jl")
include("utils.jl")

function __init__()
    global trained_emulators = Dict()
    trained_emulators["CAMB_MNUW0WACDM"] = Dict()
    for spectrum in ("TT", "TE", "EE", "BB", "PP")
        trained_emulators["CAMB_MNUW0WACDM"][spectrum] =
            load_emulator(joinpath(artifact"CAMB_MNUW0WACDM", spectrum))
    end
end

end # module
