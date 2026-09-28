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
    artifact_root = artifact"CAMB_MNUW0WACDM"
    global trained_emulators = Dict(
        "CAMB_MNUW0WACDM" => Dict(
            "TT" => _load_bundled_emulator(artifact_root, "TT"),
            "TE" => _load_bundled_emulator(artifact_root, "TE"),
            "EE" => _load_bundled_emulator(artifact_root, "EE"),
            "BB" => _load_bundled_emulator(artifact_root, "BB"),
            "PP" => _load_bundled_emulator(artifact_root, "PP"),
        ),
    )
end

end # module
