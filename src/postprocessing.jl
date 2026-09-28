"""Amplitude rescaling for linear or log NN targets, with optional optical-depth damping."""
struct AsPostprocessing{LogTarget, TauDamped}
    ln10As_index::Int
    tau_index::Int
end

function (post::AsPostprocessing{LogTarget, TauDamped})(input, output, emu) where {LogTarget, TauDamped}
    factor = if ndims(input) == 1
        exp(input[post.ln10As_index]) * 1.0e-10
    else
        exp.(input[post.ln10As_index:post.ln10As_index, :]) .* 1.0e-10
    end
    if TauDamped
        damping = if ndims(input) == 1
            exp(-2 * input[post.tau_index])
        else
            exp.(-2 .* input[post.tau_index:post.tau_index, :])
        end
        factor = factor .* damping
    end
    values = LogTarget ? exp.(output) : output
    return values .* factor
end

postprocessing_as_tau_linear(as_index::Int, tau_index::Int) =
    AsPostprocessing{false, true}(as_index, tau_index)
postprocessing_as_tau_log(as_index::Int, tau_index::Int) =
    AsPostprocessing{true, true}(as_index, tau_index)
postprocessing_as_log(as_index::Int, tau_index::Int) =
    AsPostprocessing{true, false}(as_index, tau_index)

const BUILTIN_POSTPROCESSING = Dict{String, Function}(
    "as_tau_linear" => postprocessing_as_tau_linear,
    "as_tau_log" => postprocessing_as_tau_log,
    "as_log" => postprocessing_as_log,
)
