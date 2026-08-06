abstract type AbstractCℓEmulators end

"""
    CℓEmulator(; TrainedEmulator, ℓgrid, InMinMax, OutMinMax, Postprocessing,
               interpolation=:auto, max_spline_knots=2048, endpoint_tolerance=0.1)

Main struct for CMB angular power spectrum emulation.

# Fields
- `TrainedEmulator::AbstractTrainedEmulators`: Trained neural network model
- `ℓgrid::AbstractVector`: Constructor keyword containing the training multipoles
- `InMinMax::AbstractMatrix`: Min-max normalization parameters for inputs (2×n_params)
- `OutMinMax::AbstractMatrix`: Min-max normalization parameters for outputs (2×n_ℓ)
- `Postprocessing::Function`: Post-processing function with signature `f(input, output, emulator)`
- `TrainingℓGrid`: Multipoles used while training the emulator
- `PredictionℓGrid`: Multipoles returned by `get_Cℓ`
- `InterpolationMethod`: Identity or cubic interpolation applied after postprocessing

# Example
```julia
# Typically created via load_emulator, but can be constructed manually:
emulator = CℓEmulator(
    TrainedEmulator = trained_nn,
    ℓgrid = collect(2:2500),
    InMinMax = [mins max_vals],  # 2×n_params matrix
    OutMinMax = [mins max_vals],  # 2×n_ℓ matrix
    Postprocessing = (input, output, emu) -> output
)
```

See also: [`load_emulator`](@ref), [`get_Cℓ`](@ref)
"""
struct CℓEmulator{TE, LT, LP, I, O, P, S} <: AbstractCℓEmulators
    TrainedEmulator::TE
    TrainingℓGrid::LT
    PredictionℓGrid::LP
    InMinMax::I
    OutMinMax::O
    Postprocessing::P
    InterpolationMethod::S
end

function CℓEmulator(;
    TrainedEmulator,
    ℓgrid,
    InMinMax,
    OutMinMax,
    Postprocessing,
    interpolation=:auto,
    max_spline_knots::Integer=2048,
    endpoint_tolerance::Real=0.1,
    InterpolationMethod=nothing,
)
    training_ℓgrid = resolve_training_ℓgrid(ℓgrid, size(OutMinMax, 1))
    interpolation_method = isnothing(InterpolationMethod) ?
        prepare_interpolation_method(
            training_ℓgrid;
            interpolation,
            max_spline_knots,
            endpoint_tolerance,
        ) : InterpolationMethod
    return CℓEmulator(
        TrainedEmulator,
        training_ℓgrid,
        interpolation_method.PredictionℓGrid,
        InMinMax,
        OutMinMax,
        Postprocessing,
        interpolation_method,
    )
end

Adapt.@adapt_structure CℓEmulator
