abstract type AbstractInterpolationMethod end

struct IdentityInterpolation{L} <: AbstractInterpolationMethod
    PredictionℓGrid::L
end

(::IdentityInterpolation)(values) = values

"""
    SplinePlan(ℓgrid; plan_type=CubicSplinePlan, endpoint_tolerance=0.1)

Prepare interpolation from a fixed training grid to every integer multipole in
the inclusive range `minimum(ℓgrid):maximum(ℓgrid)`. `CubicSplinePlan` is the
default implementation. Source endpoints within `endpoint_tolerance` of an
integer are snapped to that integer; other endpoints are moved inward.
"""
struct SplinePlan{P, L} <: AbstractInterpolationMethod
    Plan::P
    PredictionℓGrid::L
    SourceAscending::Bool
end

Adapt.@adapt_structure IdentityInterpolation
Adapt.@adapt_structure SplinePlan

function _source_grid_orientation(ℓgrid::AbstractVector)
    length(ℓgrid) >= 2 || throw(ArgumentError("The multipole grid needs at least two points"))

    differences = diff(ℓgrid)
    source_ascending = all(>(zero(eltype(differences))), differences)
    source_descending = all(<(zero(eltype(differences))), differences)
    (source_ascending || source_descending) ||
        throw(ArgumentError("The multipole grid must be strictly monotonic"))
    return source_ascending
end

function _is_dense_integer_grid(ℓgrid::AbstractVector, source_ascending::Bool)
    ordered_grid = source_ascending ? ℓgrid : reverse(ℓgrid)
    ℓ_min, ℓ_max = extrema(ordered_grid)
    (isinteger(ℓ_min) && isinteger(ℓ_max)) || return false
    dense_grid = collect(Int(ℓ_min):Int(ℓ_max))
    return length(ordered_grid) == length(dense_grid) && ordered_grid == dense_grid
end

function resolve_training_ℓgrid(ℓgrid::AbstractVector, output_length::Integer)
    length(ℓgrid) == output_length && return ℓgrid

    # Legacy Capse artifacts stored the complete CAMB grid 0:10050 even though
    # the network output was explicitly defined on ℓ=2:5000. Preserve that
    # documented convention without guessing for any other malformed grid.
    if length(ℓgrid) >= output_length + 2 &&
       first(ℓgrid) == 0 &&
       all(diff(ℓgrid) .== 1)
        @warn "Using the legacy Capse ℓ=2:$(output_length + 1) output grid " *
              "because l.npy does not match the network output length." maxlog=1
        return ℓgrid[3:(output_length + 2)]
    end

    throw(ArgumentError(
        "The multipole grid length ($(length(ℓgrid))) does not match the " *
        "emulator output length ($output_length)",
    ))
end


function _dense_endpoint(value::Real, side::Symbol, tolerance::Real)
    0 <= tolerance < 0.5 ||
        throw(ArgumentError("endpoint_tolerance must satisfy 0 ≤ tolerance < 0.5"))
    nearest_integer = round(Int, value)
    abs(value - nearest_integer) <= tolerance && return nearest_integer
    return side === :left ? ceil(Int, value) : floor(Int, value)
end

function SplinePlan(
    ℓgrid::AbstractVector;
    plan_type=CubicSplinePlan,
    endpoint_tolerance::Real=0.1,
)
    source_ascending = _source_grid_orientation(ℓgrid)

    ℓ_min_raw, ℓ_max_raw = extrema(ℓgrid)
    ℓ_min = _dense_endpoint(ℓ_min_raw, :left, endpoint_tolerance)
    ℓ_max = _dense_endpoint(ℓ_max_raw, :right, endpoint_tolerance)
    ℓ_min <= ℓ_max || throw(ArgumentError("The inferred dense multipole grid is empty"))
    prediction_ℓgrid = collect(ℓ_min:ℓ_max)
    knots = source_ascending ? collect(ℓgrid) : reverse(collect(ℓgrid))
    return SplinePlan(
        plan_type(knots, prediction_ℓgrid),
        prediction_ℓgrid,
        source_ascending,
    )
end

function prepare_interpolation_method(
    ℓgrid::AbstractVector;
    interpolation=:auto,
    max_spline_knots::Integer=2048,
    endpoint_tolerance::Real=0.1,
)
    max_spline_knots >= 2 || throw(ArgumentError("max_spline_knots must be at least two"))
    0 <= endpoint_tolerance < 0.5 ||
        throw(ArgumentError("endpoint_tolerance must satisfy 0 ≤ tolerance < 0.5"))
    source_ascending = _source_grid_orientation(ℓgrid)

    if interpolation === :none
        return IdentityInterpolation(collect(ℓgrid))
    elseif interpolation === :cubic
        return SplinePlan(ℓgrid; endpoint_tolerance)
    elseif interpolation !== :auto
        throw(ArgumentError("interpolation must be :auto, :none, or :cubic"))
    end

    if length(ℓgrid) > max_spline_knots ||
       _is_dense_integer_grid(ℓgrid, source_ascending)
        return IdentityInterpolation(collect(ℓgrid))
    end
    return SplinePlan(ℓgrid; endpoint_tolerance)
end

SplinePlan(Cℓemu::AbstractCℓEmulators; kwargs...) =
    SplinePlan(get_training_ℓgrid(Cℓemu); kwargs...)

function (plan::SplinePlan)(values::AbstractVector)
    ordered_values = plan.SourceAscending ? values : reverse(values)
    return plan.Plan(ordered_values)
end

function (plan::SplinePlan)(values::AbstractMatrix)
    ordered_values = plan.SourceAscending ? values : reverse(values; dims=1)
    return plan.Plan(ordered_values)
end

interp_Cℓ(values::Union{AbstractVector, AbstractMatrix}, plan::SplinePlan) = plan(values)
