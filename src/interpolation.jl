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

"""
    ChebyshevInterpolPlan{P, T}

Pre-computed plan for interpolating Cℓ spectra from a Chebyshev ℓ-grid onto a user-defined ℓ-grid.
Construct with `prepare_Cℓ_interpolation(emulator, ℓgrid_new)`.
"""
struct ChebyshevInterpolPlan{P, T}
    cheb_plan::ChebyshevPlan{1, P, T}  # FFT plan for decomposition
    T_mat::Matrix{T}                    # Chebyshev basis at ℓgrid_new, shape (n_new, K+1)
    ℓ_min::T
    ℓ_max::T
    K::Int
    ascending::Bool                     # true if the emulator ℓgrid was stored ascending
end

"""
    prepare_Cℓ_interpolation(Cℓemu, ℓgrid_new; tol=1e-6) -> ChebyshevInterpolPlan

Prepare a reusable interpolation plan for `interp_Cℓ`.

# Arguments
- `Cℓemu::AbstractCℓEmulators`: The emulator whose `ℓgrid` defines the Chebyshev nodes.
- `ℓgrid_new::AbstractVector`: Target ℓ-values for evaluation.
- `tol::Real=1e-6`: Tolerance for Chebyshev grid validation.

# Grid-orientation logic
1. Retrieve `ℓgrid` from the emulator.
2. If ascending, set `ascending=true` and reverse internally (Chebyshev nodes are descending).
3. Compute `ℓ_min`, `ℓ_max`, `K = length(ℓgrid) - 1`.
4. Generate the *expected* Chebyshev grid via `chebpoints(K, ℓ_min, ℓ_max)` and compare
   element-wise to the (possibly reversed) emulator grid.
   - If `norm(expected - observed) / norm(expected) > tol`, emit a `@warn`
     informing the user that the ℓ-grid may not be a Chebyshev grid and accuracy
     could be degraded.
5. Prepare the FFT plan and precompute `T_mat = chebyshev_polynomials(ℓgrid_new, ℓ_min, ℓ_max, K)`.
"""
function prepare_Cℓ_interpolation(Cℓemu::AbstractCℓEmulators,
                                   ℓgrid_new::AbstractVector;
                                   tol::Real = 1e-6)
    ℓgrid = get_training_ℓgrid(Cℓemu)
    ascending = issorted(ℓgrid)          # ascending  ↔ needs reversal for FFTW
    ℓgrid_desc = ascending ? reverse(ℓgrid) : ℓgrid

    ℓ_min = Float64(last(ℓgrid_desc))   # smallest ℓ (tail of descending vector)
    ℓ_max = Float64(first(ℓgrid_desc))  # largest  ℓ (head of descending vector)
    K = length(ℓgrid) - 1

    # Validate against theoretical Chebyshev nodes
    expected = chebpoints(K, ℓ_min, ℓ_max)  # descending, K+1 points
    rel_err = maximum(abs.(expected .- ℓgrid_desc)) / maximum(abs.(expected))

    if rel_err > tol
        @warn "The emulator ℓ-grid does not appear to be a Chebyshev grid " *
              "(relative max deviation = $(round(rel_err; sigdigits=3))). " *
              "Interpolation accuracy may be degraded."
    end

    cheb_plan = prepare_chebyshev_plan(ℓ_min, ℓ_max, K)
    T_mat     = chebyshev_polynomials(Float64.(ℓgrid_new), ℓ_min, ℓ_max, K)

    return ChebyshevInterpolPlan(cheb_plan, T_mat, ℓ_min, ℓ_max, K, ascending)
end

"""
    interp_Cℓ(Cℓ_vals, plan) -> Vector

Interpolate a single Cℓ spectrum from its Chebyshev ℓ-grid onto the target ℓ-grid
baked into `plan`.

# Arguments
- `Cℓ_vals::AbstractVector`: Spectrum values on the emulator's Chebyshev ℓ-grid (length K+1).
  Must be in the *same orientation* as the emulator's stored ℓ-grid (ascending or descending).
- `plan::ChebyshevInterpolPlan`: Prepared interpolation plan.

# Returns
- `Vector`: Spectrum evaluated at the target ℓ-grid.
"""
function interp_Cℓ(Cℓ_vals::AbstractVector, plan::ChebyshevInterpolPlan)
    # If emulator stored ascending, reverse so that values match descending Chebyshev nodes
    vals_desc = plan.ascending ? reverse(Cℓ_vals) : Cℓ_vals
    coeffs    = chebyshev_decomposition(plan.cheb_plan, vals_desc)
    return plan.T_mat * coeffs
end

"""
    interp_Cℓ(Cℓ_mat, plan) -> Matrix

Interpolate multiple Cℓ spectra (columns of a matrix) onto the target ℓ-grid
baked into `plan`.

# Arguments
- `Cℓ_mat::AbstractMatrix`: Shape `(n_ℓ, n_spectra)`. Each column is a spectrum on the
  emulator's Chebyshev ℓ-grid, in the same orientation as the stored ℓ-grid.
- `plan::ChebyshevInterpolPlan`

# Returns
- `Matrix`: Shape `(length(ℓgrid_new), n_spectra)`.
"""
function interp_Cℓ(Cℓ_mat::AbstractMatrix, plan::ChebyshevInterpolPlan)
    # Reverse rows if the emulator's ℓ-grid was ascending
    mat_desc  = plan.ascending ? reverse(Cℓ_mat; dims=1) : Cℓ_mat
    # chebyshev_decomposition already handles batched (matrix) input column-wise
    coeffs    = chebyshev_decomposition(plan.cheb_plan, mat_desc)  # (K+1, n_spectra)
    return plan.T_mat * coeffs                                       # (n_new, n_spectra)
end
