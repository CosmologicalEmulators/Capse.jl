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
    ℓgrid = get_ℓgrid(Cℓemu)
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
