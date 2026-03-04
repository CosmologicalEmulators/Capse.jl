using AbstractCosmologicalEmulators: get_ℓgrid
using FastChebyshev
import FastChebyshev: ChebyshevPlan

"""
    ChebyshevInterpolPlan{P, T}

Pre-computed plan for interpolating Cℓ spectra from a Chebyshev ℓ-grid onto a user-defined ℓ-grid.
Construct with `prepare_Cℓ_interpolation(emulator, ℓgrid_new)`.
"""
struct ChebyshevInterpolPlan{P, T}
    cheb_plan::ChebyshevPlan{1, P, T}  # FFT plan for decomposition
    T_mat::Matrix{Float64}             # Evaluation matrix
    ℓ_min::Int
    ℓ_max::Int
    K::Int                             # Number of nodes
    ascending::Bool
end

"""
    prepare_Cℓ_interpolation(Cℓemu, ℓgrid_new; tol=1e-6) -> ChebyshevInterpolPlan

Prepare an interpolation plan for the emulator `Cℓemu` onto an arbitrary smooth/linear target
ℓ-grid `ℓgrid_new`.

# Arguments
- `Cℓemu::AbstractCℓEmulators`: The loaded emulator instance.
- `ℓgrid_new::AbstractVector`: The new target ℓ-grid on which to interpolate the spectra.

# Keyword Arguments
- `tol::Real=1e-6`: Tolerance for verifying that the underlying ℓ-grid in `Cℓemu` is truly a Chebyshev grid.

# Example
```julia
plan = prepare_Cℓ_interpolation(emu, 2:3000)
Cℓ_interp = interp_Cℓ(Cℓ_predict, plan)
```
"""
function prepare_Cℓ_interpolation(Cℓemu::AbstractCℓEmulators,
                                   ℓgrid_new::AbstractVector;
                                   tol::Real = 1e-6)
    ℓgrid = get_ℓgrid(Cℓemu)
    n_nodes = length(ℓgrid)
    
    # Check orientation
    ascending = ℓgrid[1] < ℓgrid[end]
    ordered_ℓ = ascending ? reverse(ℓgrid) : ℓgrid
    
    ℓ_min = Int(ordered_ℓ[end])
    ℓ_max = Int(ordered_ℓ[1])
    K = n_nodes

    # Verify Chebyshev property (descending cosine nodes)
    expected_nodes = chebyshev_nodes(K, Float64(ℓ_min), Float64(ℓ_max))
    if maximum(abs.(ordered_ℓ .- expected_nodes)) > tol
        @warn "The emulator's ℓ-grid differs from a standard Chebyshev grid by more than \$tol. Interpolation may be inaccurate."
    end

    cheb_plan = prepare_chebyshev_plan(ℓ_min, ℓ_max, K)
    T_mat     = chebyshev_polynomials(Float64.(ℓgrid_new), ℓ_min, ℓ_max, K)

    return ChebyshevInterpolPlan(cheb_plan, T_mat, ℓ_min, ℓ_max, K, ascending)
end

"""
    interp_Cℓ(Cℓ_vals, plan) -> Vector

Interpolate a single Cℓ spectrum from its Chebyshev ℓ-grid onto the target ℓ-grid
baked into `plan`.
"""
function interp_Cℓ(Cℓ_vals::AbstractVector, plan::ChebyshevInterpolPlan)
    desc_vals = plan.ascending ? reverse(Cℓ_vals) : Cℓ_vals
    coeffs = plan.cheb_plan * desc_vals
    return plan.T_mat * coeffs
end

"""
    interp_Cℓ(Cℓ_mat, plan) -> Matrix

Interpolate multiple Cℓ spectra (where each column is a spectrum) from the
emulator's Chebyshev ℓ-grid onto the target ℓ-grid baked into `plan`.
"""
function interp_Cℓ(Cℓ_mat::AbstractMatrix, plan::ChebyshevInterpolPlan)
    # Reverse rows if the emulator's ℓ-grid was ascending
    mat_desc  = plan.ascending ? reverse(Cℓ_mat; dims=1) : Cℓ_mat
    
    # cheb_plan * mat_desc computes coefficients column-by-column
    # Returns a matrix `coeffs` of size (K, n_spectra).
    coeffs = plan.cheb_plan * mat_desc
    
    # Multiply the evaluation matrix `T_mat` (n_new, K) by `coeffs` (K, n_spectra)
    # to yield a matrix of size (n_new, n_spectra).
    return plan.T_mat * coeffs
end
