"""
    Lensing

Gravitational lensing corrections for CMB power spectra.
Functions live directly in the `Capse` module (no submodule).

Implements Challinor & Lewis (2005), astro-ph/0601594, §3,
using the angular correlation function approach.
"""

using FastGaussQuadrature: gausslegendre
using LinearAlgebra: mul!, lmul!
using LoopVectorization: @turbo, @tturbo
using Base.Threads: @threads, nthreads, threadid

# ---------------------------------------------------------------------------
# 1. Gauss-Legendre quadrature nodes
# ---------------------------------------------------------------------------

"""
    gauss_legendre_weights(n) -> (mu, w)

Return the `n` Gauss-Legendre quadrature nodes `mu` ∈ (-1,1) and weights `w`,
ordered from -1 to +1.
"""
function gauss_legendre_weights(n::Int)
    mu, w = gausslegendre(n)
    return mu, w
end

# ---------------------------------------------------------------------------
# 2. Wigner reduced d-matrix  d^ℓ_{mn}(arccos μ)
# ---------------------------------------------------------------------------

"""
    wigner_d_matrix(mu, ells, m, n) -> Matrix{Float64}  (Nmu × Nell)

Compute reduced Wigner d-matrix elements d^ℓ_{mn}(β) where β = arccos(μ),
for all `mu` values and all `ells` values simultaneously.

Uses the three-term recurrence relation matching the Python ABCMB implementation.
`ells` must start at `ells[1] == m` and increase by 1.
`m` must be non-negative and `|n| ≤ m`.
"""
function wigner_d_matrix(mu::AbstractVector{<:Real},
    ells::AbstractVector{<:Integer},
    m::Int, n::Int)
    @assert m >= 0 && abs(n) <= m

    Nmu = length(mu)
    Nell = length(ells)
    out = Matrix{Float64}(undef, Nmu, Nell)

    _lfac(k) = k <= 1 ? 0.0 : log(Float64(factorial(k)))
    log_norm = 0.5 * log(2m + 1) - 0.5 * log(2) +
               0.5 * (_lfac(2m) - _lfac(m + n) - _lfac(m - n))
    norm = exp(log_norm)

    d_base = Vector{Float64}(undef, Nmu)
    for i in 1:Nmu
        β = acos(clamp(mu[i], -1.0, 1.0))
        d_base[i] = norm * cos(β / 2)^(m + n) * (-sin(β / 2))^(m - n)
    end

    Ac = Vector{Float64}(undef, Nell)
    Bc = Vector{Float64}(undef, Nell)
    Cc = Vector{Float64}(undef, Nell)

    for k in 1:Nell
        ℓ = ells[k]
        denom2 = ((ℓ + 1)^2 - m^2) * ((ℓ + 1)^2 - n^2)
        if denom2 <= 0
            Ac[k] = Bc[k] = Cc[k] = 0.0
        else
            denom = sqrt(denom2)
            normA = sqrt((2ℓ + 3) / (2ℓ + 1))
            normC = (ℓ > 0) ? sqrt((2ℓ + 3) / (2ℓ - 1)) : 0.0
            a = normA * (ℓ + 1) * (2ℓ + 1) / denom
            b = (ℓ > 0 && ℓ * (ℓ + 1) != 0) ? -a * m * n / (ℓ * (ℓ + 1)) : 0.0
            c2 = (ℓ^2 - m^2) * (ℓ^2 - n^2)
            c = (c2 > 0 && ℓ > 0) ? -normC * sqrt(c2) / denom * (ℓ + 1) / ℓ : 0.0
            Ac[k] = a
            Bc[k] = b
            Cc[k] = c
        end
    end

    for i in 1:Nmu
        d_prev = 0.0
        d_curr = d_base[i]
        for k in 1:Nell
            ℓ = ells[k]
            out[i, k] = d_curr * sqrt(2.0 / (2ℓ + 1))
            d_next = Ac[k] * mu[i] * d_curr + Bc[k] * d_curr + Cc[k] * d_prev
            d_prev = d_curr
            d_curr = d_next
        end
    end

    return out
end

# ---------------------------------------------------------------------------
# 3. Named Wigner helpers matching Python ABCMB API
# ---------------------------------------------------------------------------

"""d00(mu, ells) — d^ℓ_{00}(arccos μ) = Legendre polynomial P_ℓ(μ)."""
function d00(mu::AbstractVector, ells::AbstractVector{<:Integer})
    res = wigner_d_matrix(mu, vcat([0, 1], ells), 0, 0)
    return res[:, 3:end]
end

"""d1n(mu, ells, n) — d^ℓ_{1n}(arccos μ)."""
function d1n(mu::AbstractVector, ells::AbstractVector{<:Integer}, n::Int)
    res = wigner_d_matrix(mu, vcat([1], ells), 1, n)
    return res[:, 2:end]
end

"""d2n(mu, ells, n) — d^ℓ_{2n}(arccos μ)."""
function d2n(mu::AbstractVector, ells::AbstractVector{<:Integer}, n::Int)
    return wigner_d_matrix(mu, ells, 2, n)
end

"""d3n(mu, ells, n) — d^ℓ_{3n}(arccos μ), zero-padded for ℓ < 3."""
function d3n(mu::AbstractVector, ells::AbstractVector{<:Integer}, n::Int)
    res = wigner_d_matrix(mu, ells[2:end], 3, n)
    return hcat(zeros(Float64, length(mu), 1), res)
end

"""d4n(mu, ells, n) — d^ℓ_{4n}(arccos μ), zero-padded for ℓ < 4."""
function d4n(mu::AbstractVector, ells::AbstractVector{<:Integer}, n::Int)
    res = wigner_d_matrix(mu, ells[3:end], 4, n)
    return hcat(zeros(Float64, length(mu), 2), res)
end

# ---------------------------------------------------------------------------
# 4. LensingConfig – precomputed workspace
# ---------------------------------------------------------------------------

"""
    LensingConfig

Precomputed workspace for the lensing pipeline.
Build once (slow), reuse for every cosmology (fast).

# Fields
- `ells`, `lensing_ells` : multipole ranges
- `mu`, `w`              : GL nodes/weights
- `mu_ext`, `w_ext`      : GL nodes/weights extended with μ=+1, w=0
- 12 precomputed Wigner d-matrices (Nmu_ext × Nell)
"""
struct LensingConfig
    ells::Vector{Int}
    lensing_ells::Vector{Int}
    mu::Vector{Float64}
    w::Vector{Float64}
    mu_ext::Vector{Float64}
    w_ext::Vector{Float64}
    d00_mat::Matrix{Float64}
    d11_mat::Matrix{Float64}
    d1m1_mat::Matrix{Float64}
    d22_mat::Matrix{Float64}
    d2m2_mat::Matrix{Float64}
    d20_mat::Matrix{Float64}
    d31_mat::Matrix{Float64}
    d3m1_mat::Matrix{Float64}
    d3m3_mat::Matrix{Float64}
    d40_mat::Matrix{Float64}
    d4m2_mat::Matrix{Float64}
    d4m4_mat::Matrix{Float64}
end

"""
    LensingConfig(ellmin, ellmax)

Precompute GL nodes/weights and all 12 Wigner d-matrices for the lensing pipeline.
This is the expensive one-time setup step.
"""
function LensingConfig(ellmin::Int, ellmax::Int)
    ells = collect(ellmin:ellmax)
    lensing_ells = collect(ellmin:(ellmax+500))
    n_gl = (ellmax + 500) + 70
    mu, w = gauss_legendre_weights(n_gl)
    mu_ext = vcat(mu, [1.0])
    w_ext = vcat(w, [0.0])

    return LensingConfig(
        ells, lensing_ells, mu, w, mu_ext, w_ext,
        d00(mu_ext, lensing_ells),
        d1n(mu_ext, lensing_ells, 1),
        d1n(mu_ext, lensing_ells, -1),
        d2n(mu_ext, lensing_ells, 2),
        d2n(mu_ext, lensing_ells, -2),
        d2n(mu_ext, lensing_ells, 0),
        d3n(mu_ext, lensing_ells, 1),
        d3n(mu_ext, lensing_ells, -1),
        d3n(mu_ext, lensing_ells, -3),
        d4n(mu_ext, lensing_ells, 0),
        d4n(mu_ext, lensing_ells, -2),
        d4n(mu_ext, lensing_ells, -4),
    )
end

# ---------------------------------------------------------------------------
# 5. lensed_Cls – optimised lensing pipeline
# ---------------------------------------------------------------------------

"""
    lensed_Cls(ClTT_unl, ClTE_unl, ClEE_unl, Clpp, cfg) -> (ClTT, ClTE, ClEE)

Apply gravitational lensing corrections to unlensed CMB power spectra.

Implements Challinor & Lewis (2005), astro-ph/0601594, §3.

**Performance** — the function iterates over multipoles (outer loop) and GL
nodes (inner loop), keeping all working arrays at size O(N_GL) rather than
O(N_GL × N_ell). This avoids allocating several GB of temporaries at lmax=3000
and reduces per-call time from ~7 s to ~100 ms.

All input spectra must be indexed over `cfg.lensing_ells`.
`Clpp` is the lensing potential C^{φφ}_ℓ (passed directly — Capse never runs
a Boltzmann solver).

Returns `(ClTT_lensed, ClTE_lensed, ClEE_lensed)` indexed over
`cfg.lensing_ells`.
"""
function lensed_Cls(ClTT_unl::AbstractVector,
    ClTE_unl::AbstractVector,
    ClEE_unl::AbstractVector,
    Clpp::AbstractVector,
    cfg::LensingConfig)

    ells = cfg.lensing_ells
    Nell = length(ells)
    Nmu = length(cfg.mu_ext)

    D00 = cfg.d00_mat
    D11 = cfg.d11_mat
    D1m1 = cfg.d1m1_mat
    D22 = cfg.d22_mat
    D2m2 = cfg.d2m2_mat
    D20 = cfg.d20_mat
    D31 = cfg.d31_mat
    D3m1 = cfg.d3m1_mat
    D3m3 = cfg.d3m3_mat
    D40 = cfg.d40_mat
    D4m2 = cfg.d4m2_mat
    D4m4 = cfg.d4m4_mat

    # ── Precompute ell-dependent scalars ─────────────────────────────────────
    llp1 = Vector{Float64}(undef, Nell)
    two_l1 = Vector{Float64}(undef, Nell)
    s220 = Vector{Float64}(undef, Nell)
    s121 = Vector{Float64}(undef, Nell)
    s132 = Vector{Float64}(undef, Nell)
    s242 = Vector{Float64}(undef, Nell)
    inv_slp1 = Vector{Float64}(undef, Nell)
    f0_vals = Vector{Float64}(undef, Nell)
    f4_vals = Vector{Float64}(undef, Nell)
    i8lp1_vals = Vector{Float64}(undef, Nell)

    @inbounds for j in 1:Nell
        ℓ = Float64(ells[j])
        lp1 = ℓ * (ℓ + 1)
        llp1[j] = lp1
        two_l1[j] = 2ℓ + 1
        s220[j] = sqrt(max((ℓ + 2) * (ℓ - 1) * ℓ * (ℓ + 1), 0.0)) / 4
        s121[j] = -sqrt(max((ℓ + 2) * (ℓ - 1), 0.0)) / 2
        s132[j] = -sqrt(max((ℓ + 3) * (ℓ - 2), 0.0)) / 2
        s242[j] = sqrt(max((ℓ + 4) * (ℓ + 3) * (ℓ - 2) * (ℓ - 3), 0.0)) / 4
        inv_slp1[j] = lp1 > 0 ? 1 / sqrt(lp1) : 0.0
        f0_vals[j] = lp1 / 4.0
        f4_vals[j] = (lp1 - 4.0) / 4.0
        i8lp1_vals[j] = lp1 > 0 ? 8.0 / lp1 : 0.0
    end

    # ── Step 1: Cgl, Cgl2, sigma2 via BLAS gemv ──────────────────────────────
    coeff = @. (two_l1 * llp1) * Clpp   # (Nell,)
    Cgl = Vector{Float64}(undef, Nmu)
    Cgl2 = Vector{Float64}(undef, Nmu)
    mul!(Cgl, D11, coeff)
    mul!(Cgl2, D1m1, coeff)
    inv4pi = 1 / (4π)
    lmul!(inv4pi, Cgl)
    lmul!(inv4pi, Cgl2)
    sigma2 = Cgl[end] .- Cgl            # (Nmu,)

    # ── Step 1.5: Precompute exp factors for sigma2 ─────────────────────────
    # We need things like exp(-(lp1-4)/4 * s2) = exp(-lp1/4 * s2) * exp(s2)
    # So we compute exp(s2) and other fractions before the big loop:
    es2_05 = zeros(Float64, Nmu) # exp(s2/2)
    es2_1 = zeros(Float64, Nmu) # exp(s2)
    es2_23 = zeros(Float64, Nmu) # exp(2s2/3)
    es2_53 = zeros(Float64, Nmu) # exp(5s2/3)
    es2_25 = zeros(Float64, Nmu) # exp(2.5 * s2)

    @inbounds for i in 1:Nmu
        s2 = sigma2[i]
        es2_05[i] = exp(0.5 * s2)
        es2_1[i] = exp(s2)
        es2_23[i] = exp((2.0 / 3.0) * s2)
        es2_53[i] = exp((5.0 / 3.0) * s2)
        es2_25[i] = exp(2.5 * s2)
    end

    # ── Step 2: Accumulate ξ, ξ₊, ξ₋, ξ× — threaded outer loop ─────────────
    # Each thread accumulates into its own private (Nmu,) buffers, then we
    # reduce across threads at the end.  No race conditions; @turbo SIMD
    # inside each thread's j-slice is preserved.
    nt = nthreads()
    ksi_t = [zeros(Float64, Nmu) for _ in 1:nt]
    ksip_t = [zeros(Float64, Nmu) for _ in 1:nt]
    ksim_t = [zeros(Float64, Nmu) for _ in 1:nt]
    ksix_t = [zeros(Float64, Nmu) for _ in 1:nt]

    @threads :static for j in 1:Nell
        tid = threadid()
        _ksi = ksi_t[tid]
        _ksip = ksip_t[tid]
        _ksim = ksim_t[tid]
        _ksix = ksix_t[tid]

        tl1 = two_l1[j]
        ClT = Float64(ClTT_unl[j])
        ClE = Float64(ClEE_unl[j])
        ClX = Float64(ClTE_unl[j])
        a220 = s220[j]
        a121 = s121[j]
        a132 = s132[j]
        a242 = s242[j]
        islp1 = inv_slp1[j]
        f0 = f0_vals[j]
        f4 = f4_vals[j]
        i8lp1 = i8lp1_vals[j]

        @turbo for i in 1:Nmu
            c2 = Cgl2[i]
            s2 = sigma2[i]
            x0 = exp(-f0 * s2)
            x0p = -f0 * x0
            x022 = x0 * es2_1[i]
            x22p = -f4 * x022
            x220 = a220 * x0 * es2_05[i]
            x121 = a121 * x0 * es2_23[i]
            x132 = a132 * x0 * es2_53[i]
            x242 = a242 * x0 * es2_25[i]

            x0_2 = x0 * x0
            x0p_2 = x0p * x0p
            x220_2 = x220 * x220
            x022_2 = x022 * x022
            x22p_2 = x22p * x22p
            x242_2 = x242 * x242
            c2_2 = c2 * c2

            _ksi[i] += tl1 * ClT * (
                           x0_2 * D00[i, j] + i8lp1 * c2 * x0p_2 * D1m1[i, j] +
                           c2_2 * (x0p_2 * D00[i, j] + x220_2 * D2m2[i, j]))

            _ksip[i] += tl1 * ClE * (
                            x022_2 * D22[i, j] + 2c2 * x132 * x121 * D31[i, j] +
                            c2_2 * (x22p_2 * D22[i, j] + x242 * x220 * D40[i, j]))

            _ksim[i] += tl1 * ClE * (
                            x022_2 * D2m2[i, j] + c2 * (x121 * x121 * D1m1[i, j] + x132 * x132 * D3m3[i, j]) +
                            0.5 * c2_2 * (2x22p_2 * D2m2[i, j] + x220_2 * D00[i, j] + x242_2 * D4m4[i, j]))

            _ksix[i] += tl1 * ClX * (
                            x022 * x0 * D20[i, j] + 2c2 * x0p * islp1 * (x121 * D11[i, j] + x132 * D3m1[i, j]) +
                            0.5 * c2_2 * ((2x22p * x0p + x220_2) * D20[i, j] + x220 * x242 * D4m2[i, j]))
        end
    end

    # Reduce thread-local buffers
    ksi = ksi_t[1]
    ksip = ksip_t[1]
    ksim = ksim_t[1]
    ksix = ksix_t[1]
    for t in 2:nt
        @. ksi += ksi_t[t]
        @. ksip += ksip_t[t]
        @. ksim += ksim_t[t]
        @. ksix += ksix_t[t]
    end

    lmul!(inv4pi, ksi)
    lmul!(inv4pi, ksip)
    lmul!(inv4pi, ksim)
    lmul!(inv4pi, ksix)

    # ── Step 3: Back-transform via BLAS gemv ──────────────────────────────────
    # Weighted by w: ksi → ksi .* w  (in-place to avoid extra allocation)
    w = cfg.w_ext
    @. ksi = ksi * w
    @. ksix = ksix * w
    @. ksip = ksip * w
    @. ksim = ksim * w

    # ClTT = 2π ∑_μ w_μ ξ(μ) d00(μ,ℓ)  ↔  2π * D00ᵀ * (ksi.*w)
    ClTT_lensed = Vector{Float64}(undef, Nell)
    ClTE_lensed = Vector{Float64}(undef, Nell)
    EE_buf1 = Vector{Float64}(undef, Nell)
    EE_buf2 = Vector{Float64}(undef, Nell)

    mul!(ClTT_lensed, D00', ksi)
    mul!(ClTE_lensed, D20', ksix)
    mul!(EE_buf1, D22', ksip)
    mul!(EE_buf2, D2m2', ksim)

    lmul!(2π, ClTT_lensed)
    lmul!(2π, ClTE_lensed)
    ClEE_lensed = π .* (EE_buf1 .+ EE_buf2)

    return ClTT_lensed, ClTE_lensed, ClEE_lensed
end
