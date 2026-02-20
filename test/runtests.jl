using Test
using NPZ
using SimpleChains
using Static
using Capse

# ── Emulator test infrastructure ──────────────────────────────────────────────
mlpd = SimpleChain(
  static(6),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(identity, 40)
)

ℓ_test = Array(LinRange(0, 200, 40))
weights = SimpleChains.init_params(mlpd)
inminmax = rand(6, 2)
outminmax = rand(40, 2)

mkpath("emu")
npzwrite("emu/l.npy", ℓ_test)
npzwrite("emu/weights.npy", weights)
npzwrite("emu/inminmax.npy", inminmax)
npzwrite("emu/outminmax.npy", outminmax)
open("emu/nn_setup.json", "w") do io
  write(io, """{"n_input_features": 6, "n_output_features": 40, "n_hidden_layers": 1, "emulator_description": {}, "layers": {"layer_1": {"activation_function": "tanh", "n_neurons": 64}}}""")
end
open("emu/postprocessing.jl", "w") do io
  write(io, "postprocessing(input, output, Cℓemu) = output .* exp(input[1]-3.)")
end

emu = Capse.SimpleChainsEmulator(Architecture=mlpd, Weights=weights)
postprocessing(input, output, Cℓemu) = output .* exp(input[1] - 3.)
capse_emu = Capse.CℓEmulator(TrainedEmulator=emu, ℓgrid=ℓ_test,
  InMinMax=inminmax, OutMinMax=outminmax,
  Postprocessing=postprocessing)
capse_loaded_emu = Capse.load_emulator("emu/")

@testset "Capse emulator tests" begin
  cosmo = ones(6)
  cosmo_vec = ones(6, 6)
  output = Capse.get_Cℓ(cosmo, capse_emu)
  output_vec = Capse.get_Cℓ(cosmo_vec, capse_emu)
  @test isapprox(output_vec[:, 1], output)
  @test ℓ_test == Capse.get_ℓgrid(capse_emu)
  @test_logs (:warn, "No emulator description found!") Capse.get_emulator_description(capse_emu)
  @test_skip Capse.get_Cℓ(cosmo_vec, capse_emu) == Capse.get_Cℓ(cosmo_vec, capse_loaded_emu)
end

# ── Load Python ABCMB reference data ─────────────────────────────────────────
# Regenerate with: python test/generate_python_reference.py
const REF_FILE = joinpath(@__DIR__, "python_lensing_reference.npz")
isfile(REF_FILE) || error("Run `python test/generate_python_reference.py` first.")
const REF = npzread(REF_FILE)

# Helper: element-wise comparison, warns with stats on failure
function check_vs_python(label, D_julia, D_python; atol=3e-13)
  diff = D_julia .- Float64.(D_python)
  abs_max = maximum(abs.(diff))
  abs_max < atol || @warn "$label: max |Julia - Python| = $abs_max (threshold=$atol)"
  return abs_max
end

@testset "Lensing tests (vs Python ABCMB)" begin
  using LinearAlgebra: dot

  py_ells = Int.(REF["ells"])    # 2:3000
  py_mu = Float64.(REF["mu"]) # 200 points
  lmax = maximum(py_ells)    # 3000

  # ── 1. Gauss-Legendre weights vs Python ───────────────────────────────────
  @testset "gauss_legendre_weights vs Python (n for LensingConfig(2,$lmax))" begin
    n = Int(REF["gl_n"][1])
    j_mu, j_w = gauss_legendre_weights(n)
    @test sum(j_w) ≈ 2.0 atol = 1e-12
    @test dot(j_w, j_mu .^ 2) ≈ 2 / 3 atol = 1e-12
    @test dot(j_w, j_mu .^ 4) ≈ 2 / 5 atol = 1e-12
    @test maximum(abs.(j_mu .- Float64.(REF["gl_mu"]))) < 2e-13
    @test maximum(abs.(j_w .- Float64.(REF["gl_w"]))) < 2e-13
  end

  # ── 2. All 12 Wigner matrices vs Python at ells=2:3000 ────────────────────
  wigner_cases = [
    ("d00", () -> d00(py_mu, py_ells), "d00"),
    ("d11", () -> d1n(py_mu, py_ells, 1), "d11"),
    ("d1m1", () -> d1n(py_mu, py_ells, -1), "d1m1"),
    ("d22", () -> d2n(py_mu, py_ells, 2), "d22"),
    ("d2m2", () -> d2n(py_mu, py_ells, -2), "d2m2"),
    ("d20", () -> d2n(py_mu, py_ells, 0), "d20"),
    ("d31", () -> d3n(py_mu, py_ells, 1), "d31"),
    ("d3m1", () -> d3n(py_mu, py_ells, -1), "d3m1"),
    ("d3m3", () -> d3n(py_mu, py_ells, -3), "d3m3"),
    ("d40", () -> d4n(py_mu, py_ells, 0), "d40"),
    ("d4m2", () -> d4n(py_mu, py_ells, -2), "d4m2"),
    ("d4m4", () -> d4n(py_mu, py_ells, -4), "d4m4"),
  ]

  @testset "wigner_d_matrix: $name vs Python (ells=2:$lmax)" for (name, jlfn, npz_key) in wigner_cases
    @test check_vs_python(name, jlfn(), REF[npz_key]) < 3e-13
  end

  # ── 3. d00 = Legendre polynomial P_ℓ(μ) ─────────────────────────────────
  @testset "d00 = Legendre polynomial P_ℓ (ells=2:$lmax)" begin
    function legendre_pl(x, lmax_p)
      P = zeros(lmax_p + 1)
      P[1] = 1.0
      lmax_p >= 1 && (P[2] = x)
      for l in 2:lmax_p
        P[l+1] = ((2l - 1) * x * P[l] - (l - 1) * P[l-1]) / l
      end
      P
    end
    D = d00(py_mu, py_ells)
    max_err = let e = 0.0
      for (i, x) in enumerate(py_mu)
        Pref = legendre_pl(x, lmax)
        for (j, ℓ) in enumerate(py_ells)
          e = max(e, abs(D[i, j] - Pref[ℓ+1]))
        end
      end
      e
    end
    @test max_err < 2e-13
  end

  # ── 4. d22 boundary symmetry ──────────────────────────────────────────────
  @testset "d22 boundary symmetry" begin
    ells_t = collect(2:20)
    @test all(abs.(d2n([1.0 - 1e-12], ells_t, 2)[1, :] .- 1.0) .< 1e-4)
    @test all(abs.(d2n([-1.0 + 1e-12], ells_t, 2)[1, :]) .< 1e-4)
    @test all(abs.(d2n([-1.0 + 1e-12], ells_t, -2)[1, :] .- [(-1.0)^ℓ for ℓ in ells_t]) .< 1e-4)
  end

  # ── 5. GL orthogonality ───────────────────────────────────────────────────
  @testset "GL orthogonality ∫Pℓ² dμ = 2/(2ℓ+1)  (ells=2:200)" begin
    ells_t = collect(2:200)
    mu_gl, w_gl = gauss_legendre_weights(500)
    D = d00(mu_gl, ells_t)
    for (j, ℓ) in enumerate(ells_t)
      @test dot(w_gl, D[:, j] .^ 2) ≈ 2 / (2ℓ + 1) atol = 1e-8
    end
  end

  # ── 6. lensed_Cls identity (Clpp=0) at lmax=3000 ─────────────────────────
  @testset "lensed_Cls identity (Clpp=0, LensingConfig(2,$lmax))" begin
    cfg = LensingConfig(2, lmax)
    Nell = length(cfg.lensing_ells)
    ClTT = rand(Nell)
    ClTE = rand(Nell)
    ClEE = rand(Nell)
    TT_l, TE_l, EE_l = lensed_Cls(ClTT, ClTE, ClEE, zeros(Nell), cfg)
    @test TT_l ≈ ClTT rtol = 1e-3
    @test TE_l ≈ ClTE rtol = 1e-3
    @test EE_l ≈ ClEE rtol = 1e-3
  end

  # ── 7. Full pipeline vs Python at ellmax=600 ──────────────────────────────
  @testset "lensed_Cls vs Python ABCMB (Clpp=0, ellmax=600)" begin
    cfg = LensingConfig(2, 600)
    TT_j, TE_j, EE_j = lensed_Cls(
      Float64.(REF["lensed_cls_ClTT_unl_600"]),
      Float64.(REF["lensed_cls_ClTE_unl_600"]),
      Float64.(REF["lensed_cls_ClEE_unl_600"]),
      Float64.(REF["lensed_cls_Clpp_600"]), cfg)
    @test maximum(abs.(TT_j .- Float64.(REF["lensed_cls_ClTT_python_600"]))) < 1e-11
    @test maximum(abs.(TE_j .- Float64.(REF["lensed_cls_ClTE_python_600"]))) < 1e-11
    @test maximum(abs.(EE_j .- Float64.(REF["lensed_cls_ClEE_python_600"]))) < 1e-11
  end

  # ── 8. Full pipeline vs Python at ellmax=3000 (Planck range) ─────────────
  @testset "lensed_Cls vs Python ABCMB (Clpp=0, ellmax=$lmax)" begin
    cfg = LensingConfig(2, lmax)
    TT_j, TE_j, EE_j = lensed_Cls(
      Float64.(REF["lensed_cls_ClTT_unl_3000"]),
      Float64.(REF["lensed_cls_ClTE_unl_3000"]),
      Float64.(REF["lensed_cls_ClEE_unl_3000"]),
      Float64.(REF["lensed_cls_Clpp_3000"]), cfg)
    @test maximum(abs.(TT_j .- Float64.(REF["lensed_cls_ClTT_python_3000"]))) < 1e-11
    @test maximum(abs.(TE_j .- Float64.(REF["lensed_cls_ClTE_python_3000"]))) < 1e-11
    @test maximum(abs.(EE_j .- Float64.(REF["lensed_cls_ClEE_python_3000"]))) < 1e-11
  end
end
