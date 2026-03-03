using Test
using NPZ
using SimpleChains
using Static
using Capse
using AbstractCosmologicalEmulators: chebpoints
using DifferentiationInterface
import ForwardDiff, Zygote, Mooncake

mlpd = SimpleChain(
  static(6),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(identity, 40)
)

ℓ_test = Array(LinRange(0,200, 40))
weights = SimpleChains.init_params(mlpd)
inminmax = rand(6,2)
outminmax = rand(40,2)
npzwrite("emu/l.npy", ℓ_test)
npzwrite("emu/weights.npy", weights)
npzwrite("emu/inminmax.npy", inminmax)
npzwrite("emu/outminmax.npy", outminmax)
emu = Capse.SimpleChainsEmulator(Architecture = mlpd, Weights = weights)

postprocessing(input, output, Cℓemu) = output .* exp(input[1]-3.)

capse_emu = Capse.CℓEmulator(TrainedEmulator = emu, ℓgrid=ℓ_test, InMinMax = inminmax,
                                OutMinMax = outminmax, Postprocessing = postprocessing)
capse_loaded_emu = Capse.load_emulator("emu/")

@testset "Capse tests" begin
    cosmo = ones(6)
    cosmo_vec = ones(6,6)
    output = Capse.get_Cℓ(cosmo,  capse_emu)
    output_vec = Capse.get_Cℓ(cosmo_vec, capse_emu)
    @test isapprox(output_vec[:,1], output)
    @test ℓ_test == Capse.get_ℓgrid(capse_emu)
    @test_logs (:warn, "No emulator description found!") Capse.get_emulator_description(capse_emu)
    @test Capse.get_Cℓ(cosmo_vec, capse_emu) == Capse.get_Cℓ(cosmo_vec, capse_loaded_emu)
end

@testset "Chebyshev interpolation" begin
    # 1. Correctness on smooth function
    K = 39
    ℓ_min, ℓ_max = 2.0, 2500.0
    ℓ_cheb_desc = chebpoints(K, ℓ_min, ℓ_max)  # Descending
    
    # Test function: smooth spectrum-like shape
    f_test(l) = 1e4 * exp(-l / 500)
    
    Cℓ_true_desc = f_test.(ℓ_cheb_desc)
    
    # Create fake emulator with descending grid
    emu_desc = Capse.CℓEmulator(TrainedEmulator = emu, ℓgrid=ℓ_cheb_desc, InMinMax = inminmax,
                                OutMinMax = outminmax, Postprocessing = postprocessing)
    
    ℓ_new = collect(LinRange(ℓ_min, ℓ_max, 100))
    Cℓ_exact = f_test.(ℓ_new)
    
    # Prepare plan
    plan_desc = prepare_Cℓ_interpolation(emu_desc, ℓ_new)
    @test plan_desc.ascending == false
    
    # Vector interp
    Cℓ_interp_desc = interp_Cℓ(Cℓ_true_desc, plan_desc)
    @test length(Cℓ_interp_desc) == 100
    @test isapprox(Cℓ_interp_desc, Cℓ_exact; rtol=1e-10)
    
    # 2. Ascending grid
    ℓ_cheb_asc = reverse(ℓ_cheb_desc)
    Cℓ_true_asc = reverse(Cℓ_true_desc)
    
    emu_asc = Capse.CℓEmulator(TrainedEmulator = emu, ℓgrid=ℓ_cheb_asc, InMinMax = inminmax,
                                OutMinMax = outminmax, Postprocessing = postprocessing)
                                
    plan_asc = prepare_Cℓ_interpolation(emu_asc, ℓ_new)
    @test plan_asc.ascending == true
    
    Cℓ_interp_asc = interp_Cℓ(Cℓ_true_asc, plan_asc)
    @test isapprox(Cℓ_interp_asc, Cℓ_interp_desc; rtol=1e-14)
    
    # 3. Matrix layout
    # Columns are spectra
    Cℓ_mat_desc = hcat(Cℓ_true_desc, Cℓ_true_desc .* 1.1)
    Cℓ_mat_interp = interp_Cℓ(Cℓ_mat_desc, plan_desc)
    @test size(Cℓ_mat_interp) == (100, 2)
    @test isapprox(Cℓ_mat_interp[:, 1], Cℓ_interp_desc; rtol=1e-14)
    @test isapprox(Cℓ_mat_interp[:, 2], Cℓ_interp_desc .* 1.1; rtol=1e-14)
    
    # 4. Warnings on non-Chebyshev grid
    ℓ_uniform = collect(LinRange(ℓ_min, ℓ_max, K+1))
    emu_uniform = Capse.CℓEmulator(TrainedEmulator = emu, ℓgrid=ℓ_uniform, InMinMax = inminmax,
                                OutMinMax = outminmax, Postprocessing = postprocessing)
    @test_logs (:warn, r"The emulator ℓ-grid does not appear to be a Chebyshev grid") prepare_Cℓ_interpolation(emu_uniform, ℓ_new)
end

@testset "Chebyshev AD tests" begin
    K = 39
    ℓ_min, ℓ_max = 2.0, 2500.0
    ℓ_cheb_desc = chebpoints(K, ℓ_min, ℓ_max)
    ℓ_new = collect(LinRange(ℓ_min, ℓ_max, 100))
    
    emu = Capse.CℓEmulator(TrainedEmulator = capse_emu.TrainedEmulator, ℓgrid=ℓ_cheb_desc, 
                           InMinMax = inminmax, OutMinMax = outminmax, Postprocessing = postprocessing)
    plan = prepare_Cℓ_interpolation(emu, ℓ_new)
    
    v = rand(K+1)
    M = rand(K+1, 3)
    
    backends = [
        AutoForwardDiff(),
        AutoZygote(),
        AutoMooncake(config=nothing)
    ]
    
    for b in backends
        @testset "AD backend: $(typeof(b))" begin
            # Vector case
            f_vec(x) = sum(interp_Cℓ(x, plan))
            g_vec = DifferentiationInterface.gradient(f_vec, b, v)
            @test size(g_vec) == size(v)
            @test all(isfinite, g_vec)
            
            # Matrix case
            f_mat(X) = sum(interp_Cℓ(X, plan))
            g_mat = DifferentiationInterface.gradient(f_mat, b, M)
            @test size(g_mat) == size(M)
            @test all(isfinite, g_mat)
        end
    end
end
