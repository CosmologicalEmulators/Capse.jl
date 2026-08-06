using Test
using NPZ
using SimpleChains
using Static
using Capse
using AbstractCosmologicalEmulators: AkimaSplinePlan, CubicSplinePlan, chebpoints
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
    @test Capse.get_training_ℓgrid(capse_emu) == ℓ_test
    @test Capse.get_ℓgrid(capse_emu) == collect(0:200)
    @test length(output) == length(Capse.get_ℓgrid(capse_emu))
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

    # Test combined interpolation method (get_Cℓ with plan)
    # For this test, we'll use the existing `capse_emu` and `cosmo` from the "Capse tests" block
    # and define a new target ℓ-grid.
    local cosmo = ones(6) # Use local to avoid conflict if defined elsewhere
    local emu_for_interp = Capse.CℓEmulator(TrainedEmulator = capse_emu.TrainedEmulator, ℓgrid=ℓ_cheb_desc, 
                                            InMinMax = inminmax, OutMinMax = outminmax, Postprocessing = postprocessing)
    local plan_for_interp = prepare_Cℓ_interpolation(emu_for_interp, ℓ_new)

    # Get raw emulator output (before postprocessing and interpolation)
    Cℓ_pred_raw = Capse.get_emulator_output(cosmo, emu_for_interp)
    # Apply postprocessing manually
    Cℓ_pred = emu_for_interp.Postprocessing(cosmo, Cℓ_pred_raw, emu_for_interp)

    # Interpolate using `interp_Cℓ`
    Cℓ_interp = interp_Cℓ(Cℓ_pred, plan_for_interp)
    
    # Interpolate using the one-shot `get_Cℓ` method
    Cℓ_oneshot = Capse.get_Cℓ(cosmo, emu_for_interp, plan_for_interp)
    
    # Check shape
    @test length(Cℓ_interp) == length(ℓ_new)
    
    # Check that the two methods yield the exact same result
    @test Cℓ_interp ≈ Cℓ_oneshot
    
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

@testset "Dense spline prediction" begin
    ℓ_sparse = [2.0, 3.5, 7.0, 12.0, 20.0]
    ℓ_dense = collect(2:20)
    values_1 = @. exp(-ℓ_sparse / 10) * (1 + 0.1 * sin(ℓ_sparse))
    values_2 = @. cos(ℓ_sparse / 7) + 0.2 * sin(ℓ_sparse / 3)
    values_matrix = hcat(values_1, values_2)

    plan = SplinePlan(ℓ_sparse)
    @test plan.Plan isa CubicSplinePlan
    @test plan.SourceAscending
    @test plan.PredictionℓGrid == ℓ_dense
    @test plan(values_1) ≈ CubicSplinePlan(ℓ_sparse, ℓ_dense)(values_1) atol=1e-14
    @test plan(values_matrix) ≈ CubicSplinePlan(ℓ_sparse, ℓ_dense)(values_matrix) atol=1e-14
    @test size(plan(values_matrix)) == (length(ℓ_dense), 2)
    @test @inferred(plan(values_1)) ≈ plan(values_1)

    plan_descending = SplinePlan(reverse(ℓ_sparse))
    @test !plan_descending.SourceAscending
    @test plan_descending(reverse(values_1)) ≈ plan(values_1) atol=1e-14
    @test plan_descending(reverse(values_matrix; dims=1)) ≈ plan(values_matrix) atol=1e-14

    akima_plan = SplinePlan(ℓ_sparse; plan_type=AkimaSplinePlan)
    @test akima_plan.Plan isa AkimaSplinePlan
    @test akima_plan(values_1) ≈ AkimaSplinePlan(ℓ_sparse, ℓ_dense)(values_1) atol=1e-14

    identity_plan = Capse.prepare_interpolation_method(collect(2:20))
    dense_values = @. exp(-ℓ_dense / 10)
    dense_values_matrix = hcat(dense_values, 2 .* dense_values)
    @test identity_plan(dense_values) === dense_values
    @test identity_plan(dense_values_matrix) === dense_values_matrix

    oversized_grid = collect(range(2.0, 5000.0; length=2049))
    oversized_method = Capse.prepare_interpolation_method(oversized_grid)
    @test oversized_method isa Capse.IdentityInterpolation
    @test Capse.prepare_interpolation_method(ℓ_sparse; interpolation=:none) isa
          Capse.IdentityInterpolation
    @test Capse.prepare_interpolation_method(
        oversized_grid;
        interpolation=:cubic,
    ) isa SplinePlan

    @test_throws ArgumentError SplinePlan([2.0])
    @test_throws ArgumentError SplinePlan([2.0, 5.0, 4.0, 10.0])
    @test SplinePlan([2.0001, 5.0, 9.999]).PredictionℓGrid == collect(2:10)
    @test SplinePlan([2.999, 5.0, 9.999]).PredictionℓGrid == collect(3:10)
    @test SplinePlan([2.101, 5.0, 9.899]).PredictionℓGrid == collect(3:9)
    @test SplinePlan([2.5, 5.0, 10.5]).PredictionℓGrid == collect(3:10)
    @test_throws ArgumentError SplinePlan(
        [2.0, 5.0, 10.0];
        endpoint_tolerance=0.5,
    )

    n_first_kind = 512
    first_kind_grid = sort(
        @. 4501.0 + 4499.0 * cos(
            (2 * (1:n_first_kind) - 1) * π / (2 * n_first_kind)
        )
    )
    @test SplinePlan(first_kind_grid).PredictionℓGrid == collect(2:9000)

    cosmo = ones(6)
    cosmo_batch = ones(6, 3)
    raw_prediction = Capse.get_emulator_output(cosmo, capse_emu)
    sparse_prediction = capse_emu.Postprocessing(cosmo, raw_prediction, capse_emu)
    dense_prediction = get_Cℓ(cosmo, capse_emu)
    @test get_training_ℓgrid(capse_emu) == ℓ_test
    @test get_ℓgrid(capse_emu) == collect(0:200)
    @test length(dense_prediction) == 201
    @test dense_prediction ≈ capse_emu.InterpolationMethod(sparse_prediction) atol=1e-14

    dense_batch = get_Cℓ(cosmo_batch, capse_emu)
    @test size(dense_batch) == (201, 3)
    raw_batch = Capse.get_emulator_output(cosmo_batch, capse_emu)
    sparse_batch = capse_emu.Postprocessing(cosmo_batch, raw_batch, capse_emu)
    @test dense_batch ≈ capse_emu.InterpolationMethod(sparse_batch) atol=1e-14

    loaded_dense_prediction = get_Cℓ(cosmo, capse_loaded_emu)
    @test loaded_dense_prediction ≈ dense_prediction atol=1e-14
    @test capse_loaded_emu.InterpolationMethod.Plan isa CubicSplinePlan

    second_cosmo = fill(0.9, 6)
    @test get_Cℓ(second_cosmo, capse_emu) != dense_prediction

    for backend in (
        AutoForwardDiff(),
        AutoZygote(),
        AutoMooncake(config=nothing),
    )
        vector_loss = x -> sum(plan(x))
        matrix_loss = x -> sum(plan(x))
        vector_gradient = DifferentiationInterface.gradient(vector_loss, backend, values_1)
        matrix_gradient = DifferentiationInterface.gradient(matrix_loss, backend, values_matrix)
        @test all(isfinite, vector_gradient)
        @test all(isfinite, matrix_gradient)
        @test size(vector_gradient) == size(values_1)
        @test size(matrix_gradient) == size(values_matrix)
    end

    for backend in (AutoForwardDiff(), AutoZygote())
        dense_prediction_loss = x -> sum(get_Cℓ(x, capse_emu))
        dense_prediction_gradient = DifferentiationInterface.gradient(
            dense_prediction_loss,
            backend,
            cosmo,
        )
        @test all(isfinite, dense_prediction_gradient)
        @test size(dense_prediction_gradient) == size(cosmo)
    end
end

@testset "Bundled Emulators" begin
    @test haskey(Capse.trained_emulators, "CAMB_LCDM")
    @test haskey(Capse.trained_emulators["CAMB_LCDM"], "TT")
    
    # Check that we can get one of them successfully and run it
    emu_tt = Capse.trained_emulators["CAMB_LCDM"]["TT"]
    @test emu_tt isa Capse.CℓEmulator
    
    # Provide dummy parameters for LCDM (6 params)
    params = [0.022, 0.12, 67.0, 0.96, 0.05, 2.1e-9]
    Cℓ = Capse.get_Cℓ(params, emu_tt)
    
    @test length(Cℓ) > 0
    @test all(isfinite, Cℓ)
    @test emu_tt.InterpolationMethod isa Capse.IdentityInterpolation
    @test Capse.get_training_ℓgrid(emu_tt) == collect(2:5000)
    @test Capse.get_ℓgrid(emu_tt) == collect(2:5000)
    @test length(Cℓ) == length(Capse.get_ℓgrid(emu_tt))
end
