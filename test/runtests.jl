using Test
using NPZ
using SimpleChains
using Static
using Capse
using AbstractCosmologicalEmulators: AkimaSplinePlan, CubicSplinePlan
using DifferentiationInterface
import ForwardDiff, Zygote, Mooncake

mlpd = SimpleChain(
    static(6),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(tanh, 64),
    TurboDense(identity, 40),
)

ℓ_test = Array(LinRange(0, 200, 40))
weights = SimpleChains.init_params(mlpd)
inminmax = rand(6, 2)
outminmax = rand(40, 2)
npzwrite("emu/l.npy", ℓ_test)
npzwrite("emu/weights.npy", weights)
npzwrite("emu/inminmax.npy", inminmax)
npzwrite("emu/outminmax.npy", outminmax)
emu = Capse.SimpleChainsEmulator(Architecture=mlpd, Weights=weights)

postprocessing(input, output, Cℓemu) = output .* exp(input[1] - 3)

capse_emu = Capse.CℓEmulator(
    TrainedEmulator=emu,
    ℓgrid=ℓ_test,
    InMinMax=inminmax,
    OutMinMax=outminmax,
    Postprocessing=postprocessing,
)
capse_loaded_emu = Capse.load_emulator("emu/")

@testset "Capse predictions" begin
    cosmo = ones(6)
    cosmo_batch = ones(6, 6)
    prediction = Capse.get_Cℓ(cosmo, capse_emu)
    prediction_batch = Capse.get_Cℓ(cosmo_batch, capse_emu)

    @test prediction_batch[:, 1] ≈ prediction
    @test Capse.get_training_ℓgrid(capse_emu) == ℓ_test
    @test Capse.get_ℓgrid(capse_emu) == collect(0:200)
    @test length(prediction) == length(Capse.get_ℓgrid(capse_emu))
    @test_logs (:warn, "No emulator description found!") Capse.get_emulator_description(capse_emu)
    @test Capse.get_Cℓ(cosmo_batch, capse_emu) == Capse.get_Cℓ(cosmo_batch, capse_loaded_emu)
end

@testset "Named and legacy postprocessing" begin
    @test capse_loaded_emu.Postprocessing isa Function
    @test !(capse_loaded_emu.Postprocessing isa Capse.AsPostprocessing)

    mktempdir() do dir
        for filename in ("l.npy", "weights.npy", "inminmax.npy", "outminmax.npy")
            cp(joinpath(@__DIR__, "emu", filename), joinpath(dir, filename))
        end
        configuration = read(joinpath(@__DIR__, "emu", "nn_setup.json"), String)
        configuration = replace(configuration, "\"n_input_features\": 6," =>
            "\"n_input_features\": 6, \"postprocessing_name\": \"as_tau_log\", " *
            "\"ln10As_index\": 1, \"tau_index\": 3,")
        write(joinpath(dir, "nn_setup.json"), configuration)

        # No postprocessing.jl exists here. Metadata and explicit names must not load a file.
        function load_and_predict(path; kwargs...)
            loaded = Capse.load_emulator(path; kwargs...)
            return loaded.Postprocessing, Capse.get_Cℓ(ones(6), loaded)
        end
        named_function, named_values = load_and_predict(dir)
        @test named_function isa Capse.AsPostprocessing{true, true}
        @test named_function.tau_index == 3
        @test all(isfinite, named_values)

        explicit_function, _ = load_and_predict(dir;
            postprocessing_name=:as_tau_linear, tau_index=6)
        @test explicit_function isa Capse.AsPostprocessing{false, true}
        @test explicit_function.tau_index == 6
        @test_throws ArgumentError Capse.load_emulator(dir; postprocessing_name=:unknown)
        @test_throws ArgumentError Capse.load_emulator(dir; tau_index=7)

        nested_configuration = replace(configuration,
            "\"postprocessing_name\": \"as_tau_log\", \"ln10As_index\": 1, \"tau_index\": 3," => "")
        nested_configuration = replace(nested_configuration, "\"author\" :" =>
            "\"postprocessing_name\" : \"as_tau_linear\", " *
            "\"ln10As_index\" : 1, \"tau_index\" : 6, \"author\" :")
        write(joinpath(dir, "nn_setup.json"), nested_configuration)
        nested_function, _ = load_and_predict(dir)
        @test nested_function isa Capse.AsPostprocessing{false, true}
        @test nested_function.tau_index == 6

        input = [3.0, 0.96, 0.04, 70.0, 0.12, 0.11]
        output = [0.2, 0.4]
        @test nested_function(input, output, nothing) ≈
            output .* exp(input[1]) .* 1e-10 .* exp(-2 * input[6])
        @test named_function(input, output, nothing) ≈
            exp.(output) .* exp(input[1]) .* 1e-10 .* exp(-2 * input[3])
        input_batch = hcat(input, input .+ [0.1, 0, 0.01, 0, 0, -0.01])
        output_batch = hcat(output, 2 .* output)
        @test nested_function(input_batch, output_batch, nothing) ≈
            output_batch .* reshape(exp.(input_batch[1, :]) .* 1e-10 .* exp.(-2 .* input_batch[6, :]), 1, :)
        log_only = Capse.postprocessing_as_log(1, 0)
        @test log_only(input, output, nothing) ≈ exp.(output) .* exp(input[1]) .* 1e-10
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
    @test Capse.prepare_interpolation_method(oversized_grid; interpolation=:cubic) isa SplinePlan

    @test_throws ArgumentError SplinePlan([2.0])
    @test_throws ArgumentError SplinePlan([2.0, 5.0, 4.0, 10.0])
    @test SplinePlan([2.0001, 5.0, 9.999]).PredictionℓGrid == collect(2:10)
    @test SplinePlan([2.999, 5.0, 9.999]).PredictionℓGrid == collect(3:10)
    @test SplinePlan([2.101, 5.0, 9.899]).PredictionℓGrid == collect(3:9)
    @test SplinePlan([2.5, 5.0, 10.5]).PredictionℓGrid == collect(3:10)
    @test_throws ArgumentError SplinePlan([2.0, 5.0, 10.0]; endpoint_tolerance=0.5)

    nonuniform_grid = sort(@. 4501.0 + 4499.0 * cos((2 * (1:512) - 1) * π / (2 * 512)))
    @test SplinePlan(nonuniform_grid).PredictionℓGrid == collect(2:9000)

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

    @test get_Cℓ(cosmo, capse_loaded_emu) ≈ dense_prediction atol=1e-14
    @test capse_loaded_emu.InterpolationMethod.Plan isa CubicSplinePlan

    for backend in (AutoForwardDiff(), AutoZygote(), AutoMooncake(config=nothing))
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

include("published_artifact.jl")
