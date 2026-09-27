using Test
using Capse
using Artifacts
using DifferentiationInterface
import ForwardDiff, Mooncake

@testset "Published CAMB Mnu-w0-wa artifact" begin
    @test Set(keys(Capse.trained_emulators)) == Set(("CAMB_MNUW0WACDM",))
    models = Capse.trained_emulators["CAMB_MNUW0WACDM"]
    @test Set(keys(models)) == Set(("TT", "TE", "EE", "BB", "PP"))
    @test all(model -> model.Postprocessing isa Capse.AsPostprocessing, values(models))
    inputs = [parse.(Float64, split(line)) for line in eachline(joinpath(@__DIR__, "camb_mnuw0wacdm_inputs.txt"))
              if !startswith(line, "#")]
    multipoles = [2, 20, 200, 1000, 3000, 5000, 9500]
    for line in eachline(joinpath(@__DIR__, "camb_mnuw0wacdm_reference.txt"))
        startswith(line, "#") && continue
        fields = split(line)
        spectrum, sample = fields[1], parse(Int, fields[2])
        expected = parse.(Float64, fields[3:end])
        model = models[spectrum]
        @test get_ℓgrid(model) == collect(2:9500)
        prediction = get_Cℓ(inputs[sample], model)
        @test length(prediction) == 9499 && all(isfinite, prediction)
        @test maximum(abs.(prediction[multipoles .- 1] .- expected)) /
              maximum(abs.(expected)) < 1e-12
    end
end

@testset "Bundled models reject unnamed postprocessing" begin
    mktempdir() do dir
        spectrum_dir = joinpath(dir, "TT")
        mkpath(spectrum_dir)
        write(joinpath(spectrum_dir, "nn_setup.json"), "{\"n_input_features\": 9}")
        @test_throws ArgumentError Capse._load_bundled_emulator(dir, "TT")
    end
end

@testset "Published Lux emulator gradients" begin
    manifest = joinpath(pkgdir(Capse), "Artifacts.toml")
    tree = artifact_hash("CAMB_MNUW0WACDM", manifest)
    @test !isnothing(tree)
    artifact_root = artifact_path(tree)
    models = Dict(
        spectrum => Capse.load_emulator(joinpath(artifact_root, spectrum); emu=Capse.LuxEmulator)
        for spectrum in ("TT", "TE", "EE", "BB", "PP")
    )
    params = [3.044, 0.965, 0.054, 67.4, 0.02237, 0.12, 0.06, -1.0, 0.0]
    mooncake = AutoMooncake(config=nothing)

    for (spectrum, emulator) in models
        @test emulator.TrainedEmulator isa Capse.LuxEmulator
        loss = x -> sum(Capse.get_Cℓ(x, emulator)[100:301])
        forward = ForwardDiff.gradient(loss, params)
        prep = DifferentiationInterface.prepare_gradient(loss, mooncake, params)
        reverse = DifferentiationInterface.gradient(loss, prep, mooncake, params)
        @test isapprox(reverse, forward; rtol=1e-10, atol=1e-12)
    end
end
