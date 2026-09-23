using Test
using Capse

@testset "Published CAMB Mnu-w0-wa artifact" begin
    @test Set(keys(Capse.trained_emulators)) == Set(("CAMB_MNUW0WACDM",))
    models = Capse.trained_emulators["CAMB_MNUW0WACDM"]
    @test Set(keys(models)) == Set(("TT", "TE", "EE", "BB", "PP"))
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
