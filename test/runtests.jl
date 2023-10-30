using Test
using NPZ
using SimpleChains
using Static
using Capse

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

postprocessing(input, output) = output

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
    @test_logs (:warn, "We do not know which parameters were included in the emulators training space. Use this trained emulator with caution!") Capse.get_emulator_description(capse_emu)
    @test Capse.get_Cℓ(cosmo_vec, capse_emu) == Capse.get_Cℓ(cosmo_vec, capse_loaded_emu)
end
