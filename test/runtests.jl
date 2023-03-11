using Test
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

weights = SimpleChains.init_params(mlpd)
emu = Capse.SimpleChainsEmulator(Architecture = mlpd, Weights = weights)
ℓ_test = Array(LinRange(0,200, 40))
capse_emu = Capse.CℓEmulator(TrainedEmulator = emu, ℓgrid=ℓ_test, InMinMax = rand(6,2),
                                OutMinMax = rand(40,2))

@testset "Capse tests" begin
    cosmo = ones(6)
    cosmo_vec = ones(6,6)
    output = Capse.get_Cℓ(cosmo,  capse_emu)
    output_vec = Capse.get_Cℓ(cosmo_vec, capse_emu)
    @test isapprox(output_vec[:,1], output)
end
