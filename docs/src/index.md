# Capse.jl

```@setup tutorial
using Plots; gr()
Plots.reset_defaults()
using JSON
using BenchmarkTools
using NPZ
using Capse
using SimpleChains

mlpd = SimpleChain(
  static(6),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(identity, 4999)
)

default(palette = palette(:tab10))
benchmark = BenchmarkTools.load("./assets/capse_benchmark.json")
path_json = "./assets/nn_setup.json"
path_data = "./assets/"
weights = rand(500000)
ℓgrid = ones(2000)
InMinMax_array = zeros(2,2000)
OutMinMax_array = zeros(2,2000)
nn_setup = JSON.parsefile(path_json)
emu = Capse.SimpleChainsEmulator(Architecture = mlpd, Weights = weights,
                                 Description = nn_setup)
postprocessing(input, output, Cℓemu) = output .* exp(input[1]-3.)
Cℓ_emu = Capse.CℓEmulator(TrainedEmulator = emu, ℓgrid=ℓgrid, InMinMax = InMinMax_array,
                                OutMinMax = OutMinMax_array,
                                Postprocessing = postprocessing)
```

`Capse.jl` is a Julia package designed to emulate the computation of the CMB Angular Power Spectrum, with a speedup of several orders of magnitude compared to standard codes such as `CAMB` or `CLASS`. The core functionalities of `Capse.jl` are inherithed by the upstream library [`AbstractCosmologicalEmulators.jl`](https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl).

## Installation

In order to install  `Capse.jl`, run on the `Julia` REPL

```julia
using Pkg, Pkg.add(url="https://github.com/CosmologicalEmulators/Capse.jl")
```

## Usage

In order to be able to use `Capse.jl`, there are two major steps that need to be performed:

- Instantiating the emulators, e.g. initializing the Neural Network, its weights and biases, and the quantities employed in pre and post-processing
- Use the instantiated emulators to retrieve the spectra

In the reminder of this section we are showing how to do this.

### Instantiation

The most direct way to instantiate an official trained emulators is given by the following one-liner

```julia
Cℓ_emu = Capse.load_emulator(weights_folder);
```

where `weights_folder` is the path to the folder containing the files required to build up the network. Some of the trained emulators can be found on [Zenodo](https://zenodo.org/record/8187935) and we plan to release more of them there in the future.

It is possible to pass an additional argument to the previous function, which is used to choose between the two NN backend now available:

- [SimpleChains](https://github.com/PumasAI/SimpleChains.jl), which is taylored for small NN running on a CPU
- [Lux](https://github.com/LuxDL/Lux.jl), which can run both on CPUs and GPUs

`SimpleChains.jl` is faster expecially for small NNs on the CPU. If you wanna use something running on a GPU, you should use `Lux.jl`, which can be loaded adding an additional argument to the `load_emulator` function, `Capse.LuxEmulator`

```julia
Cℓ_emu = Capse.load_emulator(weights_folder, emu = Capse.LuxEmulator);
```

Each trained emulator should be shipped with a description within the JSON file. In order to print the description, just run:

```@example tutorial
Capse.get_emulator_description(Cℓ_emu)
```

!!! warning

    Cosmological parameters must be fed to `Capse.jl` with **arrays**. It is the user
    responsability to check the right ordering, by reading the output of the
    `get_emulator_description` method.

After loading a trained emulator, feed it some input parameters `x` in order to get the
emulated $C_\ell$'s

```julia
x = rand(6) # generate some random input
Capse.get_Cℓ(x, Cℓ_emu) #compute the Cℓ's
```

Using `SimpleChains.jl`, we obtain a mean execution time of 45 microseconds

```@example tutorial
benchmark[1]["Capse"]["SimpleChains"] # hide
```

Using `Lux.jl`, with the same architecture, we obtain

```@example tutorial
benchmark[1]["Capse"]["Lux"] # hide
```

`SimpleChains.jl` and `Lux.jl` have almost the same performance and they give the same result up to floating point precision.

These benchmarks have been performed locally, with a 13th Gen Intel® Core™ i7-13700H, using a single core.

Considering that a high-precision settings calculation performed with [`CAMB`](https://github.com/cmbant/CAMB) on the same machine requires around 60 seconds, `Capse.jl` is 5-6 order of magnitudes faster.

!!! warning

    Currently, there is a performance issue when using `Lux.jl` in a multi-threaded scenario. This is
    something known (see discussion [here](https://github.com/LuxDL/Lux.jl/issues/847)).
    In case you want to launch multiple chains locally, the suggested (working) strategy with `Lux.jl`
    is to use distributed computing.

### Authors

- Marco Bonici, PostDoctoral researcher at Waterloo Center for Astrophysics
- Federico Bianchini, PostDoctoral researcher at Kavli Institute for Particle Physics and Cosmology
- Jaime Ruiz-Zapatero, Research Software Engineer at the Advanced Research Computing centre of University College London
- Marius Millea, Researcher at UC Davis and Berkeley Center for Cosmological Physics

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you
would like to change.

Please make sure to update tests as appropriate.

### License

`Capse.jl` is licensed under the MIT "Expat" license; see
[LICENSE](https://github.com/CosmologicalEmulators/Effort.jl/blob/main/LICENSE) for the full
license text.
