# Capse.jl

```@setup tutorial
using Plots; gr()
Plots.reset_defaults()
using JSON
using BenchmarkTools
default(palette = palette(:tab10))
benchmark = BenchmarkTools.load("./assets/capse_benchmark.json")
path_json = "./assets/nn_setup.json"
```

`Capse.jl` is a Julia package designed to emulate the computation of the CMB Angular Power Spectrum, with a speedup of several orders of magnitude.

## Installation

In order to install  `Capse.jl`, run on the `Julia` REPL

```julia
using Pkg, Pkg.add(url="https://github.com/CosmologicalEmulators/Capse.jl")
```

## Usage

If you wanna use `Capse.jl` you need to load a trained emulator.
We recommend to instantiate a trained `Capse-jl` emulator in the following way.

First of all, instantiate a neural network with its weights and biases. This is done through a method imported through the upstream library [`AbstractCosmologicalEmulators.jl`](https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl). In order to use it, you need a dictionary containg the information to instantiate the right neural network architecture and an array containing the weights and the biases (both of them can be found on [Zenodo](https://zenodo.org/record/8187935) and we plan to release more of them).
At the given link above, you can find a `JSON` file with the aforementioned NN architecture. This can be read using the `JSON` library in the following way

```@example tutorial
NN_dict = JSON.parsefile("path_json)
```

This file contains all the informations required to correctly instantiate the neural network.

After this, you just have to pass the `NN_dict` and the `weights` array to the `init_emulator` method and choose a NN backend. In this moment we support two different Neural Networks libraries:

- [SimpleChains](https://github.com/PumasAI/SimpleChains.jl), which is taylored for small NN running on a CPU
- [Lux](https://github.com/LuxDL/Lux.jl), which can run on a GPU

In order to instantiate the emulator, just run

```julia
import Capse
trained_emu = Capse.init_emulator(NN_dict, weights, Capse.SimpleChainsEmulator)
```
`SimpleChains.jl` is faster expecially for small NN on the CPU. If you prefer to use `Lux.jl`, pass as last argument `Capse.LuxEmulator`.

After instantiating the NN, we need:

- the ``\ell`-grid used to train the emulator, `ℓgrid`
- the arrays used to perform the minmax normalization of both input and output features, `InMinMax_array` and `OutMinMax_array`

Now you can instantiate the emulator, using

```julia
Cℓ_emu = Capse.CℓEmulator(TrainedEmulator = trained_emu, ℓgrid = ℓgrid,
                             InMinMax = InMinMax_array,
                             OutMinMax = OutMinMax_array)
```

After loading a trained `CℓTT_emu`, feed it some input parameters `x`.

```julia
import Capse
x = rand(6) # generate some random input
Capse.get_Cℓ(x, Cℓ_emu) #compute the Cℓ's
```

!!! warning

    In this moment the API is **not** stable: we need to pass the input cosmological parameters in an hardcoded way. We are working to add a more stable and flexible API.

Using `SimpleChains.jl`, we obtain a mean execution time of 45 microseconds

```@example tutorial
benchmark[1]["Capse"]["SimpleChains"] # hide
```

Using `Lux.jl`, with the same architecture and weights, we obtain

```@example tutorial
benchmark[1]["Capse"]["Lux"] # hide
```

SimpleChains is about 2 times faster than Lux and they give the same result up to floating point precision.

This benchmarks have been performed locally, with a 12th Gen Intel® Core™ i7-1260P.

Considering that a high-precision settings calculation performed with [`CAMB`](https://github.com/cmbant/CAMB) on the same machine requires around 60 seconds, `Capse.jl` is around ``1,000,000`` times faster.

### Authors

- Marco Bonici, INAF - Institute of Space Astrophysics and Cosmic Physics (IASF), Milano
- Federico Bianchini, PostDoctoral researcher at Stanford
- Jaime Ruiz-Zapatero, PhD Student at Oxford

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

### License

`Capse.jl` is licensed under the MIT "Expat" license; see
[LICENSE](https://github.com/CosmologicalEmulators/Effort.jl/blob/main/LICENSE) for
the full license text.
