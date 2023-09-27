# Capse.jl

```@setup tutorial
using Plots; gr()
Plots.reset_defaults()
using JSON
using BenchmarkTools
using NPZ
using Capse
default(palette = palette(:tab10))
benchmark = BenchmarkTools.load("./assets/capse_benchmark.json")
path_json = "./assets/nn_setup.json"
path_data = "./assets/"
weights = rand(500000)
ℓgrid = ones(2000)
InMinMax_array = zeros(2,2000)
OutMinMax_array = zeros(2,2000)
npzwrite("./assets/l.npy", ℓgrid)
npzwrite("./assets/weights.npy", weights)
npzwrite("./assets/inminmax.npy", InMinMax_array)
npzwrite("./assets/outminmax.npy", OutMinMax_array)
weights_folder = "./assets/"
Cℓ_emu = Capse.load_emulator(weights_folder)
```

`Capse.jl` is a Julia package designed to emulate the computation of the CMB Angular Power Spectrum, with a speedup of several orders of magnitude.

## Installation

In order to install  `Capse.jl`, run on the `Julia` REPL

```julia
using Pkg, Pkg.add(url="https://github.com/CosmologicalEmulators/Capse.jl")
```

## Usage

In order to be able to use `Capse.jl`, there two major steps that need to be performed:

- Instantiating the emulators, e.g. initializing the Neural Network, its weight and biases and the quantities employed in pre and post-processing
- Use the instantiated emulators to retrieve the spectra

In the reminder of this section we are showing how this can be done.

### Instantiation

The most direct way to instantiate an official trained emulators is given by the following one-liner

```julia
Cℓ_emu = Capse.load_emulator(weights_folder);
```

where `weights_folder` is the path to the folder containing the files required to build up the network. Some of the trained emulators can be found on [Zenodo](https://zenodo.org/record/8187935) and we plan to release more of them there in the future.

It is possible to pass an additional argument to the previous function, which is used to choose between the two NN backed now available:

- [SimpleChains](https://github.com/PumasAI/SimpleChains.jl), which is taylored for small NN running on a CPU
- [Lux](https://github.com/LuxDL/Lux.jl), which can run on a GPU

`SimpleChains.jl` is faster expecially for small NN on the CPU. If you wanna use something running on a GPU, you should use `Lux.jl`, which can be done adding an additional argument to the `load_emulator` function, `Capse.LuxEmulator`

```julia
Cℓ_emu = Capse.load_emulator(weights_folder, Capse.LuxEmulator);
```

Each trained emulator should be shipped with a description within the JSON file. In order to print the description, just runs:

```@example tutorial
Capse.get_emulator_description(Cℓ_emu)
```

After loading a trained emulator, feed it some input parameters `x` in order to get the emulated $C_\ell$'s

```julia
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

`SimpleChains.jl` is about 2 times faster than `Lux.jl` and they give the same result up to floating point precision.

This benchmarks have been performed locally, with a 12th Gen Intel® Core™ i7-1260P.

Considering that a high-precision settings calculation performed with [`CAMB`](https://github.com/cmbant/CAMB) on the same machine requires around 60 seconds, `Capse.jl` is 5-6 order of magnitudes faster.

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
