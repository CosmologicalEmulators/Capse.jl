# Capse.jl

```@setup tutorial
using Plots; gr()
Plots.reset_defaults()
using BenchmarkTools
default(palette = palette(:tab10))
benchmark = BenchmarkTools.load("./assets/capse_benchmark.json")
```

`Capse.jl` is a Julia package designed to emulate the computation of the CMB Angular Power Spectrum, with a speedup of several orders of magnitude.

## Installation

In order to install  `Capse.jl`, run on the `Julia` REPL

```julia
using Pkg, Pkg.add(url="https://github.com/CosmologicalEmulators/AbstractEmulator.jl")
```

## Usage

In order to use `Capse.jl` you need a trained emulator (some of them can be found on [Zenodo](https://zenodo.org/record/8187935)).
After loading a trained `CℓTE_emu`, feed it some input parameters `x`.

```julia
import Capse
x = rand(6) # generate some random input
Capse.get_Cℓ(x, CℓTE_emu) #compute the TT angular spectrum
```

!!! warning

    in this moment the API is **not** stable: we need to pass the input cosmological parameters in an hardcoded way. We are working to add a more stable and flexible API.

This computation is quite fast: a benchmark performed locally, with a 12th Gen Intel® Core™ i7-1260P, gives the following result

```@example tutorial
benchmark[1]["Capse"]["Cl"] # hide
```

### Authors

- Marco Bonici, INAF - Institute of Space Astrophysics and Cosmic Physics (IASF), Milano
- Federico Bianchini, PostDoctoral researcher at Stanford
- Jaime Ruiz-Zapatero, PhD Student at Oxford

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

### License

Effort is licensed under the MIT "Expat" license; see
[LICENSE](https://github.com/CosmologicalEmulators/Effort.jl/blob/main/LICENSE) for
the full license text.
