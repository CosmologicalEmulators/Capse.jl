# Capse.jl

| **Documentation** | **Build Status** | **Code style** |
|:--------:|:----------------:|:----------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://cosmologicalemulators.github.io/Capse.jl/dev) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://cosmologicalemulators.github.io/Capse.jl/stable) | [![Build status (Github Actions)](https://github.com/CosmologicalEmulators/Capse.jl/workflows/CI/badge.svg)](https://github.com/CosmologicalEmulators/Capse.jl/actions) [![codecov](https://codecov.io/gh/CosmologicalEmulators/Capse.jl/branch/main/graph/badge.svg?token=0PYHCWVL67)](https://codecov.io/gh/CosmologicalEmulators/Capse.jl) | [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) |


[![arXiv](https://img.shields.io/badge/arXiv-2307.14339-b31b1b.svg)](https://arxiv.org/abs/2307.14339)
![size](https://img.shields.io/github/repo-size/CosmologicalEmulators/Capse.jl)


Repo containing the CMB Angular Power Spectrum Emulator, `Capse.jl`.

`Capse.jl` is entirely written in `Julia`, but can be transparently called from `Python` using the [`pycapse`](https://github.com/CosmologicalEmulators/pycapse) wrapper.

## Installation and usage

Details about installing and using `Capse.jl` can be found in the official [documentation](https://cosmologicalemulators.github.io/Capse.jl/stable/), but can be summerized as follows.

In order to install `Capse.jl`, run from the `Julia` REPL

```julia
using Pkg, Pkg.add(url="https://github.com/CosmologicalEmulators/Capse.jl")
```

After installing it, you need to instantiate a trained emulator, which can be done with

```julia
Cℓ_emu = Capse.load_emulator(weights_folder)
```

where `weights_folder` is the path to the folder with the trained emulator (some of them can be found on [Zenodo](https://zenodo.org/record/8187935)). After this operation, to obtain the predictions for a set of cosmological parameters, put them in an array `x` and run

```julia
Capse.get_Cℓ(x, Cℓ_emu)
```

If you want to retrieve details about the emulators (e.g. which parameters were used to train it etc...) just run

```julia
Capse.get_emulator_description(Cℓ_emu)
```

## Citing

Free usage of the software in this repository is provided, given that you cite our release paper.

M. Bonici, F. Bianchini, J. Ruiz-Zapatero, [_Capse: efficient and auto-differentiable CMB power spectra emulation_](https://arxiv.org/abs/2307.14339)
