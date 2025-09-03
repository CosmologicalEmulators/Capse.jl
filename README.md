# Capse.jl

<div align="center">
  <img src="https://github.com/user-attachments/assets/a414b6e8-e5ed-4655-857b-3c59f26867e3" alt="Capse" width="450" />
</div>

<div align="center">

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://cosmologicalemulators.github.io/Capse.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://cosmologicalemulators.github.io/Capse.jl/dev)
[![Build Status](https://github.com/CosmologicalEmulators/Capse.jl/workflows/CI/badge.svg)](https://github.com/CosmologicalEmulators/Capse.jl/actions)
[![codecov](https://codecov.io/gh/CosmologicalEmulators/Capse.jl/branch/main/graph/badge.svg?token=0PYHCWVL67)](https://codecov.io/gh/CosmologicalEmulators/Capse.jl)
[![arXiv](https://img.shields.io/badge/arXiv-2307.14339-b31b1b.svg)](https://arxiv.org/abs/2307.14339)

[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
![Size](https://img.shields.io/github/repo-size/CosmologicalEmulators/Capse.jl)

</div>

## 🚀 High-Performance CMB Power Spectrum Emulation

**`Capse.jl`** is a `Julia` package for ultra-fast emulation of Cosmic Microwave Background (CMB) Angular Power Spectra. `Capse.jl` delivers cosmological computations **1,000,000× faster** than traditional Boltzmann codes while maintaining research-grade accuracy.

### ✨ Key Features

- **⚡ Lightning Fast**: ~45 microseconds per evaluation (vs ~ seconds for CAMB/CLASS)
- **🔧 Flexible Backends**: CPU-optimized (`SimpleChains`) and GPU-ready (`Lux`)
- **♻️ Auto-Differentiable**: Full gradient support for modern inference pipelines
- **🐍 Python Compatible**: Seamless integration via [`jaxcapse`](https://github.com/CosmologicalEmulators/jaxcapse)

## 📦 Installation

```julia
using Pkg
Pkg.add(url="https://github.com/CosmologicalEmulators/Capse.jl")
```

## 🎮 Quick Start

```julia
using Capse

# Load pre-trained emulator (weights available on Zenodo)
Cℓ_emu = Capse.load_emulator("path/to/weights/")

# Set cosmological parameters
# [ωb, ωc, h, ns, τ, As] - check ordering with get_emulator_description()
params = [0.02237, 0.1200, 0.6736, 0.9649, 0.0544, 2.042e-9]

# Compute power spectrum in microseconds!
Cℓ = Capse.get_Cℓ(params, Cℓ_emu)

# Get the ℓ-grid
ℓ_values = Capse.get_ℓgrid(Cℓ_emu)
```

## 📊 Performance Benchmarks

<div align="center">

| Method | Time per Evaluation | Speedup |
|--------|-------------------|---------|
| `CAMB` (high accuracy) | ~60 seconds | 1× |
| **`Capse.jl`** | **~45 μs** | **~1,300,000×** |

</div>

*Benchmarks on Intel Core i7-13700H (single core)*

## 🔥 Advanced Features

### Batch Processing
```julia
# Process 100 cosmologies simultaneously
params_batch = rand(6, 100)
Cℓ_batch = Capse.get_Cℓ(params_batch, Cℓ_emu)
```

### GPU Acceleration
```julia
using CUDA, Adapt

# Load and move to GPU
Cℓ_emu = Capse.load_emulator("weights/", emu=Capse.LuxEmulator)
Cℓ_emu_gpu = adapt(CuArray, Cℓ_emu)
```

### Gradient-Based Optimization
```julia
using Zygote

function loss(params)
    Cℓ_theory = Capse.get_Cℓ(params, Cℓ_emu)
    return sum((Cℓ_theory - Cℓ_observed).^2)
end

grad = gradient(loss, params)[1]
```

## 🐍 Python Integration

Use `Capse.jl` seamlessly from `Python`:

```python
import jaxcapse

# Load and use just like in Julia
emu = jaxcapse.load_emulator("path/to/weights/")
cl = jaxcapse.get_cl(params, emu)
```

## 📚 Documentation

- [**Stable Documentation**](https://cosmologicalemulators.github.io/Capse.jl/stable) - Latest release
- [**Development Documentation**](https://cosmologicalemulators.github.io/Capse.jl/dev) - Current development branch
- [**API Reference**](https://cosmologicalemulators.github.io/Capse.jl/stable/api) - Detailed function documentation

## 🎓 Citation

If you use `Capse.jl` in your research, please cite our paper:

```bibtex
@article{Bonici2024Capse,
	author = {Bonici, Marco and Bianchini, Federico and Ruiz-Zapatero, Jaime},
	journal = {The Open Journal of Astrophysics},
	doi = {10.21105/astro.2307.14339},
	year = {2024},
	month = {jan 30},
	publisher = {Maynooth Academic Publishing},
	title = {Capse.jl: efficient and auto-differentiable {CMB} power spectra emulation},
	volume = {7},
}
```

## 🤝 Contributing

We welcome contributions! Please follow the [BlueStyle](https://github.com/invenia/BlueStyle) coding standards and see [ColPrac](https://github.com/SciML/ColPrac) for collaboration guidelines.

### Development Setup
```julia
using Pkg
Pkg.develop(url="https://github.com/CosmologicalEmulators/Capse.jl")
Pkg.test("Capse")
```

## 👥 Authors

- **Marco Bonici** - Waterloo Center for Astrophysics
- **Federico Bianchini** - Kavli Institute for Particle Physics and Cosmology
- **Jaime Ruiz-Zapatero** - Advanced Research Computing Centre, UCL
- **Marius Millea** - UC Davis & Berkeley Center for Cosmological Physics

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Made with ❤️ by the Cosmological Emulators Team</sub>
</div>
