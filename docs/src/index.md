# Capse.jl Documentation

[![GitHub](https://img.shields.io/badge/github-repo-blue)](https://github.com/CosmologicalEmulators/Capse.jl)
[![arXiv](https://img.shields.io/badge/arXiv-2307.14339-b31b1b.svg)](https://arxiv.org/abs/2307.14339)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/CosmologicalEmulators/Capse.jl/blob/main/LICENSE)

## Overview

**Capse.jl** is a high-performance Julia package for emulating Cosmic Microwave Background (CMB) Angular Power Spectra using neural networks. It provides speedups of 5-6 orders of magnitude compared to traditional Boltzmann codes like CAMB or CLASS, while maintaining high accuracy.

### Key Features

- ⚡ **Ultra-fast inference**: ~45 microseconds per evaluation
- 🎯 **High accuracy**: Trained on high-precision cosmological simulations
- 🔧 **Multiple backends**: Support for CPU (SimpleChains) and GPU (Lux) computation
- 🐍 **Python interoperability**: Seamless integration via [pycapse](https://github.com/CosmologicalEmulators/pycapse)
- ♻️ **Auto-differentiable**: Full support for gradient-based optimization
- 🔬 **Extensible**: Built on AbstractCosmologicalEmulators.jl framework

## Quick Start

### Installation

```julia
using Pkg
Pkg.add(url="https://github.com/CosmologicalEmulators/Capse.jl")
```

### Basic Usage

```julia
using Capse

# Load a trained emulator (download weights from Zenodo)
Cℓ_emu = Capse.load_emulator("path/to/weights/")

# Define cosmological parameters
# Order depends on training - check with get_emulator_description()
params = [0.02237, 0.1200, 0.6736, 0.9649, 0.0544, 2.042e-9]

# Get CMB power spectrum
Cℓ = Capse.get_Cℓ(params, Cℓ_emu)

# Access the ℓ-grid
ℓ_values = Capse.get_ℓgrid(Cℓ_emu)
```

## Detailed Guide

### Loading Emulators

Capse.jl supports loading pre-trained emulators from disk. Trained weights are available on [Zenodo](https://zenodo.org/record/8187935).

```julia
# Default loading (uses SimpleChains backend)
Cℓ_emu = Capse.load_emulator("path/to/weights/")

# Specify backend explicitly
Cℓ_emu = Capse.load_emulator("path/to/weights/", emu=Capse.SimpleChainsEmulator)

# For GPU computation
Cℓ_emu = Capse.load_emulator("path/to/weights/", emu=Capse.LuxEmulator)
```

The weights folder should contain:
- `weights.npy`: Neural network weights
- `l.npy`: ℓ-grid for power spectrum
- `inminmax.npy`: Input normalization parameters
- `outminmax.npy`: Output normalization parameters
- `nn_setup.json`: Network architecture description
- `postprocessing.jl`: Post-processing function

### Understanding Parameters

Each emulator is trained on specific cosmological parameters. To check the expected parameters:

```julia
# Display emulator description including parameter ordering
Capse.get_emulator_description(Cℓ_emu)
```

Common parameter sets include:
- Standard ΛCDM: `[ωb, ωc, h, ns, σ8, As]`
- Extended models with neutrinos, dark energy, etc.

⚠️ **Important**: Parameter ordering must match the training configuration exactly.

### Batch Processing

Process multiple parameter sets efficiently:

```julia
# Create a matrix where each column is a parameter set
params_batch = rand(6, 100)  # 100 different cosmologies

# Get all power spectra at once
Cℓ_batch = Capse.get_Cℓ(params_batch, Cℓ_emu)
```

### Backend Selection

#### SimpleChains (CPU-optimized)
Best for:
- Small to medium neural networks
- Single-threaded applications
- Maximum CPU performance

```julia
using SimpleChains
Cℓ_emu = Capse.load_emulator("weights/", emu=Capse.SimpleChainsEmulator)
```

#### Lux (GPU-capable)
Best for:
- Large neural networks
- GPU acceleration
- Distributed computing

```julia
using Lux, CUDA
Cℓ_emu = Capse.load_emulator("weights/", emu=Capse.LuxEmulator)

# Move to GPU
using Adapt
Cℓ_emu_gpu = adapt(CuArray, Cℓ_emu)
```

### Error Handling

Capse.jl includes robust input validation:

```julia
# These will throw informative errors
try
    # Wrong number of parameters
    Capse.get_Cℓ(rand(5), Cℓ_emu)  # Expects 6 parameters
catch e
    @error "Parameter mismatch" exception=e
end

# Check for invalid values
params = [0.02, 0.12, NaN, 0.96, 0.05, 2e-9]  # NaN will be caught
```

## Performance

### Benchmarks

On a 13th Gen Intel Core i7-13700H (single core):

| Method | Time per evaluation | Speedup |
|--------|-------------------|---------|
| CAMB (high accuracy) | ~60 seconds | 1× |
| CLASS (high accuracy) | ~50 seconds | 0.8× |
| **Capse.jl (SimpleChains)** | **~45 μs** | **~1,300,000×** |
| **Capse.jl (Lux)** | **~50 μs** | **~1,200,000×** |

### Optimization Tips

1. **Use appropriate backend**: SimpleChains for CPU, Lux for GPU
2. **Batch evaluations**: Process multiple parameter sets together
3. **Pre-allocate arrays**: Reuse output arrays when possible
4. **Type stability**: Ensure consistent Float64/Float32 usage

## Advanced Usage

### Custom Post-processing

Define custom post-processing functions:

```julia
function custom_postprocessing(input, output, Cℓemu)
    # Apply custom transformations
    return output .* exp(input[1] - 3.0)
end

Cℓ_emu = Capse.CℓEmulator(
    TrainedEmulator = emu,
    ℓgrid = ℓ_values,
    InMinMax = input_norm,
    OutMinMax = output_norm,
    Postprocessing = custom_postprocessing
)
```

### Integration with Optimization

Use with gradient-based optimizers:

```julia
using Zygote

function loss(params)
    Cℓ_theory = Capse.get_Cℓ(params, Cℓ_emu)
    return sum((Cℓ_theory - Cℓ_observed).^2)
end

# Compute gradients
grad = gradient(loss, params)[1]
```

### Uncertainty Quantification

For emulators with uncertainty estimates:

```julia
# Some emulators may provide uncertainty bands
Cℓ_mean, Cℓ_std = Capse.get_Cℓ_with_uncertainty(params, Cℓ_emu)
```

## Python Integration

Use Capse.jl from Python via [pycapse](https://github.com/CosmologicalEmulators/pycapse):

```python
import pycapse

# Load emulator
emu = pycapse.load_emulator("path/to/weights/")

# Evaluate
import numpy as np
params = np.array([0.02237, 0.1200, 0.6736, 0.9649, 0.0544, 2.042e-9])
cl = pycapse.get_cl(params, emu)
```

## Troubleshooting

### Common Issues

1. **Dimension mismatch error**
   - Check parameter count matches emulator expectation
   - Verify with `get_emulator_description()`

2. **Performance issues with Lux in multi-threading**
   - Known issue: Use distributed computing instead
   - Or switch to SimpleChains for multi-threaded scenarios

3. **GPU out of memory**
   - Reduce batch size
   - Use mixed precision (Float32)

4. **Loading errors**
   - Ensure all required files are in weights folder
   - Check file permissions

## Contributing

We welcome contributions! Please see our [contribution guidelines](https://github.com/CosmologicalEmulators/Capse.jl/blob/main/CONTRIBUTING.md).

### Development Setup

```julia
using Pkg
Pkg.develop(url="https://github.com/CosmologicalEmulators/Capse.jl")
Pkg.test("Capse")
```

## Citation

If you use Capse.jl in your research, please cite:

```bibtex
@article{bonici2023capse,
  title={Capse: efficient and auto-differentiable CMB power spectra emulation},
  author={Bonici, M. and Bianchini, F. and Ruiz-Zapatero, J.},
  journal={arXiv preprint arXiv:2307.14339},
  year={2023}
}
```

## Authors

- **Marco Bonici** - Postdoctoral Researcher, Waterloo Center for Astrophysics
- **Federico Bianchini** - Postdoctoral Researcher, Kavli Institute for Particle Physics and Cosmology
- **Jaime Ruiz-Zapatero** - Research Software Engineer, Advanced Research Computing Centre, UCL
- **Marius Millea** - Researcher, UC Davis and Berkeley Center for Cosmological Physics

## License

MIT License - see [LICENSE](https://github.com/CosmologicalEmulators/Capse.jl/blob/main/LICENSE)

## Acknowledgments

This work builds upon [AbstractCosmologicalEmulators.jl](https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl) and benefits from the Julia ML ecosystem including SimpleChains.jl and Lux.jl.