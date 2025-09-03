# API Reference

This page provides detailed documentation for all public functions and types in Capse.jl.

## Core Types

### `CℓEmulator`

```@docs
Capse.CℓEmulator
```

The main struct for CMB power spectrum emulation. Contains the trained neural network, normalization parameters, and post-processing functions.

**Fields:**
- `TrainedEmulator::AbstractTrainedEmulators`: The trained neural network model
- `ℓgrid::AbstractVector`: Multipole moments (ℓ values) for the power spectrum
- `InMinMax::AbstractMatrix`: Min-max normalization parameters for inputs (2×n_params matrix)
- `OutMinMax::AbstractMatrix`: Min-max normalization parameters for outputs (2×n_ℓ matrix)
- `Postprocessing::Function`: Post-processing function applied to network output

**Example:**
```julia
# Manual construction (advanced users)
Cℓ_emu = Capse.CℓEmulator(
    TrainedEmulator = trained_nn,
    ℓgrid = [2, 3, 4, ..., 2500],
    InMinMax = normalization_input,
    OutMinMax = normalization_output,
    Postprocessing = (input, output, emu) -> output
)
```

## Primary Functions

### `get_Cℓ`

```@docs
Capse.get_Cℓ
```

Computes the CMB angular power spectrum for given cosmological parameters.

**Arguments:**
- `input_params`: Cosmological parameters as a vector (single evaluation) or matrix (batch evaluation)
- `Cℓemu::AbstractCℓEmulators`: The emulator instance

**Returns:**
- Vector of Cℓ values on the emulator's ℓ-grid (single evaluation)
- Matrix of Cℓ values where each column is a spectrum (batch evaluation)

**Input Validation:**
- Checks dimension compatibility with emulator
- Validates against NaN and Inf values
- Supports both 1D vectors and 2D matrices

**Example:**
```julia
# Single evaluation
params = [0.02237, 0.1200, 0.6736, 0.9649, 0.0544, 2.042e-9]
Cℓ = Capse.get_Cℓ(params, Cℓ_emu)

# Batch evaluation
params_batch = rand(6, 100)  # 100 different cosmologies
Cℓ_batch = Capse.get_Cℓ(params_batch, Cℓ_emu)
```

### `get_ℓgrid`

```@docs
Capse.get_ℓgrid
```

Returns the multipole moments (ℓ values) for which the power spectrum is computed.

**Arguments:**
- `CℓEmulator::AbstractCℓEmulators`: The emulator instance

**Returns:**
- `AbstractVector`: Array of ℓ values

**Example:**
```julia
ℓ_values = Capse.get_ℓgrid(Cℓ_emu)
# Typically returns something like [2, 3, 4, ..., 2500]
```

### `get_emulator_description`

```@docs
Capse.get_emulator_description
```

Displays detailed information about the emulator configuration.

**Arguments:**
- `Cℓemu::AbstractCℓEmulators`: The emulator instance

**Returns:**
- `nothing` (prints to stdout)

**Information Displayed:**
- Parameter names and ordering
- Training configuration
- Network architecture details
- Accuracy metrics (if available)
- Version information

**Example:**
```julia
Capse.get_emulator_description(Cℓ_emu)
# Output:
# Emulator Description:
# Parameters: [ωb, ωc, h, ns, σ8, As]
# Architecture: 6 → 64 → 64 → 64 → 64 → 2500
# Training samples: 100,000
# ...
```

### `load_emulator`

```@docs
Capse.load_emulator
```

Loads a pre-trained emulator from disk.

**Arguments:**
- `path::String`: Path to the directory containing emulator files

**Keyword Arguments:**
- `emu`: Backend type (default: `SimpleChainsEmulator`)
  - `SimpleChainsEmulator`: CPU-optimized backend
  - `LuxEmulator`: GPU-capable backend
- `ℓ_file`: Filename for ℓ-grid (default: `"l.npy"`)
- `weights_file`: Filename for network weights (default: `"weights.npy"`)
- `inminmax_file`: Filename for input normalization (default: `"inminmax.npy"`)
- `outminmax_file`: Filename for output normalization (default: `"outminmax.npy"`)
- `nn_setup_file`: Filename for network configuration (default: `"nn_setup.json"`)
- `postprocessing_file`: Filename for post-processing function (default: `"postprocessing.jl"`)

**Returns:**
- `CℓEmulator`: Loaded emulator ready for inference

**Required Files:**
The specified directory must contain:
1. Neural network weights (`.npy` format)
2. ℓ-grid specification (`.npy` format)
3. Normalization parameters (`.npy` format)
4. Network architecture description (`.json` format)
5. Post-processing function (`.jl` format)

**Example:**
```julia
# Default loading
Cℓ_emu = Capse.load_emulator("/path/to/weights/")

# Specify backend
Cℓ_emu = Capse.load_emulator("/path/to/weights/", emu=Capse.LuxEmulator)

# Custom filenames
Cℓ_emu = Capse.load_emulator(
    "/path/to/weights/",
    weights_file = "custom_weights.npy",
    ℓ_file = "multipoles.npy"
)
```

## Backend Types

### `SimpleChainsEmulator`

CPU-optimized backend using SimpleChains.jl. Best for:
- Small to medium networks (< 1M parameters)
- Single-threaded evaluation
- Maximum CPU performance
- Low latency requirements

### `LuxEmulator`

Flexible backend using Lux.jl. Best for:
- Large networks (> 1M parameters)
- GPU acceleration
- Distributed computing
- Mixed precision computation

## Internal Functions

These functions are used internally but may be useful for advanced users:

### `maximin`
Applies min-max normalization to inputs.

### `inv_maximin`
Reverses min-max normalization on outputs.

### `run_emulator`
Executes the neural network forward pass.

## Type Hierarchy

```
AbstractCℓEmulators
└── CℓEmulator

AbstractTrainedEmulators
├── SimpleChainsEmulator
└── LuxEmulator
```

## Error Types

The package may throw the following errors:

- `ArgumentError`: Invalid input dimensions or values
- `AssertionError`: Failed validation checks
- `LoadError`: Missing or corrupted emulator files
- `MethodError`: Incompatible backend or function calls

## Performance Considerations

1. **Memory Layout**: Column-major order for batch processing
2. **Type Stability**: Use consistent Float64 or Float32
3. **Backend Selection**: SimpleChains for CPU, Lux for GPU
4. **Batch Size**: Optimal batch size depends on hardware
   - CPU: 10-100 samples
   - GPU: 100-10,000 samples

## Thread Safety

- `SimpleChainsEmulator`: Thread-safe for read operations
- `LuxEmulator`: Use distributed computing for parallelism
- Emulator structs are immutable (as of v0.3.5)

## GPU Support

For GPU execution with Lux backend:

```julia
using CUDA, Adapt

# Load on CPU
Cℓ_emu = Capse.load_emulator("weights/", emu=Capse.LuxEmulator)

# Move to GPU
Cℓ_emu_gpu = adapt(CuArray, Cℓ_emu)

# Use normally
params_gpu = CuArray(params)
Cℓ_gpu = Capse.get_Cℓ(params_gpu, Cℓ_emu_gpu)
```

## Extending Capse.jl

To implement custom emulator backends:

1. Subtype `AbstractTrainedEmulators`
2. Implement `run_emulator` method
3. Ensure compatibility with normalization functions

Example:
```julia
struct MyCustomEmulator <: AbstractTrainedEmulators
    # Your fields
end

function Capse.run_emulator(input, emu::MyCustomEmulator)
    # Your implementation
end
```