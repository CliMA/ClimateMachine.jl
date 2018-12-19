# CLIMA-atmos Dycore

## Setup with CPUs

```bash
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.API.precompile()"
```

## Setup with GPUs

```bash
julia --project=env/gpu -e "using Pkg; Pkg.instantiate(); Pkg.API.precompile()"
```

## Running locally with CPUs

```bash
mpirun -n 4 julia --project=. drivers/rising_thermal_bubble.jl
```

## Running locally with GPUs

```bash
mpirun -n 4 julia --project=env/gpu drivers/rising_thermal_bubble.jl
```
