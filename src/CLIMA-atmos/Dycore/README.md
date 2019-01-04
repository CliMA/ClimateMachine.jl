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

## CLIMA-atmos Dycore build problems?
CLIMA-atmos Dycore depends on
[`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) which in turn requires
`cmake`. If you have a failed build with the message `MPI not properly
installed` you should try to manually build MPI and check the error message:

```bash
julia --project
]build MPI
```

Common problems are:

  - a lack of `cmake` being on the system
  - the need for environment variables to be set; see the
    [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) website for more details
