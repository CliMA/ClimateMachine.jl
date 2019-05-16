# CLIMA-atmos Dycore

The following instructions to install and run the code assume that Julia
version 1.0.1 or greater installed.  The [MPI.jl][0] package that is used
assumes that you have a working MPI installation and [CMake][1] installed.

## Setup with CPUs

```bash
julia --project=@. -e "using Pkg; Pkg.instantiate(); Pkg.API.precompile()"
```

## Problems building MPI.jl

If you are having problems building MPI.jl then most likely CMake cannot find
your MPI compilers.  Try running

```bash
CC=$(which mpicc) CXX=$(which mpicxx) FC=$(which mpif90) julia --project=@. -e "using Pkg; Pkg.build(\"MPI\")"
```

which points the `CC`, `CXX`, and `FC` environment variables to the version of
MPI in your path.

## Running locally with CPUs

```bash
mpirun -n 4 julia --project=@. test/rising_thermal_bubble.jl
```

[0]: https://github.com/JuliaParallel/MPI.jl
[1]: https://cmake.org
