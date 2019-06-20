# CLIMA
Climate Machine

| **Documentation**                             | **Build Status**                                                                                                                                         |
|:--------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------------------------------|
| [![latest][docs-latest-img]][docs-latest-url] | [![travis][travis-img]][travis-url] [![appveyor][appveyor-img]][appveyor-url] [![codecov][codecov-img]][codecov-url] |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://climate-machine.github.io/CLIMA/latest/

[travis-img]: https://travis-ci.org/climate-machine/CLIMA.svg?branch=master
[travis-url]: https://travis-ci.org/climate-machine/CLIMA

[appveyor-img]: https://ci.appveyor.com/api/projects/status/anqnislvfn75tduv/branch/master?svg=true
[appveyor-url]: https://ci.appveyor.com/project/climate-machine/clima/branch/master

[codecov-img]: https://codecov.io/gh/climate-machine/CLIMA/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/climate-machine/CLIMA

# Some notes on using CLIMA

The following instructions to install and run the code assume that Julia
version 1.0.1 or greater installed; GPU version requires version 1.2 or greater
and an NVIDIA GPU.

The [MPI.jl][0] package that is used assumes that you have a working MPI
installation and [CMake][1] installed.

## Setup with CPUs

```bash
julia --project=@. -e "using Pkg; Pkg.instantiate(); Pkg.API.precompile()"
```
You can test that things were installed properly with
```bash
julia --project=@. $CLIMA_HOME/test/runtests.jl
```
where `$CLIMA_HOME` is the path to the base CLIMA directory

## Problems building MPI.jl

If you are having problems building MPI.jl then most likely CMake cannot find
your MPI compilers.  Try running

```bash
CC=$(which mpicc) CXX=$(which mpicxx) FC=$(which mpif90) julia --project=@. -e "using Pkg; Pkg.build(\"MPI\")"
```

which points the `CC`, `CXX`, and `FC` environment variables to the version of
MPI in your path.

## Setup with GPUs

```bash
julia --project=$CLIMA_HOME/env/gpu. -e "using Pkg; Pkg.instantiate(); Pkg.API.precompile()"
```
where `$CLIMA_HOME` is the path to the base CLIMA directory

You can test that things were installed properly with
```bash
julia --project=$CLIMA_HOME/env/gpu $CLIMA_HOME/test/runtests.jl
```

[0]: https://github.com/JuliaParallel/MPI.jl
[1]: https://cmake.org
