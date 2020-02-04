# CLIMA
The Climate Machine is a new Earth system model that leverages recent advances in the computational and data sciences to learn directly from a wealth of Earth observations from space and the ground. The Climate Machine will harness more data than ever before, providing a new level of accuracy to predictions of droughts, heat waves, and rainfall extremes.

| **Documentation**                             | **Build Status**                                                        |
|:--------------------------------------------- |:------------------------------------------------------------------------|
| [![latest][docs-latest-img]][docs-latest-url] | [![azure][azure-img]][azure-url] [![codecov][codecov-img]][codecov-url] |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://climate-machine.github.io/CLIMA/latest/

[azure-img]: https://dev.azure.com/climate-machine/CLIMA/_apis/build/status/climate-machine.CLIMA?branchName=master
[azure-url]: https://dev.azure.com/climate-machine/CLIMA/_build/latest?definitionId=5&branchName=master

[codecov-img]: https://codecov.io/gh/climate-machine/CLIMA/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/climate-machine/CLIMA

# Some notes on using CLIMA

The following instructions to install and run the code assume that Julia
version 1.0.1 or greater installed; GPU version requires version 1.2 or greater
and an NVIDIA GPU.

The [MPI.jl][0] package that is used assumes that you have a working MPI
installation

## Setup

```bash
julia --project=@. -e "using Pkg; Pkg.instantiate(); Pkg.API.precompile()"
```
You can test that things were installed properly with
```bash
julia --project=@. $CLIMA_HOME/test/runtests.jl
```
where `$CLIMA_HOME` is the path to the base CLIMA directory.

CLIMA will default to running on the GPU on a GPU-enabled system.
To force a CPU run set the environment variable `CLIMA_GPU` to `false`.

## Problems building MPI.jl

If you are having problems building MPI.jl then most likely you need to set the
environment variable `JULIA_MPI_PATH`. Additionally, if your MPI is not
installed in a single place, e.g., MPI from macports in OSX, you may need to set
`JULIA_MPI_INCLUDE_PATH` and `JULIA_MPI_LIBRARY_PATH`; for macports installs of
MPI these would be subdirectories in `/opt/local/include` and `/opt/local/lib`.

[0]: https://github.com/JuliaParallel/MPI.jl
