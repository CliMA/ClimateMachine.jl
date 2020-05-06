# Installing CLIMA

Installation of CLIMA can be divided into 3 parts:

1. Installation of Julia (1.3.1)
2. Installation of MPI
3. Installation of CLIMA

## Installation of Julia (1.3.1)

The Climate Machine (CLIMA) uses the Julia programming language; however, at the current time, certain dependencies are only available on Julia 1.3, so please download the download the appropriate binary for your platform from [Julia's old releases](https://julialang.org/downloads/oldreleases/#v131_dec_30_2019).

## Installation of MPI

The Message Passing Interface (MPI) is a CLIMA dependency that allows users to send messages between distributed computing devices.
There are multiple versions of MPI available, the most common of which are:

- [Open MPI](https://www.open-mpi.org/)
- [MPICH](https://www.mpich.org/)
- [Microsoft MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
- [SpectrumMPI](https://www.ibm.com/us-en/marketplace/spectrum-mpi) (for POWER systems).

If you are using a cluster, it is very likely that a version of MPI is already installed on your system, and this can probably be queried with `module avail` or some similar command for your system.

If you are working from a desktop or laptop computer or are maintaining a system without MPI, here are our recommendation for desktop users:

- Windows -- [Microsoft MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
- MacOS -- [Open MPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/), which can be installed via a package manager like [homebrew](https://brew.sh/) or installing manually
- Linux -- [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/), which are available from package managers and user repos

Once you have MPI running on your machine, please add the `MPI` package in Julia with

```julia
Pkg> add MPI
Pkg> build MPI
Pkg> test MPI
```

If all the tests pass, you are good to go!

If you are having problems building MPI.jl then most likely you need to set the environment variable `JULIA_MPI_PATH`.
 Additionally, if your MPI is not installed in a single place, e.g., MPI from macports in OSX, you may need to set `JULIA_MPI_INCLUDE_PATH` and `JULIA_MPI_LIBRARY_PATH`; for macports installs of MPI these would be subdirectories in `/opt/local/include` and `/opt/local/lib`.

If issues persist, there might be an issue with the MPI version installed on your device or a problem with the Julia MPI package.
If you suspect there may be a problem with the Julia MPI package, please [create an issue on GitHub](https://github.com/JuliaParallel/MPI.jl)

## Installation of CLIMA

As of now, CLIMA is not registered as an official Julia package, so you will need to download [the source](https://github.com/CliMA/CLIMA.git) with a command, such as:

```
git clone https://github.com/CliMA/CLIMA.git
```

Once on your machine, you will need to run CLIMA with
```
julia --project=@. -e "using Pkg; Pkg.instantiate(); Pkg.API.precompile()"
```
To test your CLIMA installation, please run
```
julia --project=@. $CLIMA_HOME/test/runtests.jl
```
where `$CLIMA_HOME` is the path to the base CLIMA directory.

From here, you can run any of the tutorials.
For example, if you would like to run the dry Rayleigh Bernard tutorial, you might run

```julia
julia> include("tutorials/Atmos/dry_rayleigh_benard.jl")
```

or, from outside the REPL, in the `CLIMA/` directory:

```
julia --project=. tutorials/Atmos/dry_rayleigh_benard.jl
```

CLIMA will default to running on the GPU on a GPU-enabled system.
To run CLIMA on the CPU run set the environment variable `CLIMA_GPU` to `false`.
This can be done in the Julia REPL with:

```julia
julia> ENV["CLIMA_GPU"] = true
```

Otherwise, you can initialize CLIMA without a GPU with the `CLIMA.init(disable_gpu=true)` command or set the environmental variable locally.
