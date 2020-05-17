# Installing ClimateMachine

Installation of ClimateMachine can be divided into 3 parts:

1. Installation of Julia (1.3.1)
2. Configuration of MPI (optional)
3. Installation of ClimateMachine

## Installation of Julia (1.3.1)

ClimateMachine uses the Julia programming language; however, at the current time, certain dependencies are only available on Julia 1.3, so please download the download the appropriate binary for your platform from [Julia's old releases](https://julialang.org/downloads/oldreleases/#v131_dec_30_2019).

## Configuration of MPI (optional)

ClimateMachine uses the Message Passing Interface (MPI) for parallelisation, via the [MPI.jl](https://github.com/JuliaParallel/MPI.jl) package. By default this will install and download a compatible MPI implementation, however on high-performance computing systems you will probably want to configure this package to use the system-provided implementation, which can be configured by setting the environment variable `JULIA_MPI_BINARY=system`. See [Configuration of MPI.jl](https://juliaparallel.github.io/MPI.jl/stable/configuration/) for more information.

## Installation of ClimateMachine

As of now, ClimateMachine is not registered as an official Julia package, so you will need to download [the source](https://github.com/CliMA/ClimateMachine.jl.git) with a command, such as:

```
git clone https://github.com/CliMA/ClimateMachine.jl.git
```

Once on your machine, you will need to run ClimateMachine with
```
julia --project=@. -e "using Pkg; Pkg.instantiate(); Pkg.API.precompile()"
```
To test your ClimateMachine installation, please run
```
julia --project=@. $CLIMATEMACHINE_HOME/test/runtests.jl
```
where `$CLIMATEMACHINE_HOME` is the path to the base ClimateMachine directory.

From here, you can run any of the tutorials.
For example, if you would like to run the dry Rayleigh Bernard tutorial, you might run

```julia
julia> include("tutorials/Atmos/dry_rayleigh_benard.jl")
```

or, from outside the REPL, in the `ClimateMachine.jl/` directory:

```
julia --project=. tutorials/Atmos/dry_rayleigh_benard.jl
```

ClimateMachine will default to running on the GPU on a GPU-enabled system.
To run ClimateMachine on the CPU run set the environment variable `CLIMATEMACHINE_GPU` to `false`.
This can be done in the Julia REPL with:

```julia
julia> ENV["CLIMATEMACHINE_GPU"] = false
```

Otherwise, you can initialize ClimateMachine without a GPU with the `ClimateMachine.init(disable_gpu=true)` command or set the environmental variable locally.
