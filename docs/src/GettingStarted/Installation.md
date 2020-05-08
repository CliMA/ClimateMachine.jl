# Installing ClimateMachine

## Install Julia

The current release of the `ClimateMachine` is verified to work with
Julia 1.3.1. Download it for your platform from [Julia's old
releases](https://julialang.org/downloads/oldreleases/#v131_dec_30_2019).

## Install MPI (optional)

The `ClimateMachine` uses the Message Passing Interface (MPI) for
distributed processing via the
[MPI.jl](https://github.com/JuliaParallel/MPI.jl) package. This package
downloads and installs an MPI implementation for your platform. However, on
high-performance computing systems, you will probably want to configure this
package to use the system-provided MPI implementation. You can do so by setting
the environment variable `JULIA_MPI_BINARY=system`.

Install `MPI.jl` in Julia using the built-in package manager (press `]` at
the Julia prompt):

```julia
julia> ]
(v1.3) pkg> add MPI
```

The package should be installed and built without errors. You can verify
that all is well with:

```julia
julia> ]
(v1.3) pkg> test MPI
```

If you are having problems, see the [`MPI.jl`
documentation](https://juliaparallel.github.io/MPI.jl/stable/configuration.html)
for help.

## Install the `ClimateMachine`

Download the `ClimateMachine`
[source](https://github.com/CliMA/ClimateMachine.jl.git) (you will need
[`Git`](https://git-scm.com/)):

```
$ git clone https://github.com/CliMA/ClimateMachine.jl.git
```

Now change into the `ClimateMachine.jl` directory and install all the packages
required with:

```
$ julia --project -e 'using Pkg; pkg"instantiate"; pkg"build MPI"'
```

Pre-compile the packages to allow the `ClimateMachine` to start faster:

```
$ julia --project -e 'using Pkg; pkg"precompile"'
```

You can verify your installation with:

```
$ julia --project test/runtests.jl
```

This will take a while!

You are now ready to run one of the tutorials. For instance, the dry
Rayleigh Benard tutorial:

```
$ julia --project tutorials/Atmos/dry_rayleigh_benard.jl
```

The `ClimateMachine` is CUDA-enabled and will use GPU(s) if available. To run
on the CPU, set the environment variable `CLIMATEMACHINE_GPU` to `false`.
