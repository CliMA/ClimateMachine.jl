# Running the `ClimateMachine`

The `ClimateMachine` is composed of three models for the Earth system, a
dynamical core, and a number of other components. These are put together to
set up a simulation by a _driver_, for example the Held-Suarez atmospheric
GCM, or the Rising Bubble atmospheric LES. The driver specifies:
- the dimensions and resolution of the simulation domain,
- the duration of the simulation,
- boundary conditions,
- source terms,
- a reference state,
- the turbulence model,
- the moisture model,
- diagnostics of interest,
- initial conditions,
- etc.

Additionally, the driver chooses the time integrator to be used to run the
simulation and may specify the Courant number used to compute the timestep.

Thus, running the `ClimateMachine` requires a driver. For example, the
Held-Suarez atmospheric GCM is run with:

```
$ julia --project experiments/AtmosGCM/heldsuarez.jl
```

## Input and output

The `ClimateMachine` provides the [`ArtifactWrappers`](@ref
ArtifactWrappers-docs) module to assist a driver in sourcing input data for a
simulation, but any mechanism may be used.

Output takes the form of various [groups of diagnostic variables](@ref
Diagnostics-groups) that are written to NetCDF files at user-specified
intervals by the `ClimateMachine` when configured to do so by a driver.

The `ClimateMachine` can also output conservative and auxiliary state variables
to VTK files at specified intervals.

Whether or not output is generated, and if so, at what interval, is a
`ClimateMachine` _setting_.

## `ClimateMachine` settings

Some aspects of the `ClimateMachine`'s behavior can be controlled via its
_settings_ such as use of the GPU, diagnostics output and frequency,
checkpointing/restarting, etc. There are 3 ways in which these settings can be
changed:

1. [Command line arguments](@ref ClimateMachine-clargs) have the highest
   precedence, but it is possible for a driver to disable parsing of command
   line arguments. In such a case, only the next two ways can be used to change
   settings.

3. [Programmatic settings](@ref ClimateMachine-kwargs) have the next highest
   precedence.

2. [Environment variables](@ref ClimateMachine-envvars) have the lowest
   precedence.

### [Command line arguments](@id ClimateMachine-clargs)

If a driver configures the `ClimateMachine` to parse command line arguments (by
passing `parse_clargs = true` to `ClimateMachine.init()`), you can query the
list of arguments understood, for example:

```
$ julia --project experiments/AtmosGCM/heldsuarez.jl --help
usage: experiments/AtmosGCM/heldsuarez.jl [--disable-gpu]
                        [--show-updates <interval>]
                        [--diagnostics <interval>] [--vtk <interval>]
                        [--monitor-timestep-duration <interval>]
                        [--monitor-courant-numbers <interval>]
                        [--checkpoint <interval>]
                        [--checkpoint-keep-all] [--checkpoint-at-end]
                        [--checkpoint-dir <path>]
                        [--restart-from-num <number>]
                        [--fix-rng-seed]
                        [--disable-custom-logger]
                        [--log-level <level>] [--output-dir <path>]
                        [--integration-testing]
                        [--number-of-tracers <number>] [-h]

Climate Machine: an Earth System Model that automatically learns from data

optional arguments:
  --number-of-tracers <number>
                        Number of dummy tracers (type: Int64, default:
                        0)
  -h, --help            show this help message and exit

ClimateMachine:
  --disable-gpu         do not use the GPU
  --show-updates <interval>
                        interval at which to show simulation updates
                        (default: "60secs")
  --diagnostics <interval>
                        interval at which to collect diagnostics
                        (default: "never")
  --vtk <interval>      interval at which to output VTK (default:
                        "never")
  --monitor-timestep-duration <interval>
                        interval in time-steps at which to output
                        wall-clock time per time-step (default:
                        "never")
  --monitor-courant-numbers <interval>
                        interval at which to output acoustic,
                        advective, and diffusive Courant numbers
                        (default: "never")
  --checkpoint <interval>
                        interval at which to create a checkpoint
                        (default: "never")
  --checkpoint-keep-all
                        keep all checkpoints (instead of just the most
                        recent)
  --checkpoint-at-end   create a checkpoint at the end of the
                        simulation
  --checkpoint-dir <path>
                        the directory in which to store checkpoints
                        (default: "checkpoint")
  --restart-from-num <number>
                        checkpoint number from which to restart (in
                        <checkpoint-dir>) (type: Int64, default: -1)
  --fix-rng-seed        set RNG seed to a fixed value for
                        reproducibility
  --disable-custom-logger
                        do not use a custom logger
  --log-level <level>   set the log level to one of
                        debug/info/warn/error (default: "info")
  --output-dir <path>   directory for output data (default: "output")
  --debug-init          fill state arrays with NaNs and dump them
                        post-initialization"
  --integration-testing
                        enable integration testing

Any <interval> unless otherwise stated may be specified as:
    - 2hours or 10mins or 30secs => wall-clock time
    - 9.5smonths or 3.3sdays or 1.5shours => simulation time
    - 1000steps => simulation steps
    - never => disable
    - default => use experiment specified interval (only for diagnostics at present)
```

There may also be driver-specific command line arguments.

### [Programmatic control](@id ClimateMachine-kwargs)

Every `ClimateMachine` setting can also be controlled via keyword arguments to
the `ClimateMachine` initialization function, `ClimateMachine.init()`. For
instance, a driver can specify that VTK output should occur every 5 simulation
minutes with:

```julia
ClimateMachine.init(vtk = "5smins")
```

This can be overridden by by passing `--vtk=never` on the command line, _if_
the `ClimateMachine` is parsing command line arguments.

!!! note

    The `ClimateMachine` will only process command line arguments if a driver
    requests that it do so with:
    ```julia
    ClimateMachine.init(parse_clargs = true)
    ```

### [Environment variables](@id ClimateMachine-envvars)

Every `ClimateMachine` command line argument has an equivalent environment
variable that takes the form `CLIMATEMACHINE_SETTINGS_<SETTING_NAME>`, however
command line arguments and programmatic control have higher precedence.

## Running with MPI

Use MPI to start a distributed run of the `ClimateMachine`. For example:

```
mpiexec -np 4 julia --project experiments/AtmosGCM/heldsuarez.jl
```

will run the Held-Suarez experiment with four MPI ranks. If you are running on
a cluster, you would use this command within a SLURM batch script (or the
equivalent) that allocates four tasks. On a stand-alone machine, MPI will
likely require that you have at least four cores.

Note that unless GPU use is disabled (by changing the setting in one of the
ways described above), each `ClimateMachine` process will use GPU acceleration.
If there are insufficient GPUs (four in the example above), the
`ClimateMachine` processes will share the GPU resources available.

## Scripts for end-to-end runs, logging and visualization

The `ClimateMachine` [wiki](https://github.com/CliMA/ClimateMachine.jl/wiki)
contains detailed examples of [Slurm
scripts](https://github.com/CliMA/ClimateMachine.jl/wiki/Bash-Run-Scripts) that
run the `ClimateMachine`, record specified performance metrics and produce
basic visualization output. 
