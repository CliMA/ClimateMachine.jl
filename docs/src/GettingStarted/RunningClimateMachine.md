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

## [Command line arguments](@id ClimateMachine-args)

All drivers accept some common command line options. For example:

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
                        [--log-level <level>] [--output-dir <path>]
                        [--integration-testing] [-h]

Climate Machine: an Earth System Model that automatically learns from data

optional arguments:
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
  --log-level <level>   set the log level to one of
                        debug/info/warn/error (default: "info")
  --output-dir <path>   directory for output data (default: "output")
  --integration-testing
                        enable integration testing

Any <interval> unless otherwise stated may be specified as:
    - 2hours or 10mins or 30secs => wall-clock time
    - 9.5smonths or 3.3sdays or 1.5shours => simulation time
    - 1000steps => simulation steps
    - never => disable
    - default => use experiment specified interval (only for diagnostics at
      present)
```

There may also be driver-specific command line arguments.

## Environment variables

Currently, the `ClimateMachine` only looks at one environment variable --
`CLIMATEMACHINE_GPU`, which if set to `false`, disables GPU use.

## Running with MPI

Use MPI to start a distributed run of the `ClimateMachine`. For example:

```
mpiexec -np 4 julia --project experiments/AtmosGCM/heldsuarez.jl
```

will run the Held-Suarez experiment with four MPI ranks. If you are running on
a cluster, you would use this command within a SLURM batch script (or the
equivalent) that allocates four tasks. On a stand-alone machine, MPI will
likely require that you have at least four cores.

Note that unless GPU use is disabled as above, each `ClimateMachine`
process will use GPU acceleration. If there are insufficient GPUs (four
in the example above), the `ClimateMachine` processes will share the
GPU resources available.
