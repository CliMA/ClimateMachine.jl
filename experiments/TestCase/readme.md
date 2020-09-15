# Dycore test cases

## Test cases
Currently we have the following test cases:
- Dry and moist rising thermal bubble (rising-bubble.jl)
- Solid body rotation without flow (solid-body-rotation.jl)
- Dry and moist baroclinic wave (baroclinic-wave.jl)

## Running the tests
For example, the dry rising thermal bubble is run with:
```
$ julia --project experiments/TestCase/rising-bubble.jl
``` 
Please see [RunningClimateMachine](https://clima.github.io/ClimateMachine.jl/latest/GettingStarted/RunningClimateMachine/)
for available command line arguments. To run the moist rising thermal bubble or baroclinic wave, add `--with-moisture`
in the command line arguments.

## Timestepping
The default time steppers are:
- `LSRK144NiegemannDiehlBusch` for rising thermal bubble
- `IMEX` for solid body rotation
- `IMEX` for baroclinic wave

## Resolution and domain size
For rising thermal bubble:
```
N = 4
Δh = FT(125)
Δv = FT(125)
resolution = (Δh, Δh, Δv)
xmax = FT(10000)
ymax = FT(500)
zmax = FT(10000)
```
For solid body rotation and baroclinic wave:
```
poly_order = 3
n_horz = 12
n_vert = 6
domain_height::FT = 30e3
```

To make the aspect ratio of rising thermal bubble similar to that of the GCM 
(you may need to change the time stepper as well), set `Δv = FT(1.25)`.

To make the domain height of rising thermal bubble similar to that of the GCM,
set `zmax = FT(30000)`.

## Numerical Flux
Rusanov flux is used by default. To use other numerical fluxes, specify `numerical_flux_first_order`.

## Filter
The default filters are:
- No filter or hyperdiffusion for rising thermal bubble
- No filter or hyperdiffusion for solid body rotation
- Hyperdiffusion with a time scale of `8h` for baroclinic wave

To add hyperdiffusion, uncomment `hyperdiffusion = DryBiharmonic(FT(8 * 3600)),`.

To add the exponential filter, uncomment `user_callbacks = (cbfilter,),`.

## Other information
Please refer to the [ClimateMachine docs](https://clima.github.io/ClimateMachine.jl/latest/)
for more information.
