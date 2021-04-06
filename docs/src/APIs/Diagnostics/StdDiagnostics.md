# [StdDiagnostics](@id StdDiagnostics)

```@meta
CurrentModule = ClimateMachine.StdDiagnostics
```

```@docs
StdDiagnostics
```

### Available groups

#### StdDiagnostics.AtmosLESDefault

Note that this group does not currently density-average the output.

Included diagnostic variables:
- `u`: x-velocity
- `v`: y-velocity
- `w`: z-velocity
- `avg\_rho`: air density (_not_ density-averaged)
- `rho`: air density
- `temp`: air temperature
- `pres`: air pressure
- `thd`: dry potential temperature
- `et`: total specific energy
- `ei`: specific internal energy
- `ht`: specific enthalpy based on total energy
- `hi`: specific enthalpy based on internal energy
- `w\_ht\_sgs`: vertical sgs flux of total specific enthalpy

When a non-`DryModel` moisture model is used:
- `qt`: mass fraction of total water in air
- `ql`: mass fraction of liquid water in air
- `qi`: mass fraction of ice in air
- `qv`: mass fraction of water vapor in air
- `thv`: virtual potential temperature
- `thl`: liquid-ice potential temperature
- `w\_qt\_sgs`: vertical sgs flux of total specific humidity

Some variable products, to allow computing variances and co-variances:
- `uu`
- `vv`
- `ww`
- `www`
- `eiei`
- `wu`
- `wv`
- `wrho`
- `wthd`
- `wei`
- `qtqt`
- `thlthl`
- `wqt`
- `wql`
- `wqi`
- `wqv`
- `wthv`
- `wthl`
- `qtthl`
- `qtei`

#### StdDiagnostics.AtmosGCMDefault

Included diagnostic variables:
- `u`: zonal wind
- `v`: meridional wind
- `w`: vertical wind
- `rho`: air density
- `temp`: air temperature
- `pres`: air pressure
- `thd`: dry potential temperature
- `et`: total specific energy
- `ei`: specific internal energy
- `ht`: specific enthalpy based on total energy
- `hi`: specific enthalpy based on internal energy
- `vort`: vertical component of relative vorticity
- `vort2`: vertical component of relative vorticity from DGModel kernels via a mini balance law

When a non-`DryModel` moisture model is used:
- `qt`: mass fraction of total water in air
- `ql`: mass fraction of liquid water in air
- `qv`: mass fraction of water vapor in air
- `qi`: mass fraction of ice in air
- `thv`: virtual potential temperature
- `thl`: liquid-ice potential temperature
