# [ClimateMachine Output](@id Model-output)

These are the output data types currently supported by the `ClimateMachine`.

1. **Diagnostics**
    = user-selected variables calculated using the Diagnostics module
    - NetCDF4 with hdf5 format
        - this format allows compression of large datasets with metadata (following the CF convention). It is currently the mainstream format for geospatial data, so using it facilitates use of existing third party analysis packages.
        - this format does not currently support storing connectivity information, so dumping on the DG grid is impractical. The Interpolation module enables the 3D fields to be saved on a user-defined diagnostic grid.
    - a diagnostics group has to be pre-defined and selected by the user (see the [how to guide](@ref How-to-diagnostics)).
    - command line argument: `[--diagnostics <interval>]`
2. **Model variable dump**
    = dump of prognostic and auxiliary variables on the model's DG grid
    - VTK format
        - this format retains connectivity information, so visualisation packages (e.g., [ParaView](https://www.paraview.org) or [VisIt](https://visitusers.org/index.php?title=Main_Page)) can approximately reconstruct the mesh. This is particularly useful for quick visual checks.
    - command line argument: `[--vtk <interval>]`
3. **Checkpoints**
    = prognostic and auxiliary variables, saved separately by each MPI rank (see [this script](https://github.com/CliMA/ClimateMachine.jl/wiki/Assemble-checkpoints) to combine them) for checkpointing and restart purposes
    - binary format (currently JLD2)
    - command line arguments: `[--checkpoint <interval>]`, `[--checkpoint-keep-all]`, `[--checkpoint-at-end]`, `[--checkpoint-dir <path>]`
