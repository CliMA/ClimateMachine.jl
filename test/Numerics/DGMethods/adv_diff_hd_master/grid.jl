function get_grid(FT, topl, ArrayType, polynomialorder)

    # Specify DG grid
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
        meshwarp = ClimateMachine.Mesh.Topologies.cubedshellwarp,
    )
end