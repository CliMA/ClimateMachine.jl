using MPI
using ArgParse
using ClimateMachine
ClimateMachine.init()
using ClimateMachine.MPIStateArrays
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.GenericCallbacks
using ClimateMachine.DGMethods
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Geometry
using ClimateMachine.Orientations

using Test

@testset "EmptyModel Get Minimal Nodal Distance GCM" begin
    FT = Float64
    Npoly = (5, 5)
    Nelem = (8, 5)
    Δh, Δv = get_min_node_distance(AtmosGCMConfigType(), Npoly, Nelem, FT(30e3))
    # Values established from "complete" driver run 
    # min_node_distance is independently tested.
    @test Δh ≈ 103918.2452822826
    @test Δv ≈ 704.834028207901
end
@testset "EmptyModel Get Minimal Nodal Distance LES" begin
    FT = Float64
    Npoly = (5, 5)
    xmax::FT = 2000
    ymax::FT = 400
    zmax::FT = 2000
    Δx::FT = 50
    Δy::FT = 25
    Δz::FT = 10
    # Values established from "complete" driver run 
    # min_node_distance is independently tested.
    Δh, Δv = get_min_node_distance(
        AtmosLESConfigType(),
        Npoly,
        (Δx, Δy, Δz),
        xmax,
        ymax,
        zmax,
    )
    @test Δh ≈ 15.662978404702244
    @test Δv ≈ 5.873616901762716
end
