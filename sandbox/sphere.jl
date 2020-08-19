using Test
using ClimateMachine
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.MPIStateArrays
using ClimateMachine.Abstractions
import ClimateMachine.Abstractions: DiscontinuousSpectralElementGrid
using Impero, Printf, MPI, LinearAlgebra, Statistics, GaussQuadrature
using Makie, GLMakie, AbstractPlotting
using ImageTransformations, Colors
using AbstractPlotting.MakieLayout

include(pwd() * "/sandbox/test_utils.jl")
ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()
const mpicomm = MPI.COMM_WORLD
const FT = Float64

N = 4
Ne_vert  = 1
Ne_horz  = 3
Rrange = range(0.5; length = Ne_vert + 1, stop = FT(1))

topl = StackedCubedSphereTopology(mpicomm, Ne_horz, Rrange)
grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
        meshwarp = Topologies.cubedshellwarp,
)

x, y, z = coordinates(grid)

##
function visualize(g::DiscontinuousSpectralElementGrid)
    e = collect(1:size(x)[2])
    nfaces = 6
    faces = collect(1:nfaces)
    opacities = collect(range(0,1, length = 10))

    scene, layout = layoutscene()

    element = Node{Any}(e[1])
    face = Node{Any}(faces[6])
    opacity = Node{Any}(opacities[1])


    tmpx = @lift(x[:, $element])
    tmpy = @lift(y[:, $element])
    tmpz = @lift(z[:, $element])

    tmpxf = @lift(x[grid.vmap⁻[:, $face, $element]])
    tmpyf = @lift(y[grid.vmap⁻[:, $face, $element]])
    tmpzf = @lift(z[grid.vmap⁻[:, $face, $element]])

    total_color = @lift(RGBA(0,0,0, $opacity))

    lscene = layout[1:4, 2:4] = LScene(scene)
    Makie.scatter!(lscene, x[:], y[:], z[:], color = total_color, markersize = 100.0, strokewidth = 0)
    Makie.scatter!(lscene, tmpx, tmpy, tmpz, color = RGBA(0,0,0,0.5), markersize = 100.0, strokewidth = 0, camera = cam3d!)
    Makie.scatter!(lscene, tmpxf, tmpyf, tmpzf, color = RGBA(1,0,0,1.0), markersize = 100.0, strokewidth = 0, camera = cam3d!)
    supertitle = layout[1,2] = LText(scene, " "^10 * " Gauss-Lobatto Points " * " "^10, textsize = 50, color = :black)

    menu = LMenu(scene, options = zip(e,e))
    menu2 = LMenu(scene, options = zip(faces,faces))
    menu3 = LMenu(scene, options = zip(opacities,opacities))
    layout[1, 1] = vbox!(
        LText(scene, "Element", width = nothing),
        menu,
        LText(scene, "Face", width = nothing),
        menu2,
        LText(scene, "Opacity", width = nothing),
        menu3,
    )
    on(menu.selection) do s
        element[] = s
    end
    on(menu2.selection) do s
        face[] = s
    end
    on(menu3.selection) do s
        opacity[] = s
    end
    display(scene)
end

##
visualize(grid)