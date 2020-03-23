using MPI, Test

MPI.Initialized() || MPI.Init()

include("BrickMesh.jl")
include("Elements.jl")
include("Metrics.jl")

include("topology.jl")
include("grid_integral.jl")
include("filter.jl")
include("Geometry.jl")

# runmpi won't work if we do not finalize
# This is not so nice since other tests that are run direction and call MPI.Init
# will fail if we do finalize here (since runmpi won't work in an initialized
# state)
MPI.Initialized() && MPI.Finalize()
