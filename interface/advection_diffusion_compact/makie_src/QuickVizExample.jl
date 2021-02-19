module QV

include("bigfileofstuff.jl")
export cellaverage, cellcenters, coordinates, GridHelper

include("vizinanigans.jl")
export volumeslice, visualize

include("scalarfields.jl")
export ScalarField

include("grid.jl")

end # module
