module VTK

export writevtk, writepvtu

include("writemesh.jl")
include("writevtk.jl")
include("writepvtu.jl")

function __init__()
    tictoc()
end

end
