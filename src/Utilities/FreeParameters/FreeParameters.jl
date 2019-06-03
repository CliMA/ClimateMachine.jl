module FreeParameters

export flatten!, flatten, unflatten, @parameters


include("flatten.jl")
include("domains.jl")
include("macro.jl")

end
