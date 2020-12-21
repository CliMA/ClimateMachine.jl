module Domains

export RectangularDomain

using Printf

abstract type AbstractDomain end

include("rectangular_domain.jl")

end
