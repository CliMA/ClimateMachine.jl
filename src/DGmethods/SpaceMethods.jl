module SpaceMethods

export AbstractSpaceMethod, AbstractDGMethod, odefun!, postupdate!

abstract type AbstractSpaceMethod end
abstract type AbstractDGMethod <: AbstractSpaceMethod end

odefun!(m::AbstractSpaceMethod) =
error("odefun! not implemented for $(typeof(m))")

end
