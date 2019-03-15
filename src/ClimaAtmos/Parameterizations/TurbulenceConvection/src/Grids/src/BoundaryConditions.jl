module BoundaryConditions

import Base

export SingleBC
export BCSet

struct SingleBC{T}
  func :: Function
  val :: T
end

struct BCSet
  bcs :: Vector{Union{SingleBC, Nothing}}
end
BCSet(::Nothing) = BCSet([nothing, nothing])

Base.getindex(A::BCSet, k::Int64) = A.bcs[k]

function set_bcs!(a::BCSet, b::BCSet)
  for (i, v) in enumerate(b.bcs)
    a.bcs[i] = b.bcs[i]
  end
end

end