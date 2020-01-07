"""
    FreeParameters

Infrastructure for injecting/extracting free
parameters for data assimilation/machine learning.
"""
module FreeParameters


export FreeParameter
export extract_free_parameters

mutable struct FreeParameter{T}
  val::T
  hi::T
  lo::T
end

function extract_free_parameters!(fp::Vector{FreeParameter}, s)
  for fn in fieldnames(typeof(s))
    p = getproperty(s,fn)
    if typeof(p) <: FreeParameter
      push!(fp, p)
    else
      extract_free_parameters!(fp, p)
    end
  end
  return nothing
end

function extract_free_parameters(s)
  fp = FreeParameter[]
  extract_free_parameters!(fp, s)
  return fp
end

# function inject_free_parameters!(s, fp::Array{FreeParameter}, i::Int=1)
#   for fn in fieldnames(typeof(s))
#     p = getproperty(s,fn)
#     if typeof(p) <: FreeParameter
#       setproperty!(s, fn, fp[i])
#       i+=1
#     else
#       inject_free_parameters!(p, fp, i)
#     end
#   end
#   return nothing
# end


end