"""
    nextoffset = flatten!(vec, domain, val, offset=1)

Flatten `val`, taking values in `domain`, and transform to Cartesian coordinates. Insert the transformed value into `vec` starting at `offset`. Returns the next offset to use.
"""
flatten!(vec, domain, val) = flatten!(vec, domain, val, 1)

# recursively apply `flatten`
function flatten!(vec, ::Type{T}, val::T, offset) where {T}
  map(zip(fieldnames(T), domains(T))) do (fieldname, domain)
    offset = flatten!(vec, domain, getfield(val, fieldname), offset)
  end
  return offset
end

function flatten!(vec, ::Type{T}, val::T, offset) where {T<:Real}
  vec[offset] = val
  return offset+1
end




"""
    vec = flatten(domain, val)

Flatten `val`, taking values in `domain`, and transform to Cartesian coordinates. Returns a vector `vec` of length `dimension(domain)`.
"""
function flatten(domain, val)
  vec = zeros(Float64, dimension(domain))
  flatten!(vec, domain, val)
  vec
end


"""
    val, nextoffset = unflatten(vec, domain, offset)

Transform from `vec`, starting at `offset`, stored in Cartesian coordinates to value in `domain`. Returns the transformed value and the next offset.
"""
function unflatten end

"""
    val = unflatten(vec, domain)

Transform from `vec` stored in Cartesian coordinates to value in `domain`. Returns the transformed value.
"""
unflatten(vec, domain) = first(unflatten(vec, domain, 1))

function unflatten(vec, ::Type{T}, offset) where {T}
  vals = map(domains(T)) do domain
    val, offset = unflatten(vec, domain, offset)
    val
  end
  return T(vals...), offset
end

function unflatten(vec, ::Type{T}, offset) where {T<:Real}
  vec[offset], offset+1
end
