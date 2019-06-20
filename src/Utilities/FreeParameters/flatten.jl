"""
    nextoffset = flatten!(v, domain, val, offset=1)

Flatten `val`, taking values in `domain`, and transform to Cartesian coordinates. Insert the transformed value into `v` starting at `offset`. Returns the next offset to use.
"""
flatten!(v, domain, val) = flatten!(v, domain, val, 1)

# recursively apply `flatten`
function flatten!(v, ::Type{T}, val::T, offset) where {T}
  map(zip(fieldnames(T), domains(T))) do (fieldname, domain)
    offset = flatten!(v, domain, getfield(val, fieldname), offset)
  end
  return offset
end

function flatten!(v, ::Type{T}, val::T, offset) where {T<:Real}
  v[offset] = val
  return offset+1
end




"""
    v = flatten(domain, val)

Flatten `val`, taking values in `domain`, and transform to Cartesian coordinates. Returns a vector `v` of length `dimension(domain)`.
"""
function flatten(domain, val)
  v = zeros(Float64, dimension(domain))
  flatten!(v, domain, val)
  v
end


"""
    val, nextoffset = unflatten(v, domain, offset)

Transform from `v`, starting at `offset`, stored in Cartesian coordinates to value in `domain`. Returns the transformed value and the next offset.
"""
function unflatten end

"""
    val = unflatten(v, domain)

Transform from `v` stored in Cartesian coordinates to value in `domain`. Returns the transformed value.
"""
unflatten(v, domain) = first(unflatten(v, domain, 1))

function unflatten(v, ::Type{T}, offset) where {T}
  vals = map(domains(T)) do domain
    val, offset = unflatten(v, domain, offset)
    val
  end
  return T(vals...), offset
end

function unflatten(v, ::Type{T}, offset) where {T<:Real}
  v[offset], offset+1
end
