"""
    domains(T)

The domains of each field of `T`, returned as a tuple.

A domain is either a `Domain` value, or the declared type of the field.
"""
function domains(::Type{T}) where {T} # fallback
  @static if VERSION > v"1.1"
    fieldtypes(T)
  else
    map(name -> fieldtype(T, name), fieldnames(T))
  end
end


abstract type Domain end

"""
    ConstDomain(val)

Domain of a parameter which can only take a single value.
"""
struct ConstDomain{T} <: Domain
    val::T
end
dimension(d::ConstDomain) = 0

function flatten!(vec, ::ConstDomain, val, offset)
    return offset
end
function unflatten(vec, d::ConstDomain, offset)
    return d.val, offset
end


abstract type RealDomain <: Domain end

"""
    RealDomain(lo, hi)

Domain of a real-valued parameter taking values between `lo` and `hi`.
"""
function RealDomain(lo, hi)
  if hi == +Inf
    if lo == -Inf
      UnboundedRealDomain()
    else
      LowerBoundedRealDomain(lo)
    end
  else
    if lo == -Inf
      UpperBoundedRealDomain(hi)
    else
      IntervalRealDomain(lo, hi)
    end
  end
end


struct UnboundedRealDomain <: RealDomain
end
Base.in(x::Real, ::UnboundedRealDomain) = true

struct LowerBoundedRealDomain <: RealDomain
    lo::Float64
end
Base.in(x::Real, d::LowerBoundedRealDomain) = x > d.lo

struct UpperBoundedRealDomain <: RealDomain
    hi::Float64
end
Base.in(x::Real, d::UpperBoundedRealDomain) = x < d.hi

struct IntervalRealDomain <: RealDomain
    lo::Float64
    hi::Float64
end
Base.in(x::Real, d::IntervalRealDomain) = d.lo < x < d.hi


dimension(d::RealDomain) = 1

tocartesian(d::UnboundedRealDomain, val::Real) = val
tocartesian(d::LowerBoundedRealDomain, val::Real) = log(val-d.lo)
tocartesian(d::UpperBoundedRealDomain, val::Real) = -log(d.hi-val)
tocartesian(d::IntervalRealDomain, val::Real) = log((val-d.lo)/(d.hi-val))

fromcartesian(d::UnboundedRealDomain, cval::Real) = cval
fromcartesian(d::LowerBoundedRealDomain, cval::Real) = exp(cval) + d.lo
fromcartesian(d::UpperBoundedRealDomain, cval::Real) = d.hi - exp(-cval)
fromcartesian(d::IntervalRealDomain, cval::Real) = d.lo + (d.hi-d.lo)/(1+exp(-cval))

function flatten!(vec, d::RealDomain, val::Real, offset)
    vec[offset] = tocartesian(d, val)
    return offset+1
end
function unflatten(vec, d::RealDomain, offset)
    val = fromcartesian(d, vect[offset])
    return val, offset+1
end
