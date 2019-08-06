### Subdomain index helper

"""
    SubdomainIdx{N}

Indexes for each sub-domain
"""
struct SubdomainIdx{N}
  gm::Int
  en::Int
  up::NTuple{N,Int}
end

function Base.iterate(i::SubdomainIdx{N}, state=1) where N
  state == i.gm ? (i.gm, state+1) :
  state == i.en ? (i.en, state+1) :
  state == last(i.up)+1 ? nothing :
  (i.up[state-2], state+1)
end

"""
    idomains(::Val{N})

Defines indexing for sub-domain decomposition
"""
idomains(::Val{N}) where N = SubdomainIdx{N}(1, 2, ntuple(i -> i+2, N))
