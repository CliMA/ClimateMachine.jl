#### Domain decomposition indexing
export DomainDecomp

import Base
abstract type Domain{N} end
struct GridMean{N} <: Domain{N} end
struct Environment{N} <: Domain{N} end
struct Updraft{N} <: Domain{N} end

"""
    get_param(::Type{Domain})

The number of domains in domain `Domain`
"""
get_param(::Type{GridMean{N}}) where N = N
get_param(::Type{Environment{N}}) where N = N
get_param(::Type{Updraft{N}}) where N = N

"""
    DomainDecomp{GM,EN,UD}

Decomposition of number of domains
 - `gm` number of grid-mean domain
 - `en` number of environment sub-domain
 - `ud` number of updraft sub-domains
"""
struct DomainDecomp{GM,EN,UD}
  function DomainDecomp(;gm::GM=0,en::EN=0,ud::UD=0) where {GM<:Int,EN<:Int,UD<:Int}
    @assert 0<=gm<=1
    @assert 0<=en<=1
    @assert 0<=ud
    return new{GridMean{gm},Environment{en},Updraft{ud}}()
  end
end

Base.sum(   ::DomainDecomp{GM,EN,UD}) where {GM,EN,UD} = get_param(GM)+get_param(EN)+get_param(UD)
get_param(  ::DomainDecomp{GM,EN,UD}) where {GM,EN,UD} = (get_param(GM),get_param(EN),get_param(UD))
