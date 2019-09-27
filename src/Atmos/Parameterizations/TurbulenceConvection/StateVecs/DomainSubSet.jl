#### DomainSubSet

export DomainSubSet
export get_param

"""
    DomainSubSet{GM,EN,UD}

Subset of domains
 - `gm` bool indicating to include grid-mean domain
 - `en` bool indicating to include environment sub-domain
 - `ud` bool indicating to include updraft sub-domains
"""
struct DomainSubSet{GM,EN,UD}
  DomainSubSet(;gm::GM=false,en::EN=false,ud::UD=false) where {GM<:Bool,EN<:Bool,UD<:Bool} =
    new{GridMean{gm},Environment{en},Updraft{ud}}()
end

gridmean(   ::DomainSubSet{GM,EN,UD}) where {GM,EN,UD} = get_param(GM)
environment(::DomainSubSet{GM,EN,UD}) where {GM,EN,UD} = get_param(EN)
updraft(    ::DomainSubSet{GM,EN,UD}) where {GM,EN,UD} = get_param(UD)

gridmean(   ::DomainDecomp{GM,EN,UD}, dss::DomainSubSet) where {GM,EN,UD} = gridmean(dss)    ? get_param(GM) : 0
environment(::DomainDecomp{GM,EN,UD}, dss::DomainSubSet) where {GM,EN,UD} = environment(dss) ? get_param(EN) : 0
updraft(    ::DomainDecomp{GM,EN,UD}, dss::DomainSubSet) where {GM,EN,UD} = updraft(dss)     ? get_param(UD) : 0

get_param(dd::DomainDecomp, dss::DomainSubSet) = (gridmean(dd,dss), environment(dd,dss), updraft(dd,dss))
Base.sum( dd::DomainDecomp, dss::DomainSubSet) = gridmean(dd,dss) + environment(dd,dss) + updraft(dd,dss)
