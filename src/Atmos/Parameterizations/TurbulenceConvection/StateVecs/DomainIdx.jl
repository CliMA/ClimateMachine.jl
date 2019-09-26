#### DomainIdx

export DomainIdx,
       gridmean, environment, updraft,
       subdomains, alldomains, eachdomain, allcombinations,
       get_i_state_vec,
       var_suffix

"""
    DomainIdx{GM,EN,UD}

Decomposition of domain indexes, including indexes for
 - `i_gm` grid-mean domain
 - `i_en` environment sub-domain
 - `i_ud` updraft sub-domains

Note that the index ordering logic is defined here.
"""
struct DomainIdx{GM,EN,UD} end

function get_idx(gm,en,ud)
  i_gm,i_en,i_ud = 0,0,(0,)
  ud>0 && (i_ud = Tuple([i for i in 1:ud]))
  en>0 && (i_en = max(i_ud...)+1)
  gm>0 && en>0 && (i_gm = i_en+1)
  gm>0 && !(en>0) && (i_gm = max(i_ud...)+1)
  return i_gm,i_en,i_ud
end

""" Constructor for DomainDecomp """
function DomainIdx(dd::DomainDecomp)
  i_gm,i_en,i_ud = get_idx(get_param(dd)...)
  return DomainIdx{i_gm,i_en,i_ud}()
end

gridmean(   ::DomainIdx{GM,EN,UD}) where {GM,EN,UD} = GM
environment(::DomainIdx{GM,EN,UD}) where {GM,EN,UD} = EN
updraft(    ::DomainIdx{GM,EN,UD}) where {GM,EN,UD} = UD

has_gridmean(   ::DomainIdx{GM,EN,UD}) where {GM,EN,UD} = !(GM==0)
has_environment(::DomainIdx{GM,EN,UD}) where {GM,EN,UD} = !(EN==0)
has_updraft(    ::DomainIdx{GM,EN,UD}) where {GM,EN,UD} = !(UD==(0,))

# Return flat list
subdomains(idx::DomainIdx)      = (environment(idx),updraft(idx)...)
alldomains(idx::DomainIdx)      = (gridmean(idx),environment(idx),updraft(idx)...)

# Return structured
eachdomain(idx::DomainIdx)      = (gridmean(idx),environment(idx),updraft(idx))
allcombinations(idx::DomainIdx) = (gridmean(idx),environment(idx),updraft(idx),subdomains(idx),alldomains(idx))

function DomainIdx(dd::DomainDecomp, dss::DomainSubSet{BGM,BEN,BUD}) where {BGM,BEN,BUD}
  i_gm,i_en,i_ud = get_idx(get_param(dd,dss)...)
  return DomainIdx{i_gm,i_en,i_ud}()
end

@inline get_i_state_vec(vm, a_map::AbstractArray, ϕ::Symbol, i_sd=1) = vm[ϕ][a_map[i_sd]]
@inline get_i_var(a_map::AbstractArray, i_sd=1) = a_map[i_sd]

function var_suffix(vm, idx::DomainIdx, idx_ss::DomainIdx, ϕ::Symbol, i_sd=1)
  if i_sd == gridmean(idx)
    return string(ϕ)*"_gm"
  elseif i_sd == environment(idx)
    return string(ϕ)*"_en"
  elseif i_sd in updraft(idx)
    return string(ϕ)*"_ud_"*string(i_sd)
  else
    throw(BoundsError(vm[ϕ], i_sd))
  end
end
