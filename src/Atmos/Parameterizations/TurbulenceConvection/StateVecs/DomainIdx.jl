#### DomainIdx

export DomainIdx, subdomains, alldomains, eachdomain, allcombinations

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

# TODO: write a state-vector wrapper for the exported functions
"""
    subdomains(idx::DomainIdx)

Tuple of sub-domain indexes, including environment and updrafts
"""
subdomains(idx::DomainIdx)      = (environment(idx),updraft(idx)...)

"""
    alldomains(idx::DomainIdx)

Tuple of all sub-domain indexes, including grid-mean, environment and updrafts
"""
alldomains(idx::DomainIdx)      = (gridmean(idx),environment(idx),updraft(idx)...)

"""
    eachdomain(idx::DomainIdx)

Tuple of indexes including
 - grid-mean
 - environment
 - Tuple of updraft indexes
"""
eachdomain(idx::DomainIdx)      = (gridmean(idx),environment(idx),updraft(idx))

"""
    allcombinations(idx::DomainIdx)

Tuple of indexes including
 - grid-mean
 - environment
 - Tuple of updraft indexes
 - Tuple of all sub-domains
 - Tuple of all domains
"""
allcombinations(idx::DomainIdx) = (gridmean(idx),environment(idx),updraft(idx),subdomains(idx),alldomains(idx))

function DomainIdx(dd::DomainDecomp, dss::DomainSubSet{BGM,BEN,BUD}) where {BGM,BEN,BUD}
  i_gm,i_en,i_ud = get_idx(get_param(dd,dss)...)
  return DomainIdx{i_gm,i_en,i_ud}()
end

@inline get_i_state_vec(vm, a_map::AbstractArray, ϕ::Symbol, i) = vm[ϕ][a_map[i]]
@inline get_i_var(a_map::AbstractArray, i) = a_map[i]

function var_suffix(vm, idx::DomainIdx, idx_ss::DomainIdx, ϕ::Symbol, i)
  if i == gridmean(idx)
    return "_gm"
  elseif i == environment(idx)
    return "_en"
  elseif i in updraft(idx)
    return "_ud_"*string(i)
  else
    throw(BoundsError(vm[ϕ], i))
  end
end

function var_string(vm, idx::DomainIdx, idx_ss::DomainIdx, ϕ::Symbol, i)
  string(ϕ)*var_suffix(vm, idx, idx_ss, ϕ, i)
end
