#### VarMapper

"""
    get_var_mapper(vars)

Get a `NamedTuple` that maps a variable name
and subdomain ID to a unique index. Example:
```
julia> vars = ((:ρ_0, 1), (:w, 3), (:a, 3), (:α_0, 1));

julia> var_mapper = get_var_mapper(vars)
(ρ_0 = (1,), w = (2, 3, 4), a = (5, 6, 7), α_0 = (8,))
```
"""
function get_var_mapper(vars, dd::DomainDecomp)
  var_names = tuple([v for (v, dss) in vars]...)
  n_sd_per_var = [sum(dd, dss) for (v, dss) in vars]
  end_index, n_sd_max  = cumsum(n_sd_per_var), max(n_sd_per_var...)
  start_index = [1,[1+x for x in end_index][1:end-1]...]
  vals = Tuple([Tuple(a:b) for (a,b) in zip(start_index, end_index)])
  var_mapper = NamedTuple{var_names}(vals)
  D = Dict([v => dss for (v, dss) in vars]...)
  dss_per_var = NamedTuple{Tuple(keys(D))}(Tuple(values(D)))
  return var_mapper, dss_per_var, var_names
end

function get_sd_mapped(vars, idx::DomainIdx, dd::DomainDecomp)
  sd_mapped = Dict()
  al = alldomains(idx)
  for (v,dss) in vars
    idx_ss = DomainIdx(dd, dss)
    if !(haskey(sd_mapped,v))
      sd_mapped[v] = Int[]
    end
    for i in al
      i == gridmean(idx) && push!(sd_mapped[v], gridmean(idx_ss))
      i == environment(idx) && push!(sd_mapped[v], environment(idx_ss))
      if i in updraft(idx) && !(i in sd_mapped[v])
        push!(sd_mapped[v], updraft(idx_ss)...)
      end
    end
  end
  return sd_mapped
end

function get_sv_a_map(idx::DomainIdx, idx_ss::DomainIdx)
  a = zeros(Int, length(alldomains(idx)))
  for i in alldomains(idx)
    if i ≠ 0 # protect against null map
      if i == gridmean(idx) && has_gridmean(idx_ss)
        a[i] = gridmean(idx_ss)
      elseif i == environment(idx) && has_environment(idx_ss)
        a[i] = environment(idx_ss)
      elseif i in updraft(idx) && has_updraft(idx_ss)
        a[i] = updraft(idx_ss)[i]
      else
        a[i] = 0
      end
    end
  end
  return a
end

function get_sd_unmapped(vars, idx::DomainIdx, dd::DomainDecomp)
  sd_unmapped = Dict{Symbol,Vector{Int}}()
  al = alldomains(idx)
  for (v,dss) in vars
    idx_ss = DomainIdx(dd, dss)
    if !(haskey(sd_unmapped,v))
      sd_unmapped[v] = Int[]
    end
    for i in al
      i == gridmean(idx_ss) && push!(sd_unmapped[v], gridmean(idx))
      i == environment(idx_ss) && push!(sd_unmapped[v], environment(idx))
      if i in updraft(idx_ss) && !(i in sd_unmapped[v])
        push!(sd_unmapped[v], updraft(idx)...)
      end
    end
  end
  return sd_unmapped
end
