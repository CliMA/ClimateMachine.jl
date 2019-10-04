#### ExportFuncs

# Provides a set of export functions that operate on StateVec.

using Printf, DelimitedFiles
export export_state, import_state!

include("ExportFuncsPrimitive.jl")

"""
    export_state(sv::StateVec, grid::Grid, dir, filename)

Export StateVec to a human-readable file `filename` in directory `dir`.
"""
function export_state(sv::StateVec, grid::Grid, dir, filename)
  domain = over_elems(grid)
  vn = var_names(sv)
  DT = eltype(grid)

  headers = ["z", [var_string(sv, ϕ, i) for ϕ in vn for i in over_sub_domains(sv, ϕ)]...]

  data = zeros(DT,length(domain), length(headers)-1)
  for ϕ in vn, i in over_sub_domains(sv, ϕ), k in domain
    i_eff = get_i_state_vec(sv.var_mapper, sv.a_map[ϕ], ϕ, i)
    data[k, i_eff] = sv[ϕ, k, i]
  end

  z = grid.zc[domain]
  data_all = hcat(z, data)
  mkpath(dir)

  file = joinpath(dir, filename)
  max_len = 18 # max_len = max(length.(headers)...) # Must match fmt in export_data
  headers = [repeat(" ", max_len-length(x))*x for x in headers]
  export_data(data_all, file, headers)
end

"""
    import_state!(sv::StateVec, grid::Grid, dir, filename)

Import StateVec from a human-readable file `filename` in directory `dir`.
"""
function import_state!(sv::StateVec, grid::Grid, dir, filename)
  domain = over_elems(grid)
  vn = var_names(sv)
  DT = eltype(grid)

  headers = ["z", [var_string(sv, ϕ, i) for ϕ in vn for i in over_sub_domains(sv, ϕ)]...]
  data_all = zeros(DT, length(domain), length(headers))
  mkpath(dir)

  file = joinpath(dir, filename)
  max_len = 18 # max_len = max(length.(headers)...) # Must match fmt in export_data
  headers = [repeat(" ", max_len-length(x))*x for x in headers]
  import_data!(data_all, file, headers)

  data = data_all[:,2:end] # remove z
  for ϕ in vn, i in over_sub_domains(sv, ϕ), k in domain
    i_eff = get_i_state_vec(sv.var_mapper, sv.a_map[ϕ], ϕ, i)
    sv[ϕ, k, i] = data[k, i_eff]
  end

end
