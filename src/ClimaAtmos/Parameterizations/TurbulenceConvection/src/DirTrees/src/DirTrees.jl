module DirTrees

import Base

struct DirTree
  output :: String
  figs :: String
  vars :: Dict{Symbol, String}
end

Base.keys(A::DirTree) = keys(A.vars)
Base.getindex(A::DirTree, k::Symbol) = A.vars[k]
Base.firstindex(A::DirTree) = 1
Base.lastindex(A::DirTree) = length(A.vars)
function Base.setindex!(A::DirTree, B::String, k::Symbol)
  A.vars[k] = B
end

path_separator = Sys.iswindows() ? "\\" : "/"

function make_path_recursive(full_path)
  temp = full_path
  temp = replace(temp, "/"  => path_separator)
  temp = replace(temp, "\\" => path_separator)
  mkpath(temp)
end

function init(output_folder_name, in_vars::Tuple{Vararg{Symbol}})::DirTree
  # root_dir = "." * path_separator # optionally move this
  root_dir = "output"*path_separator

  output          = root_dir * output_folder_name * path_separator
  figs            = output   * "figs" * path_separator

  vars = Dict{Symbol, String}()
  for v in in_vars
    vars[v] = figs * string(v) * path_separator
  end

  vars[:ref_state] = figs * "ref_state" * path_separator
  vars[:environment] = figs * "environment" * path_separator
  vars[:grid_mean] = figs * "grid_mean" * path_separator
  vars[:updraft] = figs * "updraft" * path_separator

  make_path_recursive(output)
  make_path_recursive(figs)

  for (k, v) in vars
    make_path_recursive(v)
  end

  return DirTree(output, figs, vars)
end

end