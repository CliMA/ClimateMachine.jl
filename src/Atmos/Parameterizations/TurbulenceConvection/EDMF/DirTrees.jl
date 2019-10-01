#### DirTrees

export DirTree

"""
    DirTree

A struct containing all output directories.
"""
struct DirTree
  """
  the root output directory
  """
  output :: String

  """
  the directory for output figures
  """
  figs :: String

  """
  a dictionary, initialized by a Tuple of variable names,
  which contains the output directories for those variables.
  """
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

"""
    DirTree(output_folder_name::AbstractString,
            in_vars::Tuple{Vararg{Symbol}})

Gets a DirTree struct so that output directories
can be grabbed using, e.g., dir_tree[:variable_name].
"""
function DirTree(output_folder_name::AbstractString, in_vars::Tuple{Vararg{Symbol}})
  root_dir = "output"*path_separator
  output_folder_name = split(output_folder_name, ".")[end]
  output_folder_name = replace(output_folder_name, "(" => "")
  output_folder_name = replace(output_folder_name, ")" => "")

  output          = root_dir * output_folder_name * path_separator
  figs            = output   * "figs" * path_separator

  vars = Dict{Symbol, String}()
  vars[:initial_conditions]           = figs * "InitialConditions"          * path_separator
  vars[:processed_initial_conditions] = figs * "ProcessedInitialConditions" * path_separator
  vars[:solution_raw]                 = figs * "SolutionRaw"                * path_separator
  vars[:solution_processed]           = figs * "SolutionProcessed"          * path_separator

  make_path_recursive(output)
  make_path_recursive(figs)

  for (k, v) in vars
    make_path_recursive(v)
  end

  return DirTree(output, figs, vars)
end
