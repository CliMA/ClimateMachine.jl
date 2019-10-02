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

Base.getindex(A::DirTree, k::Symbol) = A.vars[k]

path_separator = Sys.iswindows() ? "\\" : "/"

"""
    DirTree(output_folder_name::AbstractString,
            in_vars::Tuple{Vararg{Symbol}})

Gets a DirTree struct so that output directories
can be grabbed using, e.g., dir_tree[:variable_name].
"""
function DirTree(output_folder_name::AbstractString, in_vars::Tuple{Vararg{Symbol}})
  root_dir = "output"
  output_folder_name = split(output_folder_name, ".")[end]
  output_folder_name = replace(output_folder_name, "(" => "")
  output_folder_name = replace(output_folder_name, ")" => "")

  output          = joinpath(root_dir,output_folder_name)
  figs            = joinpath(output,"figs")

  vars = Dict{Symbol, String}()
  vars[:initial_conditions]           = joinpath(figs, "InitialConditions")*path_separator
  vars[:processed_initial_conditions] = joinpath(figs, "ProcessedInitialConditions")*path_separator
  vars[:solution_raw]                 = joinpath(figs, "SolutionRaw")*path_separator
  vars[:solution_processed]           = joinpath(figs, "SolutionProcessed")*path_separator

  mkpath(output)
  mkpath(figs)

  for (k, v) in vars
    mkpath(v)
  end

  return DirTree(output, figs, vars)
end
