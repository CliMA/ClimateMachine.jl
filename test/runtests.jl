using Test

function is_test_folder(f, path_separator)
  a = split(f, path_separator)
  dir = joinpath(a[1:end-1], path_separator)
  d = split(dir, path_separator)[end]
  return d=="test"
end

function main()
  if Sys.isunix()
    path_separator = "/"
  elseif Sys.iswindows()
    path_separator = "\\"
  else
    error("path primitives for this OS need to be defined")
  end

  up = ".."*path_separator
  root_dir = @__DIR__
  root_dir = root_dir*path_separator*up
  code_dir = root_dir*"src"*path_separator

  this_file = @__FILE__
  this_file = replace(this_file, "/" => path_separator)
  this_file = replace(this_file, "\\" => path_separator)
  this_file = split(this_file, path_separator)[end-1:end]
  this_file = code_dir*joinpath(this_file, path_separator)

  folders_to_exclude = []
  # push!(folders_to_exclude, "unused") # example for folders to exclude

  all_files = [joinpath(root,f) for (root, dirs, files) in Base.Filesystem.walkdir(code_dir) for f in files]
  all_files = [x for x in all_files if ! any([occursin(y, x) for y in folders_to_exclude])]
  all_files = [replace(x, "/" => path_separator) for x in all_files]
  all_files = [replace(x, "\\" => path_separator) for x in all_files]
  all_files = [x for x in all_files if is_test_folder(x, path_separator)]
  all_files = [x for x in all_files if split(x, ".")[end]==".jl"] # only .jl files
  all_files = [x for x in all_files if !(x==this_file)]

  print("\n******************** Test files:\n")
  for x in all_files
    print(x, "\n")
  end
  print("\n********************\n")

  for f in all_files
    cmd = `$(Base.julia_cmd()) --code-coverage --inline=no --project=$(Base.current_project()) $f`
    @test (run(cmd); true)
  end

end

main()
