#####
##### make sure there are no temporary files files (e.g., *.vtu) left around from the build in `generated_dir`
#####

file_extensions_to_remove = [".vtu", ".pvtu", ".csv", ".vtk", ".dat", ".nc"]
generated_files = [joinpath(root, f) for (root, dirs, files) in Base.Filesystem.walkdir(generated_dir) for f in files]
println("Generated files: $(generated_files)")

filter!(x->any([endswith(x, y) for y in file_extensions_to_remove]), generated_files)

println("Deleting files: $(generated_files)")
for f in generated_files
    rm(f)
end
