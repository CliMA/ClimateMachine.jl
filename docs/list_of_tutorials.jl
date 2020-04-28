####
#### Defines list of tutorials given `tutorials_dir` and `generated_dir`
####

# generate tutorials
import Literate

tutorials_files = [
    joinpath(root, f)
    for (root, dirs, files) in Base.Filesystem.walkdir(tutorials_dir)
    for f in files
]

filter!(x -> !occursin("topo.jl", x), tutorials_files)                       # currently broken, TODO: Fix me!
filter!(x -> !occursin("dry_rayleigh_benard.jl", x), tutorials_files)        # currently broken, TODO: Fix me!
filter!(x -> !occursin("ex_001_periodic_advection.jl", x), tutorials_files)  # currently broken, TODO: Fix me!
filter!(x -> !occursin("ex_002_solid_body_rotation.jl", x), tutorials_files) # currently broken, TODO: Fix me!
filter!(x -> !occursin("ex_003_acoustic_wave.jl", x), tutorials_files)       # currently broken, TODO: Fix me!
filter!(x -> !occursin("ex_004_nonnegative.jl", x), tutorials_files)         # currently broken, TODO: Fix me!
filter!(x -> !occursin("KinematicModel.jl", x), tutorials_files)             # currently broken, TODO: Fix me!
filter!(x -> !occursin("ex_1_saturation_adjustment.jl", x), tutorials_files) # currently broken, TODO: Fix me!
filter!(x -> !occursin("ex_2_Kessler.jl", x), tutorials_files)               # currently broken, TODO: Fix me!

println("Building literate tutorials:")
for e in tutorials_files
    println("    $(e)")
end

for tutorial in tutorials_files
    gen_dir = joinpath(generated_dir, relpath(dirname(tutorial), tutorials_dir))
    input = abspath(tutorial)
    script = Literate.script(input, gen_dir)
    code = strip(read(script, String))
    mdpost(str) = replace(str, "@__CODE__" => code)
    Literate.markdown(input, gen_dir, postprocess = mdpost)
    # Literate.notebook(input, gen_dir, execute = true)
end

tutorials = [
    joinpath(root, f)
    for (root, dirs, files) in Base.Filesystem.walkdir(generated_dir)
    for f in files
]
tutorials =
    map(x -> last(split(x, joinpath("docs", "src", "generated"))), tutorials)
# @show tutorials

# TODO: Can/should we construct this Dict automatically?
# We could use `titlecase(replace(basename(x), "_"=>" "))` such that
# the title for `conjugate_gradient.jl` is `Conjugate Gradient`

# These files mirror the .jl files in `CLIMA/tutorials/`:
tutorials = Any[
    "Atmos" => Any[],
    "Ocean" => Any[],
    "Numerics" => Any[
        "LinearSolvers" => Any["Conjugate Gradient" => "generated/Numerics/LinearSolvers/cg.md",],
        "Contributing" => Any["Notes on Literate" => "generated/literate_markdown.md",],
    ],
]

# Allow flag to skip generated
# tutorials since this is by
# far the slowest part of the
# docs build.
if !generate_tutorials
    tutorials = Any[]
end
