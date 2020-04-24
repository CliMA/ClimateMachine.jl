####
#### Defines list of examples given `examples_dir` and `generated_dir`
####

# generate examples
import Literate

examples_files = [
    joinpath(root, f)
    for (root, dirs, files) in Base.Filesystem.walkdir(examples_dir)
    for f in files
]

filter!(x -> !occursin("topo.jl", x), examples_files)                       # currently broken, TODO: Fix me!
filter!(x -> !occursin("dry_rayleigh_benard.jl", x), examples_files)        # currently broken, TODO: Fix me!
filter!(x -> !occursin("ex_001_periodic_advection.jl", x), examples_files)  # currently broken, TODO: Fix me!
filter!(x -> !occursin("ex_002_solid_body_rotation.jl", x), examples_files) # currently broken, TODO: Fix me!
filter!(x -> !occursin("ex_003_acoustic_wave.jl", x), examples_files)       # currently broken, TODO: Fix me!
filter!(x -> !occursin("ex_004_nonnegative.jl", x), examples_files)         # currently broken, TODO: Fix me!
filter!(x -> !occursin("KinematicModel.jl", x), examples_files)             # currently broken, TODO: Fix me!
filter!(x -> !occursin("ex_1_saturation_adjustment.jl", x), examples_files) # currently broken, TODO: Fix me!
filter!(x -> !occursin("ex_2_Kessler.jl", x), examples_files)               # currently broken, TODO: Fix me!

println("Building literate examples:")
for e in examples_files
    println("    $(e)")
end

for example in examples_files
    gen_dir = joinpath(generated_dir, relpath(dirname(example), examples_dir))
    input = abspath(example)
    script = Literate.script(input, gen_dir)
    code = strip(read(script, String))
    mdpost(str) = replace(str, "@__CODE__" => code)
    Literate.markdown(input, gen_dir, postprocess = mdpost)
    #Literate.notebook(input, gen_dir, execute = true)
end

examples = [
    joinpath(root, f)
    for (root, dirs, files) in Base.Filesystem.walkdir(generated_dir)
    for f in files
]
examples =
    map(x -> last(split(x, joinpath("docs", "src", "generated"))), examples)
# @show examples

# TODO: Can/should we construct this Dict automatically?
# We could use `titlecase(replace(basename(x), "_"=>" "))` such that
# the title for `conjugate_gradient.jl` is `Conjugate Gradient`

# These files mirror the .jl files in `CLIMA/examples/`:
examples = Any[
    "Atmos" => Any[],
    "Ocean" => Any[],
    "Numerics" => Any[
        "LinearSolvers" => Any["Conjugate Gradient" => "generated/Numerics/LinearSolvers/cg.md",],
        "Contributing" => Any["Notes on Literate" => "generated/literate_markdown.md",],
    ],
]

# Allow flag to skip generated
# examples since this is by
# far the slowest part of the
# docs build.
if !generate_examples
    examples = Any[]
end
