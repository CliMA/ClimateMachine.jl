####
#### Defines list of tutorials given `generated_dir`
####

generate_tutorials = true

tutorials = Any[]

# Allow flag to skip generated
# tutorials since this is by
# far the slowest part of the
# docs build.
if generate_tutorials

    tutorials_dir = joinpath(@__DIR__, "..", "tutorials")      # julia src files

    # generate tutorials
    import Literate

    tutorials_jl = [
        joinpath(root, f)
        for (root, dirs, files) in Base.Filesystem.walkdir(tutorials_dir)
        for f in files
    ]
    filter!(x -> endswith(x, ".jl"), tutorials_jl) # only grab .jl files

    filter!(x -> !occursin("topo.jl", x), tutorials_jl)                       # currently broken, TODO: Fix me!
    filter!(x -> !occursin("dry_rayleigh_benard.jl", x), tutorials_jl)        # currently broken, TODO: Fix me!
    filter!(x -> !occursin("ex_001_periodic_advection.jl", x), tutorials_jl)  # currently broken, TODO: Fix me!
    filter!(x -> !occursin("ex_002_solid_body_rotation.jl", x), tutorials_jl) # currently broken, TODO: Fix me!
    filter!(x -> !occursin("ex_003_acoustic_wave.jl", x), tutorials_jl)       # currently broken, TODO: Fix me!
    filter!(x -> !occursin("ex_004_nonnegative.jl", x), tutorials_jl)         # currently broken, TODO: Fix me!
    filter!(x -> !occursin("KinematicModel.jl", x), tutorials_jl)             # currently broken, TODO: Fix me!
    filter!(x -> !occursin("ex_1_saturation_adjustment.jl", x), tutorials_jl) # currently broken, TODO: Fix me!
    filter!(x -> !occursin("ex_2_Kessler.jl", x), tutorials_jl)               # currently broken, TODO: Fix me!

    println("Building literate tutorials:")
    for tutorial in tutorials_jl
        println("    $(tutorial)")
    end

    for tutorial in tutorials_jl
        gen_dir =
            joinpath(generated_dir, relpath(dirname(tutorial), tutorials_dir))
        input = abspath(tutorial)
        script = Literate.script(input, gen_dir)
        code = strip(read(script, String))
        mdpost(str) = replace(str, "@__CODE__" => code)
        Literate.markdown(input, gen_dir, postprocess = mdpost)
        # Literate.notebook(input, gen_dir, execute = true)
    end

    # TODO: Should we use AutoPages.jl?

    # These files mirror the .jl files in `ClimateMachine.jl/tutorials/`:
    tutorials = Any[
        "Atmos" => Any["Dry Idealized GCM" => "generated/Atmos/heldsuarez.md",],
        "Ocean" => Any[],
        "Numerics" => Any["LinearSolvers" => Any[
            "Conjugate Gradient" => "generated/Numerics/LinearSolvers/cg.md",
            "Batched Generalized Minimal Residual" => "generated/Numerics/LinearSolvers/bgmres.md",
        ],],
        "Contributing" => Any["Notes on Literate" => "generated/literate_markdown.md",],
    ]

end
