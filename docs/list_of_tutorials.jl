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

    non_tutorial_files = ["KinematicModel.jl"]
    skip_execute = [
        "heldsuarez.jl",                # broken
        "dry_rayleigh_benard.jl",       # takes too long
        "nonnegative.jl",               # takes too long
        "ex_2_Kessler.jl",              # takes too long
        "ex_1_saturation_adjustment.jl", # takes too long
    ]


    # generate tutorials
    import Literate

    tutorials_jl = [
        joinpath(root, f)
        for (root, dirs, files) in Base.Filesystem.walkdir(tutorials_dir)
        for f in files
    ]
    filter!(x -> endswith(x, ".jl"), tutorials_jl) # only grab .jl files

    filter!(
        x -> !any([occursin(y, x) for y in non_tutorial_files]),
        tutorials_jl,
    )

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
        if !any([occursin(y, input) for y in skip_execute])
            Literate.notebook(input, gen_dir, execute = true)
        end
    end

    # TODO: Should we use AutoPages.jl?

    # These files mirror the .jl files in `ClimateMachine.jl/tutorials/`:
    tutorials = Any[
        "Atmos" => Any["Dry Idealized GCM" => "generated/Atmos/heldsuarez.md",],
        "Ocean" => Any[],
        "Numerics" => Any[
            "LinearSolvers" => Any["Conjugate Gradient" => "generated/Numerics/LinearSolvers/cg.md",],
            "Contributing" => Any["Notes on Literate" => "generated/literate_markdown.md",],
        ],
    ]

end
