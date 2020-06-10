####
#### Defines list of tutorials given `generated_dir`
####

generate_tutorials = true

tutorials = []

# Allow flag to skip generated
# tutorials since this is by
# far the slowest part of the
# docs build.
if generate_tutorials

    # generate tutorials
    import Literate

    include("pages_helper.jl")

    tutorials_dir = joinpath(@__DIR__, "..", "tutorials")

    tutorials = [
        "Atmos" => [
            "Dry Rayleigh Bernard" => "Atmos/dry_rayleigh_benard.jl",
            "Dry Idealized GCM" => "Atmos/heldsuarez.jl",
            "Rising Thermal Bubble" => "Atmos/risingbubble.jl",
            "Microphysics" => [
                "Saturation adjustment" =>
                    "Microphysics/ex_1_saturation_adjustment.jl",
                "Kessler" => "Microphysics/ex_2_Kessler.jl",
            ],
        ],
        "Ocean" => [],
        "Land" => ["Heat" => ["Heat Equation" => "Land/Heat/heat_equation.jl"]],
        "Numerics" => [
            "System Solvers" => [
                "Conjugate Gradient" => "Numerics/SystemSolvers/cg.jl",
                "Batched Generalized Minimal Residual" =>
                    "Numerics/SystemSolvers/bgmres.jl",
            ],
            "DG Methods" => [
                "Topology" => "topo.jl",
                "Preserving positivity" => "Numerics/DGMethods/nonnegative.jl",
            ],
        ],
        "Diagnostics" => [
            "Debug" => [
                "State Statistics Regression" =>
                    "Diagnostics/Debug/StateCheck.jl",
            ],
        ],
        "Contributing" => ["Notes on Literate" => "literate_markdown.jl"],
    ]

    # Prepend tutorials_dir
    tutorials_jl = flatten_to_array_of_strings(get_second(tutorials))
    println("Building literate tutorials:")
    for tutorial in tutorials_jl
        println("    $(tutorial)")
    end
    transform(x) = joinpath(basename(generated_dir), replace(x, ".jl" => ".md"))
    tutorials = transform_second(x -> transform(x), tutorials)

    skip_execute = [
        "Atmos/dry_rayleigh_benard.jl",               # takes too long
        "Atmos/risingbubble.jl",                      # broken
        "Numerics/DGMethods/nonnegative.jl",          # broken
        "Microphysics/ex_1_saturation_adjustment.jl", # too long
        "Microphysics/ex_2_Kessler.jl",               # too long
        "topo.jl",                                    # broken
    ]

    tutorials_jl = map(x -> joinpath(tutorials_dir, x), tutorials_jl)

    for tutorial in tutorials_jl
        gen_dir =
            joinpath(generated_dir, relpath(dirname(tutorial), tutorials_dir))
        input = abspath(tutorial)
        script = Literate.script(input, gen_dir)
        code = strip(read(script, String))
        mdpost(str) = replace(str, "@__CODE__" => code)
        Literate.markdown(input, gen_dir, postprocess = mdpost)
        if !any(occursin.(skip_execute, Ref(input)))
            Literate.notebook(input, gen_dir, execute = true)
        end
    end

end
