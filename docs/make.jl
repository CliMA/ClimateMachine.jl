# reference in tree version of ClimateMachine
push!(Base.LOAD_PATH, joinpath(@__DIR__, ".."))

# https://github.com/jheinen/GR.jl/issues/278#issuecomment-587090846
ENV["GKSwstype"] = "nul"
# some of the tutorials cannot be run on the GPU
ENV["CLIMATEMACHINE_SETTINGS_DISABLE_GPU"] = "true"
# avoid problems with Documenter/Literate when using `global_logger()`
ENV["CLIMATEMACHINE_SETTINGS_DISABLE_CUSTOM_LOGGER"] = "true"

import ArgParse
using Distributed
 
using Documenter
import DocumenterCitations: CitationBibliography

import ClimateMachine

# default doc generated files directory
const OUTPUT_DIR = joinpath(@__DIR__, "src", "generated") 

# default list of extensions to cleanup after doc build
const CLEANUP_OUTPUT_EXT = [".vtu", ".pvtu", ".csv", ".vtk", ".dat", ".nc"]

"""
Remove files recursively under `output_dir` 
matching default list of output file extensions
"""
function cleanup_generated_output!(output_dir::String)
    generated_files = [
        joinpath(root, f)
        for (root, dirs, files) in walkdir(output_dir)
        for f in files
    ]
    filter!(
        x -> any((endswith(x, y) in CLEANUP_OUTPUT_EXT)),
        generated_files
    )
    for f in generated_files
        @info "cleaning up generated file: '$(f)'"
        rm(f)
    end
end

function generate_docs!(args::Dict)
    output_dir = abspath(args["generated_dir"])
    @info "generating output to '$(output_dir)'"

    overwrite_output = args["overwrite"]
    if isdir(output_dir)
        if overwrite_output
            rm(output_dir, force = true, recursive = true)
        end
    else
        mkdir(output_dir)
    end
    
    generate_tutorials = args["generate_tutorials"]
    if generate_tutorials
        remotecall_eval(Main, workers(), quote
            push!(Base.LOAD_PATH, joinpath(@__DIR__, ".."))
            using ClimateMachine
            using Literate
        end)
        include("list_of_tutorials.jl") # defines `tutorials`
    end

    include("list_of_getting_started_docs.jl")      # defines `getting_started_docs`
    include("list_of_how_to_guides.jl")             # defines `how_to_guides`
    include("list_of_apis.jl")                      # defines `apis`
    include("list_of_theory_docs.jl")               # defines `theory_docs`
    include("list_of_dev_docs.jl")                  # defines `dev_docs`

    pages = [
        "Home" => "index.md",
        "Getting started" => getting_started_docs,
        "Tutorials" => generate_tutorials ? tutorials : [],
        "How-to-guides" => how_to_guides,
        "APIs" => apis,
        "Contribution guide" => "Contributing.md",
        "Theory" => theory_docs,
        "Developer docs" => dev_docs,
        "References" => "References.md",
    ]

    mathengine = Documenter.MathJax(Dict(
        :TeX => Dict(
            :equationNumbers => Dict(:autoNumber => "AMS"),
            :Macros => Dict(),
        ),
    ))

    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "") != "" ? true : false,
        mathengine = mathengine,
        collapselevel = 1,
        analytics = get(ENV, "CI", "") != "" ? "UA-191640394-1" : "",
    )

    bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))
    makedoc_kwargs = Dict(
        :sitename => "ClimateMachine",
        :doctest => false,
        :strict => true,
        :format => format,
        :checkdocs => :exports,
        :clean => overwrite_output,
        :modules => [ClimateMachine],
        :pages => pages,
    )
    makedocs(bib; makedoc_kwargs...)

    if args["cleanup_generated_output"]
        cleanup_generated_output!(output_dir)
    end
    if args["deploy_docs"]
        deploydocs(
            repo = "github.com/CliMA/ClimateMachine.jl.git",
            target = "build",
            push_preview = true,
            forcepush = true,
        )
    end
end

function main()
    arg_exc_handler = ArgParse.default_handler
    if Base.isinteractive()
        exc_handler = ArgParse.debug_handler
    end
    arg_settings = ArgParse.ArgParseSettings(
        prog = ArgParse.PROGRAM_FILE,
        description = "ClimateMachine Documentation Build",
        preformatted_description = true, 
        exc_handler = arg_exc_handler,
        autofix_names = true,
        error_on_conflict = true, 
    )
    ArgParse.add_arg_group!(arg_settings, "ClimateMachine Docs")
    ArgParse.@add_arg_table! arg_settings begin
        "--generate-tutorials"
        help = "enable / disable building tutorials"
        metavar = "<true/false>"
        arg_type = Bool
        default = !isempty(get(ENV, "CLIMATEMACHINE_DOCS_GENERATE_TUTORIALS", ""))
        "--generated-dir"
        help = "output path to save generated documentation assets"
        metavar = "<path>"
        arg_type = String
        default = get(ENV, "CLIMATEMACHINE_DOCS_GENERATED_DIR", OUTPUT_DIR)
        "--overwrite"
        help = "overwrite generated documentation output"
        metavar = "<true/false>"
        arg_type = Bool
        default = true
        "--cleanup-generated-output"
        help = "remove output data from the generated build folder"
        metavar = "<true/false>"
        arg_type = Bool
        default = true
        "--deploy-docs"
        help = "enable / disable deploying documentation build"
        metavar = "<true/false>"
        arg_type = Bool 
        default = !isempty(get(ENV, "CI", ""))
    end
    args = ArgParse.parse_args(arg_settings)
    generate_docs!(args)
    return
end

main()