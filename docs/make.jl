# https://github.com/jheinen/GR.jl/issues/278#issuecomment-587090846
ENV["GKSwstype"] = "nul"
# some of the tutorials cannot be run on the GPU
ENV["CLIMATEMACHINE_SETTINGS_DISABLE_GPU"] = true
# avoid problems with Documenter/Literate when using `global_logger()`
ENV["CLIMATEMACHINE_SETTINGS_DISABLE_CUSTOM_LOGGER"] = true

using Distributed
using DocumenterCitations

run_single_tutorial = false && isempty(get(ENV, "CI", "")) # never use remotely
@everywhere source_dir = joinpath(@__DIR__, "src")
if run_single_tutorial
    tutorial_string_match = "<tutorial_name>"
    single_tutorial_dir = joinpath(@__DIR__, "transient_dir")
    source_dir = single_tutorial_dir
end
bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))

@everywhere push!(LOAD_PATH, joinpath(@__DIR__, ".."))
@everywhere using ClimateMachine
@everywhere using Documenter, Literate

@everywhere const clima_dir = pkgdir(ClimateMachine);

@everywhere GENERATED_DIR = joinpath(source_dir, "generated") # generated files directory
rm(GENERATED_DIR, force = true, recursive = true)
mkpath(GENERATED_DIR)

include("list_of_getting_started_docs.jl")      # defines `getting_started_docs`

include("list_of_tutorials.jl")                 # defines `tutorials`

include("list_of_how_to_guides.jl")             # defines `how_to_guides`
include("list_of_apis.jl")                      # defines `apis`
include("list_of_theory_docs.jl")               # defines `theory_docs`
include("list_of_dev_docs.jl")                  # defines `dev_docs`

pages = Any[
    "Home" => "index.md",
    "Getting started" => getting_started_docs,
    "Tutorials" => tutorials,
    "How-to-guides" => how_to_guides,
    "APIs" => apis,
    "Contribution guide" => "Contributing.md",
    "Theory" => theory_docs,
    "Developer docs" => dev_docs,
    "References" => "References.md",
]

run_single_tutorial && (pages = Any["Tutorials" => tutorials])

mathengine = MathJax(Dict(
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

makedocs(
    bib,
    source = source_dir,
    sitename = "ClimateMachine",
    doctest = false,
    strict = !run_single_tutorial,
    format = format,
    checkdocs = :exports,
    clean = true,
    modules = [ClimateMachine],
    pages = pages,
)

include("clean_build_folder.jl")

deploydocs(
    repo = "github.com/CliMA/ClimateMachine.jl.git",
    target = "build",
    push_preview = true,
    forcepush = true,
)
