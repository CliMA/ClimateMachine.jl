Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[]) # JuliaLang/julia/pull/28625

using ClimateMachine, Documenter, Literate

ENV["GKSwstype"] = "100" # https://github.com/jheinen/GR.jl/issues/278#issuecomment-587090846

generated_dir = joinpath(@__DIR__, "src", "generated") # generated files directory
rm(generated_dir, force = true, recursive = true)
mkpath(generated_dir)

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
]

mathengine = MathJax(Dict(
    :TeX => Dict(
        :equationNumbers => Dict(:autoNumber => "AMS"),
        :Macros => Dict(),
    ),
))

format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    mathengine = mathengine,
    collapselevel = 1,
    # prettyurls = !("local" in ARGS),
    # canonical = "https://CliMA.github.io/ClimateMachine.jl/stable/",
)

makedocs(
    sitename = "ClimateMachine",
    doctest = false,
    strict = false,
    linkcheck = false,
    format = format,
    checkdocs = :exports,
    # checkdocs = :all,
    clean = true,
    modules = [Documenter, ClimateMachine],
    pages = pages,
)

include("clean_build_folder.jl")

deploydocs(
    repo = "github.com/CliMA/ClimateMachine.jl.git",
    target = "build",
    push_preview = true,
)
