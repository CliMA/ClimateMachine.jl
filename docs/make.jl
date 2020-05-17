Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[]) # JuliaLang/julia/pull/28625

using ClimateMachine, Documenter, Literate
ENV["GKSwstype"] = "100" # https://github.com/jheinen/GR.jl/issues/278#issuecomment-587090846

generated_dir = joinpath(@__DIR__, "src", "generated") # generated files directory
rm(generated_dir, force = true, recursive = true)
mkpath(generated_dir)

include("list_of_experiments.jl")        # defines a nested array `experiments`
include("list_of_tutorials.jl")          # defines a nested array `tutorials`
include("list_of_how_to_guides.jl")      # defines a nested array `how_to_guides`
include("list_of_discussions.jl")        # defines a nested array `apis`
include("list_of_apis.jl")               # defines a nested array `discussions`

pages = Any[
    "Home" => "index.md",
    "Installation" => "Installation.md",
    "Experiments" => experiments,
    "Tutorials" => tutorials,
    "How-to-guides" => how_to_guides,
    "APIs" => apis,
    "Theory & design philosophy" => discussions,
    # TODO: Move everything below here into one of the above sections
    "Developer docs" => Any[
        "CodingConventions.md",
        "AcceptableUnicode.md",
        "VariableList.md",
    ],
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
