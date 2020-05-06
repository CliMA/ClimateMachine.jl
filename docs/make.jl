Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[]) # JuliaLang/julia/pull/28625

using CLIMA, Documenter, Literate

generated_dir = joinpath(@__DIR__, "src", "generated") # generated files directory
mkpath(generated_dir)

include("list_of_experiments.jl")        # defines a dict `experiments`
include("list_of_tutorials.jl")          # defines a dict `tutorials`
include("list_of_extending_clima.jl")    # defines a dict `extending_clima`
include("list_of_discussions.jl")        # defines a dict `apis`
include("list_of_apis.jl")               # defines a dict `discussions`

pages = Any[
    "Home" => "index.md",
    "Installation" => "Installation.md",
    "Experiments" => experiments,
    "Tutorials" => tutorials,
    "Extending CLIMA" => extending_clima,
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
    # canonical = "https://CliMA.github.io/CLIMA/stable/",
)

makedocs(
    sitename = "CLIMA",
    doctest = false,
    strict = false,
    linkcheck = false,
    format = format,
    checkdocs = :exports,
    # checkdocs = :all,
    clean = true,
    modules = [Documenter, CLIMA],
    pages = pages,
)

include("clean_build_folder.jl")

deploydocs(
    repo = "github.com/CliMA/CLIMA.git",
    target = "build",
    push_preview = true,
)
