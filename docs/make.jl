rm(joinpath(@__DIR__, "Manifest.toml"), force = true)       # Remove local Manifest.toml
# rm(joinpath(@__DIR__, "..", "Manifest.toml"), force = true) # Remove local Manifest.toml

# Avoiding having to add deps to docs/ environment:
env_viz = joinpath(@__DIR__, "..", "env", "viz")
env_doc = @__DIR__

using Pkg
push!(LOAD_PATH, env_viz); Pkg.activate(env_viz); Pkg.instantiate(; verbose=true)
push!(LOAD_PATH, env_doc); Pkg.activate(env_doc); Pkg.instantiate(; verbose=true)

cd(joinpath(@__DIR__, "..")) do
    Pkg.develop(PackageSpec(path="."))
    Pkg.activate(pwd())
    Pkg.instantiate(; verbose=true)
end

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
    prettyurls = get(ENV, "CI", "") == "true",
    mathengine = mathengine,
    collapselevel = 1,
)

makedocs(
    sitename = "ClimateMachine",
    doctest = false,
    strict = false,
    linkcheck = true,
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
)
