Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[]) # JuliaLang/julia/pull/28625

using CLIMA, Documenter

# TODO: Add generated examples back
# include("generate.jl")

GENERATED_BL_EXAMPLES =
[joinpath("examples", "DGmethods_old", "generated", f) for f in
 (
  "ex_001_periodic_advection.md",
  "ex_002_solid_body_rotation.md",
 )]
GENERATED_BL_EXAMPLES = filter!(x->isfile(x), GENERATED_BL_EXAMPLES)

pages = Any[
  "Home" => "index.md",
  "Common" => Any[
    "MoistThermodynamics" => "Common/MoistThermodynamics.md",
  ],
  "Utilites" => Any[
    "RootSolvers" => "Utilities/RootSolvers.md",
  ],
  "Atmos" => Any[
    "Atmos/SurfaceFluxes.md",
    "Atmos/TurbulenceConvection.md",
    "Atmos/EDMFEquations.md",
    "Microphysics" => "Atmos/Microphysics.md",
  ],
  "ODESolvers" => "ODESolvers.md",
  "LinearSolvers" => "LinearSolvers.md",
  "Mesh" => "Mesh.md",
  "Arrays" => "Arrays.md",
  "DGmethods_old" => "DGmethods_old.md",
  "InputOutput.md",
  "Developer docs" => Any[
    "CodingConventions.md",
    "AcceptableUnicode.md",
    "VariableList.md",
  ],
]

if !isempty(GENERATED_BL_EXAMPLES)
  push!(pages,"Balance Law Examples" => ["BalanceLawOverview.md", GENERATED_BL_EXAMPLES...])
end


makedocs(
  sitename = "CLIMA",
  doctest = false,
  strict = false,
  linkcheck = false,
  checkdocs = :exports,
  # checkdocs = :all,
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = MathJax(Dict(
            :TeX => Dict(
                :equationNumbers => Dict(:autoNumber => "AMS"),
                :Macros => Dict()
            )
        ))
    # prettyurls = !("local" in ARGS),
    # canonical = "https://climate-machine.github.io/CLIMA/stable/",
  ),
  clean = true,
  modules = [Documenter, CLIMA],
  pages = pages,
)

# make sure there are no *.vtu files left around from the build
p = joinpath(@__DIR__, "build", "examples", "DGmethods_old", "generated")
if ispath(p)
  cd(p) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
  end
end

deploydocs(
  repo = "github.com/climate-machine/CLIMA.git",
  target = "build",
)
