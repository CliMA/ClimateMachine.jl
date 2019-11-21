Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[]) # JuliaLang/julia/pull/28625

using CLIMA, Documenter

include("generate.jl")

GENERATED_BL_EXAMPLES =
[joinpath("examples", "DGmethods_old", "generated", f) for f in
 (
  "ex_001_periodic_advection.md",
  "ex_002_solid_body_rotation.md",
 )]

makedocs(
  sitename = "CLIMA",
  doctest = false,
  strict = false,
  linkcheck = false,
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    # prettyurls = !("local" in ARGS),
    # canonical = "https://climate-machine.github.io/CLIMA/stable/",
  ),
  clean = false,
  modules = [Documenter, CLIMA],
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
    "Balance Law Examples" => ["BalanceLawOverview.md",
                               GENERATED_BL_EXAMPLES...]
  ],
)

# make sure there are no *.vtu files left around from the build
cd(joinpath(@__DIR__, "build", "examples", "DGmethods_old", "generated")) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
end

deploydocs(
  repo = "github.com/climate-machine/CLIMA.git",
  target = "build",
)
