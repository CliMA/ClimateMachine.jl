Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[]) # JuliaLang/julia/pull/28625

using CLIMA, Documenter

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
  "InputOutput.md",
  "Developer docs" => Any[
    "CodingConventions.md",
    "AcceptableUnicode.md",
    "VariableList.md",
  ],
]

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

deploydocs(
  repo = "github.com/climate-machine/CLIMA.git",
  target = "build",
)
