using CLIMA, Documenter

makedocs(
  sitename = "CLIMA",
  doctest = false,
  strict = false,
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    # prettyurls = !("local" in ARGS),
    # canonical = "https://climate-machine.github.io/CLIMA/stable/",
  ),
  clean = false,
  modules = [Documenter, CLIMA],
  pages = Any[
    "Home" => "index.md",
    "Modules" => Any[
      "Utilities.md",
      "AtmosDycore.md",
    ],
    "Developer docs" => Any[
      "CodingConventions.md",
      "AcceptableUnicode.md",
      "VariableList.md",
    ]
  ],
)

deploydocs(
           repo = "github.com/climate-machine/CLIMA.git",
           target = "build",
          )
