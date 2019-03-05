# Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[]) # JuliaLang/julia/pull/28625

push!(LOAD_PATH, "../")
using Documenter
using CLIMA

makedocs(
  sitename = "CLIMA",
  doctest = false,
  strict = false,
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    # prettyurls = !("local" in ARGS),
    canonical = "https://climate-machine.github.io/CLIMA/stable/",
  ),
  clean = false,
  modules = [Documenter, CLIMA],
  pages = Any[
  "Home" => "index.md",
  ],
)

deploydocs(
           repo = "github.com/climate-machine/CLIMA.git",
           target = "build",
          )
