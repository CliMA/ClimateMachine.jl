using Pkg
using Test

subpackags = [
              "CLIMAAtmosDycore",
              "ParametersType",
              "PlanetParameters",
              "Utilities",
             ]

for subpackage in subpackags
  Pkg.test(subpackage)
end
