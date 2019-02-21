using Pkg
using Test

subpackags = [
              "ParametersType",
              "PlanetParameters",
              "Utilities",
             ]

for subpackage in subpackags
  Pkg.test(subpackage)
end
