using Pkg
using Test

subpackags = [
              "Grids",
              "StateVecs",
             ]

# Code coverage command line options; must correspond to src/julia.h
# and src/ui/repl.c
JL_LOG_NONE = 0

coverage = Base.JLOptions().code_coverage != JL_LOG_NONE

for subpackage in subpackags
  Pkg.test(subpackage; coverage=coverage)
end
