using CLIMA: haspkg
using MPI, Test
include("test_set.jl")
include("../test/testhelpers.jl")
testdir = dirname(@__FILE__)*"/../test"
@static if haspkg("CuArrays")
  runmpi(gpu_tests, testdir)
else
  runmpi(cpu_tests, testdir)
end
runmpi(cpu_gpu_tests, testdir)
