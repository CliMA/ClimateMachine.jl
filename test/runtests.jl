using Test, Pkg

ENV["DATADEPS_ALWAYS_ACCEPT"] = true
ENV["JULIA_LOG_LEVEL"] = "WARN"

function include_test(_module)
  println("Starting tests for $_module")
  t = @elapsed include(joinpath(_module,"runtests.jl"))
  println("Completed tests for $_module, $(round(Int, t)) seconds elapsed")
  return nothing
end


@testset "CLIMA" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false

    for submodule in [
                      "Utilities",
                      "Common",
                      "Atmos",
                      "Mesh",
                      "DGmethods",
                      "Diagnostics",
                      "ODESolvers",
                      "Ocean",
                      "Arrays",
                      "LinearSolvers",
                      "Driver",
                     ]

      if all_tests || "$submodule" in ARGS || "CLIMA" in ARGS
        include_test(submodule)
      end
    end

end
