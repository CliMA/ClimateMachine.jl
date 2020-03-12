using Test, Pkg

ENV["DATADEPS_ALWAYS_ACCEPT"] = true
ENV["JULIA_LOG_LEVEL"] = "WARN"

abstract type TestIntensity end
struct NormalIntensity <: TestIntensity end
struct LowIntensity <: TestIntensity end

# ENV["intensity"] = "low"

intensity = get(ENV, "intensity", "normal")=="low" ? LowIntensity() : NormalIntensity()

test(::NormalIntensity) = true
test(::LowIntensity) = false

function test_intensity(;low::T=nothing,normal::T=nothing) where {T}
  @assert low ≠ nothing
  @assert normal ≠ nothing
  if get(ENV, "intensity", "normal")=="low"
    return low
  else
    return normal
  end
end

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
