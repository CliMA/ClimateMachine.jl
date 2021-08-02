using Dates: print, isequal

using PyCall

using Test

function main()
    println("PySDM presence test")

    pysdm = pyimport("PySDM")

    @test pysdm isa PyObject
    @test pysdm.__name__ == "PySDM"

    module_content = py"dir($pysdm)"

    @test "Core" in module_content
    @test "physics" in module_content
    @test "builder" in module_content
    @test "initialisation" in module_content
    @test "backends" in module_content
end

main()
