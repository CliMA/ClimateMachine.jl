using PyCall
using Test

pysdm = pyimport("PySDM")

@test pysdm isa PyObject
@test pysdm.__name__ == "PySDM"

module_content = py"dir($pysdm)"

@test "physics" in module_content &&
      "builder" in module_content &&
      "initialisation" in module_content &&
      "backends" in module_content
