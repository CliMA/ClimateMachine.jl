# generate examples
import Literate

EXAMPLE_BL_DIR = joinpath(@__DIR__, "..", "src", "DGmethods", "examples")
GENERATED_BL_DIR = joinpath(@__DIR__, "src", "examples", "DGmethods", "generated")
for example in readdir(EXAMPLE_BL_DIR)
    endswith(example, ".jl") || continue
    input = abspath(joinpath(EXAMPLE_BL_DIR, example))
    script = Literate.script(input, GENERATED_BL_DIR)
    code = strip(read(script, String))
    mdpost(str) = replace(str, "@__CODE__" => code)
    Literate.markdown(input, GENERATED_BL_DIR, postprocess = mdpost)
    Literate.notebook(input, GENERATED_BL_DIR, execute = true)
end

# remove any .vtu files in the generated dir (should not be deployed)
cd(GENERATED_BL_DIR) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
end
