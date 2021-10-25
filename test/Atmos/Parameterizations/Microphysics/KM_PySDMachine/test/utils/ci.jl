function is_buildkite_pipeline()
    return "BUILDKITE" in keys(ENV) && ENV["BUILDKITE"] == "true"
end

function find_path_to_climatemachine_project()
    root_folder_index = findlast("ClimateMachine.jl", pwd())

    if isnothing(root_folder_index) && is_buildkite_pipeline()
        root_folder_index = findlast(ENV["BUILDKITE_PIPELINE_SLUG"], pwd())
    end

    tmp_path = pwd()[root_folder_index[1]:end]
    n = length(splitpath(tmp_path)) - 1

    if n == 0
        path = "."
    else
        path = repeat("../", n)
    end

    path
end
