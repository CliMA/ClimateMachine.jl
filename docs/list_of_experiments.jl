
experiments = Any[]
experiments_dir = "experiments"
experiments_doc_dir = joinpath("generated", "experiments")

for subdir in readdir(joinpath(@__DIR__, "..", experiments_dir))
    experiments_sub = Any[]
    mkpath(joinpath(@__DIR__, "src", experiments_doc_dir, subdir))
    for experiment in readdir(joinpath(@__DIR__, "..", experiments_dir, subdir))
        endswith(experiment, ".jl") || continue

        experiment_file = joinpath(experiments_dir, subdir, experiment)
        experiment_doc_file = joinpath(
            experiments_doc_dir,
            subdir,
            replace(experiment, ".jl" => ".md"),
        )

        helpstring = cd(joinpath(@__DIR__, "..")) do # run from top-level so we get nice paths
            read(`$(Base.julia_cmd()) --project=$(Base.active_project()) $experiment_file --help`)
        end

        open(joinpath(@__DIR__, "src", experiment_doc_file), write = true) do f
            println(f, "# `$experiment`")
            println(f, "")
            println(f, "```")
            write(f, helpstring)
            println(f, "```")
        end
        push!(experiments_sub, experiment => experiment_doc_file)
    end
    push!(experiments, subdir => experiments_sub)
end
