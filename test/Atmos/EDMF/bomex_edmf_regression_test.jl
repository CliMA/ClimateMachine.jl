using FileIO
using JLD2
using Pkg.Artifacts
using ClimateMachine.ArtifactWrappers

# Get bomex_edmf dataset folder:
bomex_edmf_dataset = ArtifactWrapper(
    joinpath(@__DIR__, "Artifacts.toml"),
    "bomex_edmf",
    ArtifactFile[ArtifactFile(
        url = "https://caltech.box.com/shared/static/jbhcy6ncc5wh1hg9hcea5f45w6t22kk2.jld2",
        filename = "bomex_edmf.jld2",
    ),],
)
bomex_edmf_dataset_path = get_data_folder(bomex_edmf_dataset)
data_file = joinpath(bomex_edmf_dataset_path, "bomex_edmf.jld2")

updraft_vars(N_up, st, tol, sym...) = Dict(ntuple(N_up) do i
    "turbconv.updraft[$i]." * join(string.(sym), ".") => (st, tol)
end)

vars_to_compare(N_up) = Dict(
    "ρ" => (Prognostic(), 1),
    "ρu[1]" => (Prognostic(), 1),
    "ρu[2]" => (Prognostic(), 1),
    "ρu[3]" => (Prognostic(), 1),
    "ρe" => (Prognostic(), 1e5),
    "moisture.ρq_tot" => (Prognostic(), 1e-2),
    "turbconv.environment.ρatke" => (Prognostic(), 5e-1),
    "turbconv.environment.T" => (Auxiliary(), 300),
    "turbconv.environment.cld_frac" => (Auxiliary(), 1),
    "turbconv.environment.buoyancy" => (Auxiliary(), 1e-2),
    updraft_vars(N_up, Prognostic(), 1 * 0.1, :ρa)...,
    updraft_vars(N_up, Prognostic(), 1 * 0.1, :ρaw)...,
    updraft_vars(N_up, Prognostic(), 300 * 0.1, :ρaθ_liq)...,
    updraft_vars(N_up, Prognostic(), 1e-2 * 0.1, :ρaq_tot)...,
    updraft_vars(N_up, Auxiliary(), 1e-2, :buoyancy)...,
)


compare = Dict()
@testset "Regression Test" begin
    N_up = n_updrafts(solver_config.dg.balance_law.turbconv)
    numerical_data =
        dict_of_nodal_states(solver_config, ["z"], (Prognostic(), Auxiliary()))
    data_to_compare = Dict()
    for (ftc, v) in vars_to_compare(N_up)
        data_to_compare[ftc] = numerical_data[ftc]
    end

    export_new_solution_jld2 = false
    if export_new_solution_jld2
        save("bomex_edmf.jld2", data_to_compare)
    end

    all_data_ref = load(data_file)

    @test all(k in keys(all_data_ref) for k in keys(data_to_compare))
    N_up = n_updrafts(solver_config.dg.balance_law.turbconv)
    comparison_vars = vars_to_compare(N_up)
    for k in keys(all_data_ref)
        data = data_to_compare[k]
        ref_data = all_data_ref[k]
        tol = comparison_vars[k][2]
        s = length(data) / 100

        @test !any(isnan.(ref_data))
        @test !any(isnan.(data))
        absΔdata = abs.(data .- ref_data)
        T1 = isapprox(norm(absΔdata), 0, atol = tol * 0.01 * s) # norm
        T2 = isapprox(maximum(absΔdata), 0, atol = tol * 0.01) # max of local
        compare[k] = (norm(absΔdata), maximum(absΔdata), tol)
        (!T1 || !T2) && @show k, norm(absΔdata), maximum(absΔdata), tol
        @test isapprox(norm(absΔdata), 0, atol = tol * 0.01 * s) # norm
        @test isapprox(maximum(absΔdata), 0, atol = tol * 0.01) # max of local
    end
end
