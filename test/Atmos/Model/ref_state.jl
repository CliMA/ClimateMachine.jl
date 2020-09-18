include("get_atmos_ref_states.jl")
using JLD2
using Pkg.Artifacts
using ClimateMachine.ArtifactWrappers

@testset "Hydrostatic reference states - regression test" begin
    ref_state_dataset = ArtifactWrapper(
        joinpath(@__DIR__, "Artifacts.toml"),
        "ref_state",
        ArtifactFile[ArtifactFile(
            url = "https://caltech.box.com/shared/static/gyq292ns79wm9xpmy1sse3qtnpcxw54q.jld2",
            filename = "ref_state.jld2",
        ),],
    )
    ref_state_dataset_path = get_data_folder(ref_state_dataset)
    data_file = joinpath(ref_state_dataset_path, "ref_state.jld2")

    RH = 0.5
    (nelem_vert, N_poly) = (20, 4)
    solver_config = get_atmos_ref_states(nelem_vert, N_poly, RH)
    all_data = dict_of_nodal_states(solver_config, ["z"], (Auxiliary(),))
    T = all_data["ref_state.T"]
    p = all_data["ref_state.p"]
    ρ = all_data["ref_state.ρ"]

    @load "$data_file" T_ref p_ref ρ_ref
    @test all(T .≈ T_ref)
    @test all(p .≈ p_ref)
    @test all(ρ .≈ ρ_ref)
end

@testset "Hydrostatic reference states - correctness" begin

    RH = 0.5
    # Fails on (80, 1)
    for (nelem_vert, N_poly) in [(40, 2), (20, 4)]
        solver_config = get_atmos_ref_states(nelem_vert, N_poly, RH)
        all_data = dict_of_nodal_states(solver_config, ["z"])
        phase_type = PhaseEquil
        T = all_data["ref_state.T"]
        p = all_data["ref_state.p"]
        ρ = all_data["ref_state.ρ"]
        q_tot = all_data["ref_state.ρq_tot"] ./ ρ
        q_pt = PhasePartition.(q_tot)

        # TODO: test that ρ and p are in discrete hydrostatic balance

        # Test state for thermodynamic consistency (with ideal gas law)
        T_igl = air_temperature_from_ideal_gas_law.(Ref(param_set), p, ρ, q_pt)
        @test all(T .≈ T_igl)

        # Test that relative humidity in reference state is approximately
        # input relative humidity
        RH_ref = relative_humidity.(Ref(param_set), T, p, Ref(phase_type), q_pt)
        @show max(abs.(RH .- RH_ref)...)
        @test all(isapprox.(RH, RH_ref, atol = 0.05))
    end

end
