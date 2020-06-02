using Pkg.Artifacts

using ClimateMachine.ArtifactWrappers

# Get dycoms dataset folder:
dycoms_dataset = ArtifactWrapper(
    joinpath(@__DIR__, "Artifacts.toml"),
    "dycoms",
    ArtifactFile[ArtifactFile(
        url = "https://caltech.box.com/shared/static/bxau6i46y6ikxn2sy9krgz0sw5vuptfo.nc",
        filename = "test_data_PhaseEquil.nc",
    ),],
)
dycoms_dataset_path = get_data_folder(dycoms_dataset)


@testset "Data tests" begin
    FT = Float64
    data = joinpath(dycoms_dataset_path, "test_data_PhaseEquil.nc")
    ds_PhaseEquil = Dataset(data, "r")
    e_int = Array{FT}(ds_PhaseEquil["e_int"][:])
    ρ = Array{FT}(ds_PhaseEquil["ρ"][:])
    q_tot = Array{FT}(ds_PhaseEquil["q_tot"][:])

    ts = PhaseEquil.(Ref(param_set), e_int, ρ, q_tot, 4)
    # ts = PhaseEquil.(Ref(param_set), e_int, ρ, q_tot, 3) # Fails
end
