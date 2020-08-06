using Pkg.Artifacts
using Test
using ClimateMachine
using ClimateMachine.ArtifactWrappers
using CLIMAParameters: AbstractEarthParameterSet
using ClimateMachine.SurfaceFluxes

using StaticArrays

using NCDatasets

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# Get dataset folder:
dataset = ArtifactWrapper(
    joinpath(@__DIR__, "Artifacts.toml"),
    "BOMEX",
    ArtifactFile[ArtifactFile(
        url = "https://caltech.box.com/shared/static/es0sqwx09yh1i5txxptvycmlhl5xpqja.nc",
        filename = "BOMEX_surface_fluxes_data_6shrs.nc",
    ),],
)
bomex_dataset_path = get_data_folder(dataset)

@testset "SurfaceFluxes - Bomex" begin
 
    FT = Float32;
    data = NCDataset(joinpath(bomex_dataset_path, "BOMEX_surface_fluxes_data_6shrs.nc"))

    # Data from 1D profiles
    qt_ave = data["qt"][:][2];
    z_ave = data["z"][:][2];
    thv_ave = data["thv"][:][2];
    u_ave = data["u"][:][2];
    x_ave = MVector(u_ave, thv_ave, 0*qt_ave);

    # Initial Guesses
    LMO_init = FT(10);
    u_star_init = FT(u_ave);
    th_star_init = FT(-100);
    qt_star_init = FT(-1e-4);
    x_init = MVector(LMO_init, u_star_init, th_star_init, qt_star_init);
    @show("InitialGuess")
    @show(x_init)

    # Surface Variables
    u_sfc = FT(0)
    thl_sfc = data["thl"][:][1];
    qt_sfc = data["qt"][:][1];
    z_sfc = data["z"][:][1];
    x_s = MVector(u_sfc, thl_sfc, 0*qt_sfc);

    # Dimensionless Scaling Factors
    dimless_num = MVector(FT(1), FT(1/3), FT(1/3));

    # Roughness Lengthscales
    z0 = MVector(FT(0.001), FT(0.001), FT(0.0001));

    # Constants
    a  = FT(4.7)
    Δz = data["z"][2];

    # F_exchange
    F_exchange = MVector(FT(-0.001), FT(-0.001), FT(-0.0001));

    result = surface_conditions(
                param_set,
                x_init,
                x_ave,
                x_s,
                z0,
                F_exchange,
                dimless_num,
                thv_ave,
                qt_ave,
                Δz,
                z_ave / 2,
                a,
                nothing
             );

    @show result
end
