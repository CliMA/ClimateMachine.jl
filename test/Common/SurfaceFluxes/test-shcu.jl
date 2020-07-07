using Pkg.Artifacts
using Test
using ClimateMachine
using ClimateMachine.ArtifactWrappers
using CLIMAParameters: AbstractEarthParameterSet
using ClimateMachine.SurfaceFluxes

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
    # Data at first interior node (x_ave)
    qt_ave = data["qt"][:][2];
    z_ave = data["z"][:][2];
    thv_ave = data["thv"][:][2];
    u_ave = data["u"][:][2];
    x_ave = [u_ave, thv_ave, qt_ave];

    # Initial guesses for MO parameters
    LMO_init = FT(100);
    u_star_init = FT(0.6);
    th_star_init = FT(290);
    qt_star_init = FT(1e-5);
    x_init = [LMO_init, u_star_init, th_star_init, qt_star_init];

    # Surface values for variables
    u_sfc = FT(0)
    thv_sfc = data["thv"][:][1];
    qt_sfc = data["qt"][:][1];
    z_sfc = data["z"][:][1];
    x_s = [u_sfc, thv_sfc, qt_sfc];

    # Dimensionless numbers
    dimless_num = [FT(1), FT(1/3), FT(1/3)];

    # Roughness
    z0 = [FT(0.001), FT(0.0001), FT(0.0001)];

    # Constants
    a  = FT(4.7)
    Δz = data["z"][2];

    # F_exchange
    F_exchange = [FT(0), FT(0), FT(0)];

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
