using ClimateMachine
ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()

using Test
using NonlinearSolvers
using ClimateMachine.SurfaceFluxes
const SF = SurfaceFluxes
using StaticArrays

using CLIMAParameters: AbstractEarthParameterSet
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

@testset "SurfaceFluxes - FMS Profiles" begin
    FT = Float32

    ## Discretisation altitude z
    z = ArrayType(FT[
        29.432779269303,
        30.0497139076724,
        31.6880000418153,
        34.1873479240475,
    ])
    ## Virtual Dry Static Energy at height z
    pt = ArrayType(FT[
        268.559120403867,
        269.799228886728,
        277.443023238556,
        295.79192777341,
    ])
    ## Surface Pottemp
    pt0 = ArrayType(FT[
        273.42369841804,
        272.551410044203,
        278.638168565727,
        298.133068766049,
    ])
    ## Roughness lengths
    z0 = ArrayType(FT[
        5.86144925739178e-05,
        0.0001,
        0.000641655193293549,
        3.23383768877187e-05,
    ])
    zt = ArrayType(FT[
        3.69403636275411e-05,
        0.0001,
        1.01735489109205e-05,
        7.63933834969505e-05,
    ])
    zq = ArrayType(FT[
        5.72575636226887e-05,
        0.0001,
        5.72575636226887e-05,
        5.72575636226887e-05,
    ])
    ## Speed
    speed = ArrayType(FT[
        2.9693638452068,
        2.43308757772094,
        5.69418282305367,
        9.5608693754561,
    ])
    ## Scale velocity and moisture
    u_star = ArrayType(FT[
        0.109462510724615,
        0.0932942802513508,
        0.223232887323184,
        0.290918439028557,
    ])
    q_star = ArrayType(FT[
        0.000110861442197537,
        9.44983279664197e-05,
        4.17643828631936e-05,
        0.000133135421415819,
    ])
    # No explicit buoyancy terms in ClimateMachine
    b_star = ArrayType([
        0.00690834676781433,
        0.00428178089592372,
        0.00121229800895103,
        0.00262353784027441,
        -0.000570314880866852,
    ])

    for ii in 1:length(u_star)
        # Data at first interior node (x_ave)
        qt_ave = FT(0)
        z_ave = Tuple(z)[ii]
        vdse_ave = Tuple(pt)[ii]
        u_ave = Tuple(speed)[ii]
        x_ave = ArrayType(FT[u_ave, vdse_ave, qt_ave])

        ## Initial guesses for MO parameters
        LMO_init = eps(FT)
        u_star_init = FT(0.1)
        th_star_init = -FT(0.1)
        qt_star_init = -FT(1e-5)
        x_init =
            ArrayType(FT[LMO_init, u_star_init, th_star_init, qt_star_init])

        # Surface values for variables
        u_sfc = FT(0)
        thv_sfc = Tuple(pt0)[ii]
        qt_sfc = Tuple(q_star)[ii]
        z_sfc = FT(0)
        x_s = ArrayType(FT[u_sfc, thv_sfc, qt_sfc])

        # Roughness
        z_rough = ArrayType(FT[Tuple(z0)[ii], Tuple(zt)[ii], Tuple(zq)[ii]])

        # Constants
        a = FT(4.7)
        Δz = Tuple(z)[ii]

        # F_exchange
        F_exchange = ArrayType(FT[0.01, -0.01, -0.000001])

        args = (
            param_set,
            x_init,
            x_ave,
            x_s,
            z_rough,
            F_exchange,
            vdse_ave,
            qt_ave,
            Δz,
            z_ave / 2,
            a,
        )

        ## Assuming surface fluxes are not given
        result = surface_conditions(args..., SF.DGScheme())

        x_star = Array(result.x_star)
        u_star_arr = Array(u_star)
        @test (abs((x_star[1] - u_star_arr[ii]) / u_star_arr[ii])) <= FT(0.15)

        # Keeping FV-based scheme in case needed
        result = surface_conditions(args..., SF.FVScheme())
    end
end

include("test_universal_functions.jl")
