module TestSurfaceFluxes

using Random
using ClimateMachine
ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()

using Test
using NonlinearSolvers
using ClimateMachine.SurfaceFluxes
const SF = SurfaceFluxes
using StaticArrays
import ClimateMachine.MPIStateArrays: array_device
import KernelAbstractions: CPU

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
        MO_param_guess =
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
        Δz = Tuple(z)[ii]

        args = (
            param_set,
            MO_param_guess,
            x_ave,
            x_s,
            z_rough,
            vdse_ave,
            z_ave / 2,
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

# Shuffing disabled in GPU, Tuple for scalar indexing
cpu_shuffle(array::ArrayType) =
    array_device(array) isa CPU ? shuffle(array) : Tuple(array)

@testset "SurfaceFluxes - Recovery" begin
    FT = Float32

    # Stochastic sampling of parameter space
    for j in 1:20
        # Define MO parameters to be recovered
        u_star = cpu_shuffle(ArrayType(FT[0.05, 0.1, 0.2, 0.3, 0.4]))
        θ_star = cpu_shuffle(ArrayType(FT[0, 50, 100, 200, 300])) # Functions not robust to θ_star < 0

        # Define temperature parameters
        θ_scale = cpu_shuffle(ArrayType(FT[290, 280, 270, 260, 250]))
        θ_s = cpu_shuffle(ArrayType(FT[290, 280, 270, 260, 250]))

        # Define a set of heights
        z = cpu_shuffle(ArrayType(FT[5, 10, 15, 20, 50]))
        z_rough = cpu_shuffle(ArrayType(FT[0.001, 0.01, 0.1, 0.5, 1])) # Must be smaller than z

        # Relative initialization of MO parameters
        LMO_init = cpu_shuffle(ArrayType(FT[-1.0, 0.2, 0.5, 1.0, 2.0]))
        u_star_init = cpu_shuffle(ArrayType(FT[0.2, 0.5, 1.0, 1.5, 2.0])) # Must be positive
        θ_star_init = cpu_shuffle(ArrayType(FT[-1e-6, 1e-6, -1e-6, 1e-6, 1e-6]))

        for ii in 1:length(u_star)
            x_star_given = Tuple(ArrayType(FT[u_star[ii], θ_star[ii]]))
            L = monin_obukhov_length(
                param_set,
                u_star[ii],
                θ_scale[ii],
                -u_star[ii] * θ_star[ii],
            )
            x_s = ArrayType(FT[FT(0), θ_s[ii]])

            u_in = recover_profile(
                param_set,
                z[ii],
                u_star[ii],
                Tuple(x_s)[1],
                z_rough[ii],
                L,
                SF.MomentumTransport(),
                SF.DGScheme(),
            )
            θ_in = recover_profile(
                param_set,
                z[ii],
                θ_star[ii],
                Tuple(x_s)[2],
                z_rough[ii],
                L,
                SF.HeatTransport(),
                SF.DGScheme(),
            )

            x_in = ArrayType(FT[u_in, θ_in])

            MO_param_guess = ArrayType(FT[
                LMO_init[ii] * L,
                u_star_init[ii] * u_star[ii],
                θ_star_init[ii] * θ_star[ii],
            ])
            recovered = surface_conditions(
                param_set,
                MO_param_guess,
                x_in,
                x_s,
                z_rough[ii],
                θ_scale[ii],
                z[ii],
                SF.DGScheme(),
            )

            # Test is recovery under 5% error for all variables
            x_star_recovered = Tuple(recovered.x_star)
            @test abs(x_star_recovered[1] - x_star_given[1]) /
                  (abs(x_star_given[1]) + FT(1e-3)) < FT(0.05)
            @test abs(x_star_recovered[2] - x_star_given[2]) /
                  (abs(x_star_given[2]) + FT(1e-3)) < FT(0.05)
        end
    end
end

include("test_universal_functions.jl")

end # module TestSurfaceFluxes
