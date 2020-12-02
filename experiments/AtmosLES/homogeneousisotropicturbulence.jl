using Random
using FFTW
using StaticArrays
using NCDatasets
using Test
using DocStringExtensions
using LinearAlgebra
using DelimitedFiles
using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.ArtifactWrappers
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using CLIMAParameters
using CLIMAParameters.Planet: R_d
using CLIMAParameters.Atmos.Microphysics
using Interpolations
struct LiquidParameterSet <: AbstractLiquidParameterSet end
struct IceParameterSet <: AbstractIceParameterSet end
struct RainParameterSet <: AbstractRainParameterSet end
struct SnowParameterSet <: AbstractSnowParameterSet end
struct MicropysicsParameterSet{L, I, R, S} <: AbstractMicrophysicsParameterSet
    liq::L
    ice::I
    rain::R
    snow::S
end
struct EarthParameterSet{M} <: AbstractEarthParameterSet
    microphys::M
end

const microphys = MicropysicsParameterSet(
    LiquidParameterSet(),
    IceParameterSet(),
    RainParameterSet(),
    SnowParameterSet(),
)
const param_set = EarthParameterSet(microphys)

using Dierckx
ClimateMachine.init(parse_clargs = true)

function init_HIT!(problem, bl, state, aux, localgeo, t, args...)
    FT = eltype(state)
    spl_pinit, spl_uinit, spl_vinit, spl_winit = args[1]
    # interpolate data
    (x, y, z) = localgeo.coord
    u = FT(spl_uinit(x,y,z))
    v = FT(spl_vinit(x,y,z))
    w = FT(spl_winit(x,y,z))
    p = FT(spl_pinit(x,y,z)) + 101325
    R_gas::FT = R_d(bl.param_set)
    ρ = FT(1.178)
    e_kin = 0.5 * (u^2 + v^2 + w^2)
    T = p / (ρ * R_gas)
    e_pot = FT(0)
    e_int = internal_energy(bl.param_set, T, PhasePartition(FT(0)))
    E = ρ * (e_kin + e_pot + e_int)
    state.ρ = ρ
    state.ρu = SVector(ρ * u, ρ * v, ρ * w)
    state.ρe = E
    @info x,y,z,u,v,w,T,p
    return nothing
end

function spline_int()
       a = zeros(7,1)
       a[1] = 5.6102
       a[2] = -1.1236 
       a[3] = -0.30961
       a[4] = 0.33172
       a[5] = -0.10959
       a[6] = -0.22320e-1
       a[7] = 0.66575e-2
       l = 55.0
       u_0 = 27.1893
       const1 = 2 * pi / l
       const2 = const1 / (u_0^2)
       Nx = 32
       Ny = 32
       Nz = 32
       dx = 16
       dy = 16
       dz = 16
       uhat = complex(zeros(Int64(Nx/2+1),Ny,Nz))
       vhat = complex(zeros(Int64(Nx/2+1),Ny,Nz))
       what = complex(zeros(Int64(Nx/2+1),Ny,Nz))
       dphat = complex(zeros(Int64(dx/2+1),dy,dz))
       phat = complex(zeros(Int64(Nx/2+1),Ny,Nz))
       for kx = 0:Nx/2
         for ky = -Ny/2:Ny/2 - 1
           for kz = -Nz/2: Nz/2 - 1
             ix = Int64(kx + 1)
             iy = Int64(ky + Ny/2 + 1)
             iz = Int64(kz + Nz/2 + 1)
             ksq = kx^2 + ky^2 + kz^2
             kwav = sqrt(ksq)
             k12 = sqrt(kx^2 + ky^2)
             if (ksq > 0)
               sum = 0.0
               for i=0:6
                 sum = sum + a[i+1] * log(const1 * kwav)^i
               end
               espec = const2 * exp(sum)
               theta1 = rand(Float64) * 2 * pi
               theta2 = rand(Float64) * 2 * pi
               phi = rand(Float64) * 2 * pi
               alp = sqrt(espec / (2 * pi) / kwav / kwav) * exp(complex(0,1) * theta1) * cos(phi)
               beta = sqrt(espec / (2 * pi) / kwav / kwav) * exp(complex(0,1) * theta2) * sin(phi)
             end
             if (ksq > 2 * Nx^2 / 9 )
               uhat[ix,iy,iz] = complex(0,0)
               vhat[ix,iy,iz] = complex(0,0)
               what[ix,iy,iz] = complex(0,0)
             elseif ((kx != 0) || (ky != 0))
               uhat[ix,iy,iz] = (alp * kwav * ky + beta * kx * kz) / kwav / k12
               vhat[ix,iy,iz] = (beta * ky * kz  - alp * kwav * kx) / kwav / k12
               what[ix,iy,iz] = - beta * k12 / kwav
             elseif (kz != 0)
               uhat[ix,iy,iz] = alp
               vhat[ix,iy,iz] = beta 
               what[ix,iy,iz] = complex(0,0)
             else
               uhat[ix,iy,iz] = complex(0,0)
               vhat[ix,iy,iz] = complex(0,0)
               what[ix,iy,iz] = complex(0,0)
             end
           end
         end
       end
       for ky = -Ny/2 + 1: Ny/2 - 1 
         if (ky != 0)
           for kz = -Nz/2 + 1: -1
             iy = Int64(-ky + Ny/2)
             iz = Int64(-kz + Nz/2)
             iy2 = Int64(ky + Ny/2 + 1)
             iz2 = Int64(kz + Nz/2 + 1)
             uhat[1, iy, iz] = conj(uhat[1,iy2,iz2])
             vhat[1, iy, iz] = conj(vhat[1,iy2,iz2])
             what[1, iy, iz] = conj(what[1,iy2,iz2])
           end
         end
       end
       zc = Int64(Nz/2) + 1
       yc = Int64(Ny/2) + 1
       for ky = 1:Ny/2 - 1
         iy = Int64(-ky + Ny/2)
         iy2 = Int64(ky + Ny/2 + 1)
         uhat[1, iy2, zc] = conj(uhat[1,iy,zc])
         vhat[1, iy2, zc] = conj(vhat[1,iy,zc])
         what[1, iy2, zc] = conj(what[1,iy,zc])
       end
       for kz = 1:Nz/2 - 1
         iz = Int64(-kz + Nz/2)
         iz2 = Int64(kz + Nz/2 + 1)
         uhat[1, yc, iz2] = conj(uhat[1,yc,iz])
         vhat[1, yc, iz2] = conj(vhat[1,yc,iz])
         what[1, yc, iz2] = conj(what[1,yc,iz])
       end
       for ky = 1:Ny/2 - 1
         iy = Int64(-ky + Ny/2 + 1)
         iy2 = Int64(ky + Ny/2 + 1)
         uhat[1, iy2, 1] = conj(uhat[1,iy,1])
         vhat[1, iy2, 1] = conj(vhat[1,iy,1])
         what[1, iy2, 1] = conj(what[1,iy,1])
       end
       for kz = 1:Nz/2 - 1
         iz = Int64(-kz + Nz/2 + 1)
         iz2 = Int64(kz + Nz/2 + 1)
         uhat[1, 1, iz2] = conj(uhat[1,1,iz])
         vhat[1, 1, iz2] = conj(vhat[1,1,iz])
         what[1, 1, iz2] = conj(what[1,1,iz])
       end
       uhat[1,yc,zc] = complex(0,0)
       uhat[1,yc,1] = complex(0,0)
       uhat[1,1,zc] = complex(0,0)
       uhat[1,1,1] = complex(0,0)
       vhat[1,yc,zc] = complex(0,0)
       vhat[1,yc,1] = complex(0,0)
       vhat[1,1,zc] = complex(0,0)
       vhat[1,1,1] = complex(0,0)
       what[1,yc,zc] = complex(0,0)
       what[1,yc,1] = complex(0,0)
       what[1,1,zc] = complex(0,0)
       what[1,1,1] = complex(0,0)
       u = irfft(uhat,32)
       v = irfft(vhat,32)
       w = irfft(what,32)
       k_max = 6
       csq = 0.01
       const3 = 32.0 / 3.0 / k_max^5 * sqrt(2.0 / pi) * csq
       for kx = 0:dx/2
         for ky = -dy/2:dy/2 - 1
           for kz = -dz/2: dz/2 - 1
             ix = Int64(kx + 1)
             iy = Int64(ky + dy/2 + 1)
             iz = Int64(kz + dz/2 + 1)
             k2 = kx^2 + ky^2 + kz^2
             espec = const3 * k2 * exp(-2 * k2 / k_max^2) / (4 * pi)
             pamp = sqrt(espec)
             phi = 2 * pi * rand(Float64)
             if (k2 == 0)
               dphat[ix,iy,iz] = complex(0,0)
             else
               dphat[ix,iy,iz] = pamp * (cos(phi) + complex(0,1) * sin(phi))
             end
           end
         end
       end
       for ky = -dy/2 + 1: dy/2 - 1
         if (ky != 0)
           for kz = -dz/2 + 1: dz/2 - 1
             iy = Int64(-ky + dy/2)
             iz = Int64(-kz + dz/2)
             iy2 = Int64(ky + dy/2 + 1)
             iz2 = Int64(kz + dz/2 + 1)
             dphat[1,iy,iz] = conj(dphat[1,iy2,iz2])
           end
         end
       end
       yc = Int64(dy/2) + 1
       zc = Int64(dz/2) + 1
       for ky = 1: dy/2 - 1
         iy = Int64(-ky + dy/2)
         iy2 = Int64(ky + dy/2 + 1)
         dphat[1,iy2,zc] = conj(dphat[1,iy,zc])
       end
       for kz = 1:dz/2 - 1
         iz = Int64(-kz + dz/2)
         iz2 = Int64(kz + dz/2 + 1)
         dphat[1,yc,iz2] = conj(dphat[1,yc,iz])
       end
       for ky = 1: dy/2 - 1
         iy = Int64(-ky + dy/2)
         iy2 = Int64(ky + dy/2 + 1)
         dphat[1,iy2,1] = conj(dphat[1,iy,1])
       end
       for kz = 1:dz/2 - 1
         iz = Int64(-kz + dz/2)
         iz2 = Int64(kz + dz/2 + 1)
         dphat[1,1,iz2] = conj(dphat[1,1,iz])
       end
       dphat[1,yc,zc] = complex(0,0)
       dphat[1,yc,1] = complex(0,0)
       dphat[1,1,zc] = complex(0,0)
       dphat[1,1,1] = complex(0,0)
       for kx = 0:Nx/2
         for ky = -Ny/2:Ny/2 - 1
           for kz = -Nz/2: Nz/2 - 1
             ix = Int64(kx + 1)
             iy = Int64(ky + Ny/2 + 1)
             iz = Int64(kz + Nz/2 + 1)
             phat[ix,iy,iz] = 0.0
           end
         end
       end 
       for kx = 0:dx/2
         for ky = -dy/2:dy/2 - 1
           for kz = -dz/2: dz/2 - 1
             ix = Int64(kx + 1)
             iy = Int64(ky + dy/2 + 1)
             iz = Int64(kz + dz/2 + 1)
             phat[ix,iy,iz] = dphat[ix,iy,iz]
           end
         end
       end
       p = irfft(phat,32)
       X = -pi:pi/16:pi-pi/16
       Y = -pi:pi/16:pi-pi/16
       Z = -pi:pi/16:pi-pi/16
       @info size(X), size(u), size(Y)
       knots = (X,Y,Z)
       itp = interpolate(knots, p, Gridded(Linear()))
       etp = extrapolate(itp, Periodic())
       p_spl = etp
       itp = interpolate(knots, u, Gridded(Linear()))
       etp = extrapolate(itp, Periodic())
       u_spl = etp
       itp = interpolate(knots, v, Gridded(Linear()))
       etp = extrapolate(itp, Periodic())
       v_spl = etp
       itp = interpolate(knots, w, Gridded(Linear()))
       etp = extrapolate(itp, Periodic())
       w_spl = etp
       return p_spl, u_spl, v_spl, w_spl
end

function config_HIT(FT, N, resolution, xmax, ymax, zmax, xmin, ymin, zmin)
    # Reference state
    # Boundary conditions
    # SGS Filter constants
    C_smag = FT(0.18) # 0.21 for stable testing, 0.18 in practice
    ics = init_HIT!

    source = ()


    problem = AtmosProblem(
        boundaryconditions = (AtmosBC(), AtmosBC()),
        init_state_prognostic = ics,
    )


    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
	orientation = NoOrientation(),
        problem = problem,
        ref_state = NoReferenceState(),
        moisture = DryModel(),
        turbulence = SmagorinskyLilly{FT}(C_smag),#ConstantViscosityWithDivergence{FT}(200),
        source = source,
    )

    ode_solver = ClimateMachine.ExplicitSolverType(solver_method = LSRK144NiegemannDiehlBusch,)

    config = ClimateMachine.AtmosLESConfiguration(
        "Homogeneous Isotropic Turbulence",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_HIT!,
        xmin = xmin,
        ymin = ymin,
	zmin = zmin,
        solver_type = ode_solver,
        model = model,
        periodicity = (true, true, true),
        #numerical_flux_first_order = RoeNumericalFlux(),
    )
    return config
end
function config_diagnostics(
    driver_config,
    (xmin, xmax, ymin, ymax, zmin, zmax),
    resolution,
    tnor,
    titer,
    snor,
)
    ts_dgngrp = setup_atmos_turbulence_stats(
        AtmosLESConfigType(),
        "360steps",
        driver_config.name,
        tnor,
        titer,
    )

    boundaries = [
        xmin ymin zmin
        xmax ymax zmax
    ]
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )
    ds_dgngrp = setup_atmos_spectra_diagnostics(
        AtmosLESConfigType(),
        "0.06ssecs",
        driver_config.name,
        interpol = interpol,
        snor,
    )
    me_dgngrp = setup_atmos_mass_energy_loss(
        AtmosLESConfigType(),
        "0.02ssecs",
        driver_config.name,
    )
    return ClimateMachine.DiagnosticsConfiguration([
        ts_dgngrp,
        ds_dgngrp,
        me_dgngrp,
    ],)

end

function main()
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Ncellsx = 32
    Ncellsy = 32
    Ncellsz = 2
    Δx = FT(2 * pi / Ncellsx)
    Δy = Δx
    Δz = Δx
    resolution = (Δx, Δy, Δz)
    xmin = FT(-pi)
    xmax = FT(pi)
    ymin = FT(-pi)
    ymax = FT(pi)
    zmin = FT(-pi)
    zmax = FT(pi)
    t0 = FT(0)
    timeend = FT(10)
    spl_pinit, spl_uinit, spl_vinit, spl_winit = spline_int()
    Cmax = FT(0.4)

    driver_config =
        config_HIT(FT, N, resolution, xmax, ymax, zmax, xmin, ymin, zmin)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        (spl_pinit, spl_uinit, spl_vinit, spl_winit);
        init_on_cpu = true,
        Courant_number = Cmax,
    )
    tnor = FT(100)
    titer = FT(0.01)
    snor = FT(10000.0)
    dgn_config = config_diagnostics(
        driver_config,
        (xmin, xmax, ymin, ymax, zmin, zmax),
        resolution,
        tnor,
        titer,
        snor,
    )
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        check_euclidean_distance = true,
    )

end

main()
