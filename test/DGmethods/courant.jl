using Test
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Grids: VerticalDirection, HorizontalDirection, EveryDirection
using CLIMA.VTK
using Logging
using Printf
using LinearAlgebra
using CLIMA.DGmethods: DGModel, init_ode_state, LocalGeometry, courant
using CLIMA.DGmethods.NumericalFluxes: Rusanov, CentralNumericalFluxGradient,
                                       CentralNumericalFluxDiffusive
using CLIMA.Courant
using CLIMA.PlanetParameters: kappa_d
using CLIMA.Atmos: AtmosModel,
                   AtmosAcousticLinearModel, RemainderModel,
                   FlatOrientation,
                   NoReferenceState, ReferenceState,
                   DryModel, NoRadiation, NoSubsidence, PeriodicBC, NoPrecipitation,
                   Gravity, HydrostaticState, IsothermalProfile,
                   ConstantViscosityWithDivergence, vars_state, soundspeed
using CLIMA.Atmos
using CLIMA.ODESolvers
using CLIMA.LowStorageRungeKuttaMethod
const ArrayType = CLIMA.array_type()

using CLIMA.MoistThermodynamics: air_density, total_energy, internal_energy,
                                 soundspeed_air

using CLIMA.VariableTemplates: Vars
using StaticArrays

function initialcondition!(state, aux, coords, t)
    FT = eltype(state)

    p∞::FT = 10 ^ 5
    T∞::FT = 300
    translation_speed::FT = 150
    translation_angle::FT = pi / 4
    α = translation_angle
    u∞ = SVector(translation_speed * coords[1], translation_speed * coords[1], 0)

    u = u∞
    T = T∞
    # adiabatic/isentropic relation
    p = p∞ * (T / T∞) ^ (FT(1) / kappa_d)
    ρ = air_density(T, p)

    state.ρ = ρ
    state.ρu = ρ * u
    e_kin = u' * u / 2
    state.ρe = ρ * total_energy(e_kin, FT(0), T)

    nothing
end


let
    # boiler plate MPI stuff
    CLIMA.init()
    mpicomm = MPI.COMM_WORLD

    # Mesh generation parameters
    N = 4
    Nq = N+1
    Neh = 10
    Nev = 4

    @testset "$(@__FILE__) DGModel matrix" begin
        for FT in (Float64, Float32)
            for dim = (2, 3)
                if dim == 2
                    brickrange = (range(FT(0); length=Neh+1, stop=1),
                                  range(FT(1); length=Nev+1, stop=2))
                elseif dim == 3
                    brickrange = (range(FT(0); length=Neh+1, stop=1),
                                  range(FT(0); length=Neh+1, stop=1),
                                  range(FT(1); length=Nev+1, stop=2))
                end

                topl = StackedBrickTopology(mpicomm, brickrange)

                function warpfun(ξ1, ξ2, ξ3)
                    FT = eltype(ξ1)

                    ξ1 ≥ FT(1//2) && (ξ1 = FT(1//2) + 2(ξ1 - FT(1//2)))
                    if dim == 2
                        ξ2 ≥ FT(3//2) && (ξ2 = FT(3//2) + 2(ξ2 - FT(3//2)))
                    elseif dim == 3
                        ξ2 ≥ FT(1//2) && (ξ2 = FT(1//2) + 2(ξ2 - FT(1//2)))
                        ξ3 ≥ FT(3//2) && (ξ3 = FT(3//2) + 2(ξ3 - FT(3//2)))
                    end
                    return (ξ1, ξ2, ξ3)
                end

                grid = DiscontinuousSpectralElementGrid(topl,
                                                        FloatType = FT,
                                                        DeviceArray = ArrayType,
                                                        polynomialorder = N)
                                                #meshwarp = warpfun)

                model = AtmosModel{FT}(AtmosLESConfiguration;
                                       ref_state=NoReferenceState(),
                                       turbulence=ConstantViscosityWithDivergence(FT(2)),
                                       moisture=DryModel(),
                                       source=Gravity(),
                                       boundarycondition=PeriodicBC(),
                                       init_state=initialcondition!)

                dg = DGModel(model, grid, Rusanov(), CentralNumericalFluxDiffusive(),
                             CentralNumericalFluxGradient())

                Δt = FT(1e-11)#FT(1//2)
                #Δt = FT(0.00001) #FT(1//10)

                Q = init_ode_state(dg, FT(0))
                solver = LSRK54CarpenterKennedy(dg, Q; dt = Δt, t0 = 0)
                solve!(Q, solver; timeend=Δt)
                Δx = min_node_distance(grid, EveryDirection())
                Δx_v = min_node_distance(grid, VerticalDirection())
                Δx_h = min_node_distance(grid, HorizontalDirection())

                T∞ = FT(300)
                translation_speed = FT(212.13203435596427)
                diff_speed_h = 734.8469228349534
                diff_speed_v = 0
                c   = Δt*(translation_speed + soundspeed_air(T∞))/Δx
                c_h = Δt*(translation_speed + soundspeed_air(T∞))/Δx_h
                c_v = Δt*(soundspeed_air(T∞))/Δx_v
                d_h = Δt*diff_speed_h/Δx_h^2
                d_v = Δt*diff_speed_v/Δx_v^2

                if (FT==Float64 && dim == 2)
                    @test abs(courant(nondiffusive_courant, dg, model, Q, Δt, HorizontalDirection()) - c_h) <= 1e-15
                    @test abs(courant(diffusive_courant,    dg, model, Q, Δt, HorizontalDirection()) - d_h) <= 1e-4 
                    @test abs(courant(nondiffusive_courant, dg, model, Q, Δt, VerticalDirection()) - c_v) <= 1e-16
                    @test abs(courant(diffusive_courant,    dg, model, Q, Δt, VerticalDirection()) - d_v) <= 1e-8 
                elseif (FT==Float64 && dim == 3)
                    @test abs(courant(nondiffusive_courant, dg, model, Q, Δt, HorizontalDirection()) - c_h) <= 1e-11
                    @test abs(courant(diffusive_courant,    dg, model, Q, Δt, HorizontalDirection()) - d_h) <= 1e-4 
                    @test abs(courant(nondiffusive_courant, dg, model, Q, Δt, VerticalDirection()) - c_v) <= 1e-11
                    @test abs(courant(diffusive_courant,    dg, model, Q, Δt, VerticalDirection()) - d_v) <= 1e-8 
                elseif (dim == 2)
                    @test abs(courant(nondiffusive_courant, dg, model, Q, Δt, HorizontalDirection()) - c_h) <= 1e-11
                    @test abs(courant(diffusive_courant,    dg, model, Q, Δt, HorizontalDirection()) - d_h) <= 1e-4
                    @test abs(courant(nondiffusive_courant, dg, model, Q, Δt, VerticalDirection()) - c_v) <= 1e-16
                    @test abs(courant(diffusive_courant,    dg, model, Q, Δt, VerticalDirection()) - d_v) <= 1e-8 
                else
                    @test abs(courant(nondiffusive_courant, dg, model, Q, Δt, HorizontalDirection()) - c_h) <= 1e-11
                    @test abs(courant(diffusive_courant,    dg, model, Q, Δt, HorizontalDirection()) - d_h) <= 1e-4
                    @test abs(courant(nondiffusive_courant, dg, model, Q, Δt, VerticalDirection()) - c_v) <= 1e-11
                    @test abs(courant(diffusive_courant,    dg, model, Q, Δt, VerticalDirection()) - d_v) <= 2e-8
                end
            end
        end
    end
end

nothing
