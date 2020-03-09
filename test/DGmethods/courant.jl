using Test
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
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
                   DryModel, NoRadiation, PeriodicBC, NoPrecipitation,
                   Gravity, HydrostaticState, IsothermalProfile,
                   ConstantViscosityWithDivergence, vars_state, soundspeed
using CLIMA.Atmos
using CLIMA.ODESolvers

using CLIMA.MoistThermodynamics: air_density, total_energy, internal_energy,
                                 soundspeed_air

using CLIMA.VariableTemplates: Vars
using StaticArrays

const p∞ = 10 ^ 5
const T∞ = 300.0

function initialcondition!(bl, state, aux, coords, t)
    FT = eltype(state)


    translation_speed::FT = 150
    translation_angle::FT = pi / 4
    α = translation_angle
    u∞ = SVector(translation_speed * coords[1], translation_speed * coords[1], 0)

    u = u∞
    T = FT(T∞)
    # adiabatic/isentropic relation
    p = FT(p∞) * (T / FT(T∞)) ^ (FT(1) / kappa_d)
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
    ArrayType = CLIMA.array_type()
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
                μ = FT(2)
                topl = StackedBrickTopology(mpicomm, brickrange)



                grid = DiscontinuousSpectralElementGrid(topl,
                                                        FloatType = FT,
                                                        DeviceArray = ArrayType,
                                                        polynomialorder = N)

                model = AtmosModel{FT}(AtmosLESConfiguration;
                                       ref_state=NoReferenceState(),
                                       turbulence=ConstantViscosityWithDivergence(μ),
                                       moisture=DryModel(),
                                       source=Gravity(),
                                       boundarycondition=PeriodicBC(),
                                       init_state=initialcondition!)

                dg = DGModel(model, grid, Rusanov(), CentralNumericalFluxDiffusive(),
                             CentralNumericalFluxGradient())

                Δt = FT(1//200)

                Q = init_ode_state(dg, FT(0))

                Δx = min_node_distance(grid, EveryDirection())
                Δx_v = min_node_distance(grid, VerticalDirection())
                Δx_h = min_node_distance(grid, HorizontalDirection())

                translation_speed = FT( norm( [150.0, 150.0, 0.0] ) )
                diff_speed_h = FT(μ / air_density(FT(T∞), FT(p∞)))
                diff_speed_v = FT(μ / air_density(FT(T∞), FT(p∞)))
                c_h = Δt*(translation_speed + soundspeed_air(T∞))/Δx_h
                c_v = Δt*(soundspeed_air(T∞))/Δx_v
                d_h = Δt*diff_speed_h/Δx_h^2
                d_v = Δt*diff_speed_v/Δx_v^2

                # tests for non diffusive courant number
                @test courant(nondiffusive_courant, dg, model, Q, Δt, HorizontalDirection()) ≈ c_h rtol=1e-4
                @test courant(nondiffusive_courant, dg, model, Q, Δt, VerticalDirection())   ≈ c_v rtol=1e-4

                # tests for diffusive courant number
                @test courant(diffusive_courant,    dg, model, Q, Δt, HorizontalDirection()) ≈ d_h
                @test courant(diffusive_courant,    dg, model, Q, Δt, VerticalDirection())   ≈ d_v
            end
        end
    end
end

nothing
