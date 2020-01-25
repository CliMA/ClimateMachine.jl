using Test, MPI
import GaussQuadrature
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Geometry
using CLIMA.Mesh.Interpolation
using StaticArrays

#------------------------------------------------
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Atmos
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.TicToc
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK

using CLIMA.Atmos: vars_state, vars_aux

using Random
using Statistics
const seed = MersenneTwister(0)

const ArrayType = CLIMA.array_type()


#------------------------------------------------
if !@isdefined integration_testing
    const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end
#------------------------------------------------
function run_brick_interpolation_test()
  MPI.Initialized() || MPI.Init()

#@testset "LocalGeometry" begin
    FT = Float64
    ArrayType = Array
    mpicomm = MPI.COMM_WORLD

    xmin, ymin, zmin = 0, 0, 0                   # defining domain extent
    xmax, ymax, zmax = 2000, 400, 2000
    xres = [FT(200), FT(200), FT(200)] # resolution of interpolation grid

    xgrd = range(xmin, xmax, step=xres[1])
    ygrd = range(ymin, ymax, step=xres[2])
    zgrd = range(zmin, zmax, step=xres[3])

 #   Ne        = (20,2,20)
    Ne        = (4,2,4)

    polynomialorder = 8 #8 #4
    #-------------------------
    _x, _y, _z = CLIMA.Mesh.Grids.vgeoid.x1id, CLIMA.Mesh.Grids.vgeoid.x2id, CLIMA.Mesh.Grids.vgeoid.x3id
    _ρ, _ρu, _ρv, _ρw = 1, 2, 3, 4
    #-------------------------

    brickrange = (range(FT(xmin); length=Ne[1]+1, stop=xmax),
                  range(FT(ymin); length=Ne[2]+1, stop=ymax),
                  range(FT(zmin); length=Ne[3]+1, stop=zmax))
    topl = StackedBrickTopology(MPI.COMM_SELF, brickrange, periodicity = (true, true, false))

    grid = DiscontinuousSpectralElementGrid(topl,
                                            FloatType = FT,
                                            DeviceArray = ArrayType,
                                            polynomialorder = polynomialorder)

    model = AtmosModel(FlatOrientation(),
                     NoReferenceState(),
					 ConstantViscosityWithDivergence(FT(0)),
                     EquilMoist(),
                     NoPrecipitation(),
                     NoRadiation(),
                     NoSubsidence{FT}(),
                     (Gravity()),
					 NoFluxBC(),
                     Initialize_Brick_Interpolation_Test!)

    dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralNumericalFluxGradient())

    Q = init_ode_state(dg, FT(0))
    #------------------------------
    x1 = @view grid.vgeo[:,_x,:]
    x2 = @view grid.vgeo[:,_y,:]
    x3 = @view grid.vgeo[:,_z,:]

    st_idx = _ρ # state vector
    elno = 10

    var = @view Q.data[:,st_idx,:]

    #fcn(x,y,z) = x .* y .* z # sample function
    fcn(x,y,z) = sin.(x) .* cos.(y) .* cos.(z) # sample function

    var .= fcn( x1 ./ xmax, x2 ./ ymax, x3 ./ zmax )
    #----calling interpolation function on state variable # st_idx--------------------------
    #intrp_brck = InterpolationBrick(grid, xres, FT)
    intrp_brck = InterpolationBrick(grid, xres)
    interpolate_brick!(intrp_brck, Q.data, st_idx)


    #------testing
    Nel = length( grid.topology.realelems )

    error = zeros(FT, Nel)

    for elno in 1:Nel
      fex = similar(intrp_brck.V[elno])
      fex = fcn( intrp_brck.x[elno][1,:] ./ xmax , intrp_brck.x[elno][2,:] ./ ymax , intrp_brck.x[elno][3,:] ./ zmax )
      error[elno] = maximum(abs.(intrp_brck.V[elno][:]-fex[:]))
    end

    l_infinity_local = maximum(error)
    l_infinity_domain = MPI.Allreduce(l_infinity_local, MPI.MAX, mpicomm)

    return l_infinity_domain < 1.0e-14
    #----------------
end #function run_brick_interpolation_test

#-----taken from Test example
function Initialize_Brick_Interpolation_Test!(state::Vars, aux::Vars, (x,y,z), t)
    FT         = eltype(state)

    # Dummy variables for initial condition function
    state.ρ     = FT(0)
    state.ρu    = SVector{3,FT}(0,0,0)
    state.ρe    = FT(0)
    state.moisture.ρq_tot = FT(0)
end
#------------------------------------------------
Base.@kwdef struct TestSphereSetup{FT}
  p_ground::FT = MSLP
  T_initial::FT = 255
  domain_height::FT = 30e3
end
#----------------------------------------------------------------------------
# Cubed sphere, lat/long interpolation test
#----------------------------------------------------------------------------
function run_cubed_sphere_interpolation_test()
    CLIMA.init()

    FT = Float64
    mpicomm = MPI.COMM_WORLD
    root = 0

    ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
    loglevel = Dict("DEBUG" => Logging.Debug,
                    "WARN"  => Logging.Warn,
                    "ERROR" => Logging.Error,
                    "INFO"  => Logging.Info)[ll]

    logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))

    domain_height = FT(30e3)

    polynomialorder = 12#1#4 #5
    numelem_horz = 3#4 #6
    numelem_vert = 4#1 #1 #1#6 #8

    #-------------------------
    _x, _y, _z = CLIMA.Mesh.Grids.vgeoid.x1id, CLIMA.Mesh.Grids.vgeoid.x2id, CLIMA.Mesh.Grids.vgeoid.x3id
    _ρ, _ρu, _ρv, _ρw = 1, 2, 3, 4
    #-------------------------
    vert_range = grid1d(FT(planet_radius), FT(planet_radius + domain_height), nelem = numelem_vert)

 #   vert_range = grid1d(FT(1.0), FT(2.0), nelem = numelem_vert)

    lat_res  = 5 * π / 180.0 # 5 degree resolution
    long_res = 5 * π / 180.0 # 5 degree resolution
    r_res    = (vert_range[end] - vert_range[1])/FT(numelem_vert) #1000.00    # 1000 m vertical resolution

    #----------------------------------------------------------
    setup = TestSphereSetup{FT}()

    topology = StackedCubedSphereTopology(mpicomm, numelem_horz, vert_range)

    grid = DiscontinuousSpectralElementGrid(topology,
                                            FloatType = FT,
                                            DeviceArray = ArrayType,
                                            polynomialorder = polynomialorder,
                                            meshwarp = CLIMA.Mesh.Topologies.cubedshellwarp)

    model = AtmosModel(SphericalOrientation(),
                       NoReferenceState(),
                       ConstantViscosityWithDivergence(FT(0)),
                       DryModel(),
                       NoPrecipitation(),
                       NoRadiation(),
                       NoSubsidence{FT}(),
                       nothing,
                       NoFluxBC(),
                       setup)

    dg = DGModel(model, grid, Rusanov(),
                 CentralNumericalFluxDiffusive(), CentralNumericalFluxGradient())

    Q = init_ode_state(dg, FT(0))
    #------------------------------
    x1 = @view grid.vgeo[:,_x,:]
    x2 = @view grid.vgeo[:,_y,:]
    x3 = @view grid.vgeo[:,_z,:]

    xmax = maximum( abs.(x1) )
    ymax = maximum( abs.(x2) )
    zmax = maximum( abs.(x3) )

    st_idx = _ρ # state vector

    var = @view Q.data[:,st_idx,:]

#    fcn(x,y,z) = x .* y .* z # sample function
    fcn(x,y,z) = sin.(x) .* cos.(y) .* cos.(z) # sample function

    var .= fcn( x1 ./ xmax, x2 ./ ymax, x3 ./ zmax )
  #------------------------------
    @time intrp_cs = InterpolationCubedSphere(grid, collect(vert_range), numelem_horz, lat_res, long_res, r_res)
    interpolate_cubed_sphere!(intrp_cs, Q.data, st_idx)
    #----------------------------------------------------------

    Nel = length( grid.topology.realelems )

    error = zeros(FT, Nel)

    for elno in 1:Nel
        if ( length(intrp_cs.V[elno]) > 0 )
            fex = similar(intrp_cs.V[elno])
            x1g = similar(intrp_cs.V[elno])
            x2g = similar(intrp_cs.V[elno])
            x3g = similar(intrp_cs.V[elno])

            x1_grd = intrp_cs.radc[elno] .* sin.(intrp_cs.latc[elno]) .* cos.(intrp_cs.longc[elno]) # inclination -> latitude; azimuthal -> longitude.
            x2_grd = intrp_cs.radc[elno] .* sin.(intrp_cs.latc[elno]) .* sin.(intrp_cs.longc[elno]) # inclination -> latitude; azimuthal -> longitude.
            x3_grd = intrp_cs.radc[elno] .* cos.(intrp_cs.latc[elno])

            fex = fcn( x1_grd ./ xmax , x2_grd ./ ymax , x3_grd ./ zmax )
            error[elno] = maximum(abs.(intrp_cs.V[elno][:]-fex[:]))
        end
    end
    #----------------------------------------------------------
    l_infinity_local = maximum(error)
    l_infinity_domain = MPI.Allreduce(l_infinity_local, MPI.MAX, mpicomm)

#----------------------------------------------------------------------------
    return l_infinity_domain < 1.0e-12
end
#----------------------------------------------------------------------------
function (setup::TestSphereSetup)(state, aux, coords, t)
  # callable to set initial conditions
  FT = eltype(state)

  r = norm(coords, 2)
  h = r - FT(planet_radius)

  scale_height = R_d * setup.T_initial / grav
  p = setup.p_ground * exp(-h / scale_height)

  state.ρ = air_density(setup.T_initial, p)
  state.ρu = SVector{3, FT}(0, 0, 0)
  state.ρe = state.ρ * (internal_energy(setup.T_initial) + aux.orientation.Φ)
  nothing
end
#----------------------------------------------------------------------------

@testset "Interpolation tests" begin
    @test run_brick_interpolation_test()
    @test run_cubed_sphere_interpolation_test()
end
#------------------------------------------------

