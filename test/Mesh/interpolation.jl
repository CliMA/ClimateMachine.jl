using Test, MPI
import GaussQuadrature
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Geometry
using CLIMA.Mesh.Interpolation
using StaticArrays
using GPUifyLoops

using CLIMA.VariableTemplates
#------------------------------------------------
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
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
#-------------------------------------
function Initialize_Brick_Interpolation_Test!(bl, state::Vars, aux::Vars, (x,y,z), t)
    FT         = eltype(state)
    # Dummy variables for initial condition function
    state.ρ               = FT(0)
    state.ρu              = SVector{3,FT}(0,0,0)
    state.ρe              = FT(0)
    state.moisture.ρq_tot = FT(0)
end
#------------------------------------------------
struct TestSphereSetup{DT}
    p_ground::DT
    T_initial::DT
    domain_height::DT

    function TestSphereSetup(p_ground::DT, T_initial::DT, domain_height::DT) where DT <: AbstractFloat
        return new{DT}(p_ground, T_initial, domain_height)
    end
end
#----------------------------------------------------------------------------
function (setup::TestSphereSetup)(bl, state, aux, coords, t)
    # callable to set initial conditions
    FT = eltype(state)

    r = norm(coords, 2)
    h = r - FT(planet_radius)

    scale_height = R_d * setup.T_initial / grav
    p = setup.p_ground * exp(-h / scale_height)

    state.ρ = air_density(setup.T_initial, p)
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = state.ρ * (internal_energy(setup.T_initial) + aux.orientation.Φ)
    return nothing
end
#----------------------------------------------------------------------------
function run_brick_interpolation_test()
    CLIMA.init()
    for FT in (Float32, Float64)
        ArrayType = CLIMA.array_type()
        DA = CLIMA.array_type()
        mpicomm = MPI.COMM_WORLD
        root = 0
        pid = MPI.Comm_rank(mpicomm)
        npr = MPI.Comm_size(mpicomm)

        xmin, ymin, zmin = 0, 0, 0                   # defining domain extent
        xmax, ymax, zmax = 2000, 400, 2000
        xres = [FT(10), FT(10), FT(10)] # resolution of interpolation grid

        Ne        = (20,4,20)

        polynomialorder = 5 #8# 5 #8 #4
        #-------------------------
        _x, _y, _z = CLIMA.Mesh.Grids.vgeoid.x1id, CLIMA.Mesh.Grids.vgeoid.x2id, CLIMA.Mesh.Grids.vgeoid.x3id
        #-------------------------

        brickrange = (range(FT(xmin); length=Ne[1]+1, stop=xmax),
                      range(FT(ymin); length=Ne[2]+1, stop=ymax),
                      range(FT(zmin); length=Ne[3]+1, stop=zmax))
        topl = StackedBrickTopology(mpicomm, brickrange, periodicity = (true, true, false))
        grid = DiscontinuousSpectralElementGrid(topl,
                                                FloatType = FT,
                                                DeviceArray = DA,
                                                polynomialorder = polynomialorder)
        model = AtmosModel{FT}(AtmosLESConfiguration;
                               ref_state=NoReferenceState(),
                               turbulence=ConstantViscosityWithDivergence(FT(0)),
                               source=(Gravity(),),
                               init_state=Initialize_Brick_Interpolation_Test!)

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

        fcn(x,y,z) = sin.(x) .* cos.(y) .* cos.(z) # sample function
        #----calling interpolation function on state variable # st_idx--------------------------
        nvars = size(Q.data,2)

        for vari in 1:nvars
            Q.data[:,vari,:] = fcn( x1 ./ xmax, x2 ./ ymax, x3 ./ zmax )
        end

        xbnd = Array{FT}(undef,2,3)

        xbnd[1,1] = FT(xmin); xbnd[2,1] = FT(xmax)
        xbnd[1,2] = FT(ymin); xbnd[2,2] = FT(ymax)
        xbnd[1,3] = FT(zmin); xbnd[2,3] = FT(zmax)
        #----------------------------------------------------------
        x1g = collect(range(xbnd[1,1], xbnd[2,1], step=xres[1]))
        x2g = collect(range(xbnd[1,2], xbnd[2,2], step=xres[2]))
        x3g = collect(range(xbnd[1,3], xbnd[2,3], step=xres[3]))

        filename = "test.nc"
        varnames = ("ρ", "ρu", "ρv", "ρw", "e", "other")

        intrp_brck = InterpolationBrick(grid, xbnd, x1g, x2g, x3g)             # sets up the interpolation structure
        iv = DA(Array{FT}(undef, intrp_brck.Npl, nvars))                       # allocating space for the interpolation variable
        interpolate_local!(intrp_brck, Q.data, iv)                             # interpolation
        svi = write_interpolated_data(intrp_brck, iv, varnames, filename)      # write interpolation data to file
        #------------------------------

        err_inf_dom = zeros(FT, nvars)

        x1g = intrp_brck.x1g
        x2g = intrp_brck.x2g
        x3g = intrp_brck.x3g

        if pid==0
            nx1 = length(x1g); nx2 = length(x2g); nx3 = length(x3g)
            x1 = Array{FT}(undef,nx1,nx2,nx3); x2 = similar(x1); x3 = similar(x1)

            for k in 1:nx3, j in 1:nx2, i in 1:nx1
                x1[i,j,k] = x1g[i]
                x2[i,j,k] = x2g[j]
                x3[i,j,k] = x3g[k]
            end
            fex = fcn( x1 ./ xmax , x2 ./ ymax , x3 ./ zmax )

            for vari in 1:nvars
                err_inf_dom[vari] = maximum( abs.(svi[:,:,:,vari] .- fex[:,:,:]) )
            end
        end

        MPI.Bcast!(err_inf_dom, root, mpicomm)

        if FT == Float64
            toler = 1.0E-9
        elseif FT == Float32
            toler = 1.0E-6
        end

        if maximum(err_inf_dom) > toler
            if pid==0
                println("err_inf_domain = $(maximum(err_inf_dom)) is larger than prescribed tolerance of $toler")
            end
            MPI.Barrier(mpicomm)
        end

        @test maximum(err_inf_dom) < toler
    end
    return nothing
    #----------------
end #function run_brick_interpolation_test
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Cubed sphere, lat/long interpolation test
#----------------------------------------------------------------------------
function run_cubed_sphere_interpolation_test()
    CLIMA.init()
    for FT in (Float32, Float64) #Float32 #Float64
        DA = CLIMA.array_type()
        device  = CLIMA.array_type() <: Array ? CPU() : CUDA()
        mpicomm = MPI.COMM_WORLD
        root = 0
        pid  = MPI.Comm_rank(mpicomm)
        npr  = MPI.Comm_size(mpicomm)

        domain_height = FT(30e3)

        polynomialorder = 5
        numelem_horz = 6
        numelem_vert = 4

        #-------------------------
        _x, _y, _z = CLIMA.Mesh.Grids.vgeoid.x1id, CLIMA.Mesh.Grids.vgeoid.x2id, CLIMA.Mesh.Grids.vgeoid.x3id
        _ρ, _ρu, _ρv, _ρw = 1, 2, 3, 4
        #-------------------------
        vert_range = grid1d(FT(planet_radius), FT(planet_radius + domain_height), nelem = numelem_vert)

        lat_res  = FT(1) # 1 degree resolution
        long_res = FT(1) # 1 degree resolution
        nel_vert_grd  = 20 #100 #50 #10#50
        rad_res    = FT((vert_range[end] - vert_range[1])/FT(nel_vert_grd)) #1000.00    # 1000 m vertical resolution
        #----------------------------------------------------------
        setup = TestSphereSetup(FT(MSLP),FT(255),FT(30e3))

        topology = StackedCubedSphereTopology(mpicomm, numelem_horz, vert_range)
   
        grid = DiscontinuousSpectralElementGrid(topology,
                                                FloatType = FT,
                                                DeviceArray = DA,
                                                polynomialorder = polynomialorder,
                                                meshwarp = CLIMA.Mesh.Topologies.cubedshellwarp)

        model = AtmosModel{FT}(AtmosLESConfiguration;
                               orientation=SphericalOrientation(),
                               ref_state=NoReferenceState(),
                               turbulence=ConstantViscosityWithDivergence(FT(0)),
                               moisture=DryModel(),
                               source=nothing,
                               init_state=setup)

        dg = DGModel(model, grid, Rusanov(), CentralNumericalFluxDiffusive(), CentralNumericalFluxGradient())
 
        Q = init_ode_state(dg, FT(0))

        device = typeof(Q.data) <: Array ? CPU() : CUDA()
        #------------------------------
        x1 = @view grid.vgeo[:,_x,:]
        x2 = @view grid.vgeo[:,_y,:]
        x3 = @view grid.vgeo[:,_z,:]

        xmax = FT(planet_radius)
        ymax = FT(planet_radius)
        zmax = FT(planet_radius)

        fcn(x,y,z) = sin.(x) .* cos.(y) .* cos.(z) # sample function

        nvars = size(Q.data,2)

        for i in 1:nvars
            Q.data[:,i,:] .= fcn( x1 ./ xmax, x2 ./ ymax, x3 ./ zmax )
        end
        #------------------------------
        lat_min,   lat_max = FT(-90.0), FT(90.0)            # inclination/zeinth angle range
        long_min, long_max = FT(-180.0), FT(180.0) 		    # azimuthal angle range
        rad_min,   rad_max = vert_range[1], vert_range[end] # radius range


        lat_grd  = collect(range(lat_min,  lat_max,  step=lat_res))
        long_grd = collect(range(long_min, long_max, step=long_res))
        rad_grd  = collect(range(rad_min,  rad_max,  step=rad_res))

        filename = "test.nc"
        varnames = ("ρ", "ρu", "ρv", "ρw", "e")
        projectv = true

        intrp_cs = InterpolationCubedSphere(grid, collect(vert_range), numelem_horz, lat_grd, long_grd, rad_grd) # sets up the interpolation structure
        iv = DA(Array{FT}(undef, intrp_cs.Npl, nvars))                  # allocatind space for the interpolation variable
        interpolate_local!(intrp_cs, Q.data, iv,project=projectv)           # interpolation
        svi = write_interpolated_data(intrp_cs, iv, varnames, filename) # write interpolated data to file
        #----------------------------------------------------------
        # Testing
        err_inf_dom = zeros(FT, nvars)

        rad   = Array(intrp_cs.rad_grd)
        lat   = Array(intrp_cs.lat_grd)
        long  = Array(intrp_cs.long_grd)

        if pid==0
            nrad = length(rad); nlat = length(lat); nlong = length(long)
            x1g = Array{FT}(undef,nrad,nlat,nlong); x2g = similar(x1g); x3g = similar(x1g)

            fex = zeros(FT, nrad, nlat, nlong, nvars)

            for vari in 1:nvars
                for k in 1:nlong, j in 1:nlat, i in 1:nrad
                    x1g_ijk = rad[i] * cosd(lat[j]) * cosd(long[k]) # inclination -> latitude; azimuthal -> longitude.
                    x2g_ijk = rad[i] * cosd(lat[j]) * sind(long[k]) # inclination -> latitude; azimuthal -> longitude.
                    x3g_ijk = rad[i] * sind(lat[j])
             
                    fex[i,j,k,vari] = fcn( x1g_ijk / xmax, x2g_ijk / ymax, x3g_ijk / zmax )
                end
            end

            if projectv
                for k in 1:nlong, j in 1:nlat, i in 1:nrad
                    fex[i,j,k,_ρu] = fex[i,j,k,_ρ] * cosd(lat[j]) * cosd(long[k]) + 
                                     fex[i,j,k,_ρ] * cosd(lat[j]) * sind(long[k]) + 
                                     fex[i,j,k,_ρ] * sind(lat[j])

                    fex[i,j,k,_ρv] = -fex[i,j,k,_ρ] * sind(lat[j]) * cosd(long[k])  
                                     -fex[i,j,k,_ρ] * sind(lat[j]) * sind(long[k]) + 
                                      fex[i,j,k,_ρ] * cosd(lat[j])

                    fex[i,j,k,_ρw] = -fex[i,j,k,_ρ] * cosd(lat[j]) * sind(long[k]) + 
                                      fex[i,j,k,_ρ] * cosd(lat[j]) * cosd(long[k])
                end
            end

            for vari in 1:nvars
                err_inf_dom[vari] = maximum( abs.(svi[:,:,:,vari] .- fex[:,:,:,vari]) )
            end
        end

        MPI.Bcast!(err_inf_dom, root, mpicomm)

        if FT == Float64
            toler = 1.0E-7
        elseif FT == Float32
            toler = 1.0E-6
        end

        if maximum(err_inf_dom) > toler
            if pid==0
                println("err_inf_domain = $(maximum(err_inf_dom)) is larger than prescribed tolerance of $toler")
            end
            MPI.Barrier(mpicomm)
        end
        @test maximum(err_inf_dom) < toler
    end
    return nothing
end
#----------------------------------------------------------------------------
@testset "Interpolation tests" begin
    run_brick_interpolation_test()
    run_cubed_sphere_interpolation_test()
end
#------------------------------------------------
