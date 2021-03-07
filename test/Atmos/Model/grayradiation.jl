using StaticArrays
using Test
using LinearAlgebra
using Plots
using ProgressMeter

using ClimateMachine
ClimateMachine.init(parse_clargs = true, show_updates = "never")

using ClimateMachine.MPIStateArrays
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.ODESolvers
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using ClimateMachine.SingleStackUtils
using ClimateMachine.GenericCallbacks
using ClimateMachine.DGMethods: FVLinear
using ClimateMachine.BalanceLaws: GradientFlux

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

####################### Adding to Preexisting Interface #######################

using UnPack
using ClimateMachine.BalanceLaws: Auxiliary, TendencyDef, Flux, FirstOrder,
    BalanceLaw, UpwardIntegrals
using ClimateMachine.DGMethods: SpaceDiscretization
using ClimateMachine.Orientations: vertical_unit_vector
using ClimateMachine.Atmos: nodal_update_auxiliary_state!
import ClimateMachine.BalanceLaws: vars_state, eq_tends, flux
import ClimateMachine.Atmos: update_auxiliary_state!

# Define a new type of radiation model that utilizes RRTMGP.
struct RRTMGPModel <: RadiationModel end

# Allocate space in which to unload the energy fluxes calculated by RRTMGP.
vars_state(::RRTMGPModel, ::Auxiliary, FT) = @vars(flux::FT)

# Define a new type of tendency that adds those fluxes to the net energy flux.
struct Radiation{PV <: Energy} <: TendencyDef{Flux{FirstOrder}, PV} end
eq_tends(::PV, ::RRTMGPModel, ::Flux{FirstOrder}) where {PV <: Energy} =
    (Radiation{PV}(),)
function flux(::Radiation{Energy}, bl, args)
    @unpack aux = args
    return aux.radiation.flux * vertical_unit_vector(bl, aux)
end

# Make update_auxiliary_state! also update data for radiation.
function update_auxiliary_state!(
    spacedisc::SpaceDiscretization,
    m::AtmosModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    FT = eltype(Q)
    state_auxiliary = spacedisc.state_auxiliary

    if number_states(m, UpwardIntegrals()) > 0
        indefinite_stack_integral!(spacedisc, m, Q, state_auxiliary, t, elems)
        reverse_indefinite_stack_integral!(spacedisc, m, Q, state_auxiliary, t, elems)
    end

    update_auxiliary_state!(nodal_update_auxiliary_state!, spacedisc, m, Q, t, elems)

    # TODO: Remove this hook. This hook was added for implementing
    # the first draft of EDMF, and should be removed so that we can
    # rely on a single vertical element traversal. This hook allows
    # us to compute globally vertical quantities specific to EDMF
    # until we're able to remove them or somehow incorporate them
    # into a higher level hierarchy.
    update_auxiliary_state!(spacedisc, m.turbconv, m, Q, t, elems)

    # Update the radiation model's auxiliary state in a seperate traversal.
    update_auxiliary_state!(spacedisc, m.radiation, m, Q, t, elems)

    return true
end

# By default, don't do anything special for radiation.
function update_auxiliary_state!(
    spacedisc::SpaceDiscretization,
    ::RadiationModel,
    m::BalanceLaw,
    state_prognostic::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
end

############################# New Code for RRTMGP #############################

using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using NCDatasets: Dataset
using RRTMGP.AtmosphericStates:      GrayAtmosphericState, AtmosphericState
using RRTMGP.LookUpTables:           LookUpLW
using RRTMGP.Vmrs:                   Vmr
using RRTMGP.Optics:                 init_optical_props, compute_col_dry!
using RRTMGP.Sources:                source_func_longwave
using RRTMGP.BCs:                    LwBCs
using RRTMGP.AngularDiscretizations: AngularDiscretization
using RRTMGP.Fluxes:                 FluxLW
using RRTMGP.RTE:                    Solver
using RRTMGP.GrayRTESolver:          solve_lw!
using ClimateMachine.BalanceLaws: number_states, Prognostic
using ClimateMachine.Thermodynamics: air_pressure, air_temperature,
    q_vap_saturation, PhasePartition, vol_vapor_mixing_ratio

const nstreams = 1
const ngpoints = 1
const ngaussangles = 1
const lookup_filename = joinpath(@__DIR__, "rrtmgp-data-lw-g256-2018-12-04.nc")

const temp_sfc = 290 # Surface temperature; used by both gray and full optics
const temp_toa = 200 # Top-of-atmosphere temperature; only used by gray optics

struct GrayRRTMGPData{ST}
    solver::ST
end
struct FullRRTMGPData{ST, LT, VT}
    solver::ST
    lookup_lw::LT
    h2o_lev::VT
end

function gray_as(DA, FT, nvert, nhorz, vgeo, Nq, nvertelem, horzelems)
    as = GrayAtmosphericState(
        similar(DA, FT, nvert - 1, nhorz),      # p_lay
        similar(DA, FT, nvert, nhorz),          # p_lev
        similar(DA, FT, nvert - 1, nhorz),      # t_lay
        similar(DA, FT, nvert, nhorz),          # t_lev
        similar(DA, FT, nvert, nhorz),          # z_lev
        similar(DA, FT, nhorz),                 # t_sfc
        FT(3.5),                                # α
        similar(DA, FT, nhorz),                 # d0
        nvert - 1,                              # nlay
        nhorz,                                  # ncol
    )

    # Initialize the surface temperature and optical thickness parameter
    # arrays.
    # TODO: get these values from the boundary conditions.
    as.t_sfc .= FT(temp_sfc)
    as.d0 .= (FT(temp_sfc) / FT(temp_toa))^4 - FT(1)

    # Initialize the z-coordinate array.
    device = array_device(vgeo)
    event = Event(device)
    event = kernel_init_z!(device, (Nq[1], Nq[2]))(
        map(Val, Nq)...,
        Val(nvertelem),
        as.z_lev,
        vgeo,
        horzelems;
        ndrange = (length(horzelems) * Nq[1], Nq[2]),
        dependencies = (event,),
    )
    wait(device, event)

    # Allocate and initialize the surface emmisivity array.
    # TODO: get this value from the boundary conditions.
    sfc_emis = similar(DA, FT, nhorz)
    sfc_emis .= FT(1)

    # Note that arrays related to air temperature and pressure (as.p_lay,
    # as.p_lev, as.t_lay, and as.t_lev) are not initialized.

    return as, sfc_emis
end

function full_as(DA, FT, nvert, nhorz)
    ds_lw = Dataset(lookup_filename, "r")
    lookup_lw, idx_gases = LookUpLW(ds_lw, Int, FT, DA)
    close(ds_lw)

    as = AtmosphericState(
        similar(DA, FT, nhorz),                 # lon
        similar(DA, FT, nhorz),                 # lat
        similar(DA, FT, nhorz),                 # sfc_emis
        similar(DA, FT, nhorz),                 # sfc_alb
        similar(DA, FT, nhorz),                 # zenith
        similar(DA, FT, nhorz),                 # irrad
        similar(DA, FT, nvert - 1, nhorz),      # p_lay
        similar(DA, FT, nvert, nhorz),          # p_lev
        similar(DA, FT, nvert - 1, nhorz),      # t_lay
        similar(DA, FT, nvert, nhorz),          # t_lev
        similar(DA, FT, nhorz),                 # t_sfc
        similar(DA, FT, nvert - 1, nhorz),      # col_dry
        Vmr(
            similar(DA, FT, nvert - 1, nhorz),  # vmr_h2o
            similar(DA, FT, nvert - 1, nhorz),  # vmr_o3
            similar(DA, FT, lookup_lw.n_gases), # vmr
        ),
        nvert - 1,                              # nlay
        nhorz,                                  # ncol
        lookup_lw.n_gases,                      # ngas
    )

    # Initialize the longitude and latitude arrays.
    # TODO: get these values from the Orientation struct.
    as.lon .= FT(0)
    as.lat .= FT(π) * FT(17) / FT(180)

    # Initialize the surface emissivity, surface albedo, solar zenith angle,
    # total solar irradiance, and surface temperature arrays.
    # TODO: get these values from the boundary conditions.
    as.sfc_emis .= FT(1)
    as.sfc_alb .= FT(0.1)
    as.zenith .= FT(0)
    as.irrad .= FT(1365)
    as.t_sfc .= FT(temp_sfc)

    # Initialize the ozone and other non-water volume mixing ratio arrays.
    # TODO: get these values from lookup tables and/or tracer variables.
    as.vmr.vmr_o3 .= FT(0)
    as.vmr.vmr .= FT(0)
    as.vmr.vmr[idx_gases["n2"]] = FT(0.780840)
    as.vmr.vmr[idx_gases["o2"]] = FT(0.209460)
    as.vmr.vmr[idx_gases["co2"]] = FT(0.000415)

    # Note that arrays related to air temperature, pressure, and water content
    # (as.p_lay, as.p_lev, as.t_lay, as.t_lev, as.col_dry, and as.vmr.vmr_h2o)
    # are not initialized.

    return as, as.sfc_emis, lookup_lw
end

# Allocate and initialize the data required by RRTMGP.
function RRTMGPData(grid, optics_symbol)
    elems = grid.topology.realelems
    nvertelem = grid.topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)
    dim = dimensionality(grid)
    npoly = polynomialorders(grid)
    Nq = (npoly[1] + 1, dim == 2 ? 1 : npoly[2] + 1, npoly[dim] + 1)

    vgeo = grid.vgeo
    FT = eltype(vgeo)
    DA = arraytype(grid)

    # Compute the dimensions of arrays for the RRTMGP Solver struct.
    nvert = Nq[3] == 1 ? nvertelem : (Nq[3] - 1) * nvertelem + 1
    nhorz = Nq[1] * Nq[2] * length(horzelems)

    if optics_symbol == :gray
        as, sfc_emis =
            gray_as(DA, FT, nvert, nhorz, vgeo, Nq, nvertelem, horzelems)
    else # :full
        as, sfc_emis, lookup_lw = full_as(DA, FT, nvert, nhorz)
    end
    
    opc = nstreams == 1 ? :OneScalar : :TwoStream
    op = init_optical_props(opc, FT, DA, nhorz, nvert - 1)
    src_lw = source_func_longwave(FT, nhorz, nvert - 1, ngpoints, opc, DA)
    bcs_lw = LwBCs(sfc_emis, nothing)
    ang_disc = AngularDiscretization(opc, FT, ngaussangles, DA)
    fluxb_lw = FluxLW(nhorz, nvert - 1, FT, DA)
    flux_lw = FluxLW(nhorz, nvert - 1, FT, DA)
    solver = Solver{
        FT,
        Int,
        DA{FT, 1},
        DA{FT, 2},
        typeof(as),
        typeof(op),
        typeof(src_lw),
        Nothing,
        typeof(bcs_lw),
        Nothing,
        typeof(ang_disc),
        typeof(fluxb_lw),
        Nothing,
        typeof(flux_lw),
        Nothing,
    }(
        as,
        op,
        src_lw,
        nothing,
        bcs_lw,
        nothing,
        ang_disc,
        fluxb_lw,
        nothing,
        flux_lw,
        nothing,
    )

    if optics_symbol == :gray
        return GrayRRTMGPData(solver)
    else # :full
        return FullRRTMGPData(solver, lookup_lw, similar(DA, FT, nvert, nhorz))
    end
end

# Update the RRTMGP data, use it to calculate radiation energy fluxes, and
# unload those fluxes into the auxiliary state.
function update_auxiliary_state!(
    spacedisc::SpaceDiscretization,
    ::RRTMGPModel,
    m::BalanceLaw,
    state_prognostic::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    RRTMGPdata = spacedisc.modeldata.RRTMGPdata
    solver = RRTMGPdata.solver
    as = solver.as

    grid = spacedisc.grid
    dim = dimensionality(grid)
    npoly = polynomialorders(grid)
    Nq = (npoly[1] + 1, dim == 2 ? 1 : npoly[2] + 1, npoly[dim] + 1)
    nvertelem = grid.topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)
    device = array_device(state_prognostic)

    # Update the RRTMGP data.
    if RRTMGPdata isa GrayRRTMGPData
        event = Event(device)
        event = kernel_load_into_rrtmgp!(device, (Nq[1], Nq[2]))(
            m,
            map(Val, Nq)...,
            Val(nvertelem),
            state_prognostic.data,
            spacedisc.state_auxiliary.data,
            horzelems,
            as.p_lev,
            as.p_lay,
            as.t_lev,
            as.t_lay;
            ndrange = (length(horzelems) * Nq[1], Nq[2]),
            dependencies = (event,),
        )
        wait(device, event)
        solve_lw!(solver)
    else # FullRRTMGPData
        event = Event(device)
        event = kernel_load_into_rrtmgp!(device, (Nq[1], Nq[2]))(
            m,
            map(Val, Nq)...,
            Val(nvertelem),
            state_prognostic.data,
            spacedisc.state_auxiliary.data,
            horzelems,
            as.p_lev,
            as.p_lay,
            as.t_lev,
            as.t_lay,
            RRTMGPdata.h2o_lev,
            as.vmr.vmr_h2o;
            ndrange = (length(horzelems) * Nq[1], Nq[2]),
            dependencies = (event,),
        )
        wait(device, event)
        compute_col_dry!(
            as.p_lev,
            as.t_lay,
            as.col_dry,
            m.param_set,
            as.vmr.vmr_h2o,
            as.lat,
        )
        solve_lw!(solver, RRTMGPdata.lookup_lw)
    end

    # Update the radiation energy flux array in the auxiliary state.
    event = Event(device)
    event = kernel_unload_from_rrtmgp!(device, (Nq[1], Nq[2]))(
        map(Val, Nq)...,
        Val(nvertelem),
        Val(varsindex(vars(spacedisc.state_auxiliary), :radiation, :flux)[1]),
        spacedisc.state_auxiliary.data,
        solver.flux_lw.flux_net,
        horzelems;
        ndrange = (length(horzelems) * Nq[1], Nq[2]),
        dependencies = (event,),
    )
    wait(device, event)
end

@kernel function kernel_init_z!(
    ::Val{Nq1}, ::Val{Nq2}, ::Val{1}, ::Val{nvertelem},
    z,
    vgeo::AbstractArray{FT},
    elems,
) where {Nq1, Nq2, nvertelem, FT}
    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]
        h = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i
        
        for ev in 1:nvertelem
            e = nvertelem * (eh - 1) + ev
            ijk = Nq1 * (j - 1) + i
            z[ev, h] = vgeo[ijk, Grids._x3, e]
        end
    end
end

@kernel function kernel_init_z!(
    ::Val{Nq1}, ::Val{Nq2}, ::Val{Nq3}, ::Val{nvertelem},
    z,
    vgeo::AbstractArray{FT},
    elems,
) where {Nq1, Nq2, Nq3, nvertelem, FT}
    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]
        h = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i
        
        # Fill in all but the top z-coordinates.
        for ev in 1:nvertelem
            e = nvertelem * (eh - 1) + ev
            @unroll for k in 1:Nq3 - 1
                ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                z[(Nq3 - 1) * (ev - 1) + k, h] = vgeo[ijk, Grids._x3, e]
            end
        end

        # Fill in the top z-coordinates (ev = nvertelem, k = Nq3).
        ijk = Nq1 * (Nq2 * (Nq3 - 1) + (j - 1)) + i
        z[(Nq3 - 1) * nvertelem + 1, h] = vgeo[ijk, Grids._x3, nvertelem * eh]
    end
end

function relhum(FT, z)
    z1 = FT(15e3)
    z2 = FT(25e3)
    relhum1 = FT(0.5)
    relhum2 = FT(1e-4)
    if z <= z1
        return relhum1
    elseif z <= z2
        return relhum2 + (relhum1 - relhum2) * FT(z2 - z) / FT(z2 - z1)
    else
        return relhum2
    end
end

function lev_set!(FT, bl, prog, aux, v, h, p_lev, p_lay, t_lev, t_lay, qvapmin)
    thermo_state = recover_thermo_state(bl, prog, aux)
    p_lev[v, h] = air_pressure(thermo_state)
    t_lev[v, h] = air_temperature(thermo_state)
    return qvapmin
end

function lev_set!(
    FT,
    bl,
    prog,
    aux,
    v,
    h,
    p_lev,
    p_lay,
    t_lev,
    t_lay,
    h2o_lev,
    h2o_lay,
    qvapmin,
)
    thermo_state = recover_thermo_state(bl, prog, aux)
    p_lev[v, h] = air_pressure(thermo_state)
    t_lev[v, h] = air_temperature(thermo_state)
    qvap = relhum(FT, aux.coord[3]) * q_vap_saturation(thermo_state)
    if qvap < qvapmin
        qvapmin = qvap
    else
        qvap = qvapmin
    end
    h2o_lev[v, h] = vol_vapor_mixing_ratio(bl.param_set, PhasePartition(qvap))
    return qvapmin
end

function lev_avg!(FT, bl, prog, aux, v, h, p_lev, p_lay, t_lev, t_lay, qvapmin)
    thermo_state = recover_thermo_state(bl, prog, aux)
    p_lev[v, h] += air_pressure(thermo_state)
    p_lev[v, h] *= FT(0.5)
    t_lev[v, h] += air_temperature(thermo_state)
    t_lev[v, h] *= FT(0.5)
    return qvapmin
end

function lev_avg!(
    FT,
    bl,
    prog,
    aux,
    v,
    h,
    p_lev,
    p_lay,
    t_lev,
    t_lay,
    h2o_lev,
    h2o_lay,
    qvapmin,
)
    thermo_state = recover_thermo_state(bl, prog, aux)
    p_lev[v, h] += air_pressure(thermo_state)
    p_lev[v, h] *= FT(0.5)
    t_lev[v, h] += air_temperature(thermo_state)
    t_lev[v, h] *= FT(0.5)
    qvap = relhum(FT, aux.coord[3]) * q_vap_saturation(thermo_state)
    if qvap < qvapmin
        qvapmin = qvap
    else
        qvap = qvapmin
    end
    h2o_lev[v, h] += vol_vapor_mixing_ratio(bl.param_set, PhasePartition(qvap))
    h2o_lev[v, h] *= FT(0.5)
    return qvapmin
end

function lev_to_lay!(FT, v, h, p_lev, p_lay, t_lev, t_lay)
    p_lay[v, h] = (p_lev[v, h] + p_lev[v + 1, h]) * FT(0.5)
    t_lay[v, h] = (t_lev[v, h] + t_lev[v + 1, h]) * FT(0.5)
end

function lev_to_lay!(FT, v, h, p_lev, p_lay, t_lev, t_lay, h2o_lev, h2o_lay)
    p_lay[v, h] = (p_lev[v, h] + p_lev[v + 1, h]) * FT(0.5)
    t_lay[v, h] = (t_lev[v, h] + t_lev[v + 1, h]) * FT(0.5)
    h2o_lay[v, h] = (h2o_lev[v, h] + h2o_lev[v + 1, h]) * FT(0.5)
end

@kernel function kernel_load_into_rrtmgp!(
    balance_law::BalanceLaw,
    ::Val{Nq1}, ::Val{Nq2}, ::Val{1}, ::Val{nvertelem},
    state_prognostic::AbstractArray{FT},
    state_auxiliary,
    elems,
    rrtmgp_arrays...,
) where {Nq1, Nq2, nvertelem, FT}
    @uniform begin
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        prog = Vars{vars_state(balance_law, Prognostic(), FT)}(
            local_state_prognostic,
        )
        aux = Vars{vars_state(balance_law, Auxiliary(), FT)}(
            local_state_auxiliary,
        )
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)
    qvapmin = FT(Inf)

    @inbounds begin
        eh = elems[_eh]
        h = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i

        # Fill in the level data.
        for ev in 1:nvertelem
            e = nvertelem * (eh - 1) + ev
            ijk = Nq1 * (j - 1) + i
            @unroll for s in 1:num_state_prognostic
                local_state_prognostic[s] = state_prognostic[ijk, s, e]
            end
            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
            end
            qvapmin = lev_set!(
                FT,
                balance_law,
                prog,
                aux,
                ev,
                h,
                rrtmgp_arrays...,
                qvapmin,
            )
        end

        # Average the level data to obtain the layer data.
        for v in 1:nvertelem - 1
            lev_to_lay!(FT, v, h, rrtmgp_arrays...)
        end
    end
end

@kernel function kernel_load_into_rrtmgp!(
    balance_law::BalanceLaw,
    ::Val{Nq1}, ::Val{Nq2}, ::Val{Nq3}, ::Val{nvertelem},
    state_prognostic::AbstractArray{FT},
    state_auxiliary,
    elems,
    rrtmgp_arrays...,
) where {Nq1, Nq2, Nq3, nvertelem, FT}
    @uniform begin
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        prog = Vars{vars_state(balance_law, Prognostic(), FT)}(
            local_state_prognostic,
        )
        aux = Vars{vars_state(balance_law, Auxiliary(), FT)}(
            local_state_auxiliary,
        )
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)
    qvapmin = FT(Inf)

    @inbounds begin
        eh = elems[_eh]
        h = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i

        # Fill in data from the bottom element (ev = 1).
        e = nvertelem * (eh - 1) + 1
        @unroll for k in 1:Nq3
            ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
            @unroll for s in 1:num_state_prognostic
                local_state_prognostic[s] = state_prognostic[ijk, s, e]
            end
            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
            end
            qvapmin = lev_set!(
                FT,
                balance_law,
                prog,
                aux,
                k,
                h,
                rrtmgp_arrays...,
                qvapmin,
            )
        end

        # Fill in data from the remaining elements.
        for ev in 2:nvertelem
            e = nvertelem * (eh - 1) + ev

            # Average duplicate data from the bottom point (k = 1) of the
            # current element. The data from the top point (k = Nq3) of the
            # element below has already been filled in.
            ijk = Nq1 * (j - 1) + i
            @unroll for s in 1:num_state_prognostic
                local_state_prognostic[s] = state_prognostic[ijk, s, e]
            end
            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
            end
            qvapmin = lev_avg!(
                FT,
                balance_law,
                prog,
                aux,
                (Nq3 - 1) * (ev - 1) + 1,
                h,
                rrtmgp_arrays...,
                qvapmin,
            )

            # Fill in data from the remaining points.
            @unroll for k in 2:Nq3
                ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                @unroll for s in 1:num_state_prognostic
                    local_state_prognostic[s] = state_prognostic[ijk, s, e]
                end
                @unroll for s in 1:num_state_auxiliary
                    local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
                end
                qvapmin = lev_set!(
                    FT,
                    balance_law,
                    prog,
                    aux,
                    (Nq3 - 1) * (ev - 1) + k,
                    h,
                    rrtmgp_arrays...,
                    qvapmin,
                )
            end
        end

        # Average the level data to obtain the layer data.
        for v in 1:(Nq3 - 1) * nvertelem
            lev_to_lay!(FT, v, h, rrtmgp_arrays...)
        end
    end
end

@kernel function kernel_unload_from_rrtmgp!(
    ::Val{Nq1}, ::Val{Nq2}, ::Val{1}, ::Val{nvertelem}, ::Val{fluxindex},
    state_auxiliary,
    flux,
    elems,
) where {Nq1, Nq2, nvertelem, fluxindex}
    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]
        h = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i

        for ev in 1:nvertelem
            e = nvertelem * (eh - 1) + ev
            ijk = Nq1 * (j - 1) + i
            state_auxiliary[ijk, fluxindex, e] = flux[ev, h]
        end
    end
end

@kernel function kernel_unload_from_rrtmgp!(
    ::Val{Nq1}, ::Val{Nq2}, ::Val{Nq3}, ::Val{nvertelem}, ::Val{fluxindex},
    state_auxiliary,
    flux,
    elems,
) where {Nq1, Nq2, Nq3, nvertelem, fluxindex}
    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]
        h = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i

        for ev in 1:nvertelem
            e = nvertelem * (eh - 1) + ev
            @unroll for k in 1:Nq3
                ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                state_auxiliary[ijk, fluxindex, e] =
                    flux[(Nq3 - 1) * (ev - 1) + k, h]
            end
        end
    end
end

########################## Temporary Debugging Stuff ##########################

# Center-pads a string or number. The value of intdigits should be positive.
cpad(v::Array, args...) = '[' * join(cpad.(v, args...), ", ") * ']'
cpad(x::Any, padding) = cpad(string(x), padding)
function cpad(s::String, padding)
    d, r = divrem(padding - length(s), 2)
    return ' '^(d + r) * s * ' '^d
end
firstndecdigits(x::Real, n) = lpad(trunc(Int, (x - trunc(x)) * 10^n), n, '0')
function cpad(x::Real, intdigits, decdigits, expectnegatives = false)
    if !isfinite(x)
        p = Int(expectnegatives) + intdigits + Int(decdigits > 0) + decdigits
        return cpad(x, p)
    end
    sign = (signbit(x) ? '-' : (expectnegatives ? ' ' : ""))
    x = abs(x)
    if x >= 10^intdigits
        pow = floor(Int, log10(x))
        x *= 10.0^(-pow)
        pow = string(pow)
        newdecdigits =
            intdigits + Int(decdigits > 0) + decdigits - length(pow) - 3
        s = string(trunc(Int, x))
        if newdecdigits >= 0
            s *= '.'
            if newdecdigits > 0
                s *= firstndecdigits(x, newdecdigits)
            end
        end
        return sign * s * 'e' * pow
    else
        s = lpad(trunc(Int, x), intdigits)
        if decdigits > 0
            s *= '.' * firstndecdigits(x, decdigits)
        end
        return sign * s
    end
end

function debug_arrays(Q, dg)
    balance_law = dg.balance_law
    state_auxiliary = dg.state_auxiliary
    grid = dg.grid
    dim = dimensionality(grid)
    npoly = polynomialorders(grid)
    Nq1 = npoly[1] + 1
    Nq2 = dim == 2 ? 1 : npoly[2] + 1
    Nq3 = npoly[3] + 1
    topology = grid.topology
    elems = topology.realelems
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)

    nvert = Nq3 * nvertelem
    nhorz = Nq1 * Nq2 * length(horzelems)
    
    FT = eltype(Q)
    num_state_prognostic = number_states(balance_law, Prognostic())
    num_state_auxiliary = number_states(balance_law, Auxiliary())
    local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
    local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
    prog = Vars{vars_state(balance_law, Prognostic(), FT)}(
        local_state_prognostic,
    )
    aux = Vars{vars_state(balance_law, Auxiliary(), FT)}(
        local_state_auxiliary,
    )

    p_avg = Array{FT}(undef, nvert)
    t_avg = Array{FT}(undef, nvert)
    f_avg = Array{FT}(undef, nvert)
    h2o_avg = Array{FT}(undef, nvert)

    qvapmins = Array{FT}(undef, nhorz)
    qvapmins .= FT(Inf)

    if nhorz > 0
        for ev in 1:nvertelem
            for k in 1:Nq3
                v = Nq3 * (ev - 1) + k
                p_avg[v] = FT(0)
                t_avg[v] = FT(0)
                f_avg[v] = FT(0)
                h2o_avg[v] = FT(0)
                for eh in horzelems, i in 1:Nq1, j in 1:Nq2
                    e = nvertelem * (eh - 1) + ev
                    ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                    for s in 1:num_state_prognostic
                        local_state_prognostic[s] = Q[ijk, s, e]
                    end
                    for s in 1:num_state_auxiliary
                        local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
                    end
                    thermo_state = recover_thermo_state(balance_law, prog, aux)
                    p_avg[v] += air_pressure(thermo_state)
                    t_avg[v] += air_temperature(thermo_state)
                    f_avg[v] += aux.radiation.flux
                    qvap = relhum(FT, aux.coord[3]) *
                        q_vap_saturation(thermo_state)
                    h = Nq1 * (Nq2 * (eh - horzelems[1]) + (j - 1)) + i
                    if qvap < qvapmins[h]
                        qvapmins[h] = qvap
                    else
                        qvap = qvapmins[h]
                    end
                    h2o_avg[v] += vol_vapor_mixing_ratio(
                        balance_law.param_set,
                        PhasePartition(qvap),
                    )
                end
                p_avg[v] /= nhorz
                t_avg[v] /= nhorz
                f_avg[v] /= nhorz
                h2o_avg[v] /= nhorz
            end
        end
    end
    
    return p_avg, t_avg, f_avg, h2o_avg
end
function print_debug(Q, dg, t, rrtmgp_type)
    println("z($(cpad(t, 4, 1))) = $(cpad(get_z(dg.grid), 5, 2))")
    p_avg, t_avg, f_avg, h2o_avg = debug_arrays(Q, dg)
    println("P($(cpad(t, 4, 1))) = $(cpad(p_avg, 3, 4))")
    println("T($(cpad(t, 4, 1))) = $(cpad(t_avg, 3, 4))")
    println("F($(cpad(t, 4, 1))) = $(cpad(f_avg, 3, 4))")
    rrtmgp_type == :full &&
        println("vmr_H2O($(cpad(t, 4, 1))) = $(cpad(h2o_avg, 3, 4))")
end

const year_to_s = 60 * 60 * 24 * 365.25636
round_year(seconds) = round(seconds / year_to_s; digits = 2)
function time_string(x)
    x < 60 && return "$x s"
    x /= 60
    x < 59.995 && return "$(round(x; sigdigits = 4)) min"
    x /= 60
    x < 23.995 && return "$(round(x; sigdigits = 4)) hr"
    x /= 24
    x < 365.25 && return "$(round(x; sigdigits = 4)) days"
    x /= 365.25636
    x < 99.995 && return "$(round(x; sigdigits = 4)) yr"
    return "$(round(Int, x)) yr"
end

################################# Driver Code #################################

function init_to_ref_state!(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)
    state.ρ = aux.ref_state.ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.energy.ρe = aux.ref_state.ρe
    if bl.moisture isa EquilMoist
        state.moisture.ρq_tot = aux.ref_state.ρq_tot
    elseif bl.moisture isa NonEquilMoist
        state.moisture.ρq_tot = aux.ref_state.ρq_tot
        state.moisture.ρq_liq = aux.ref_state.ρq_liq
        state.moisture.ρq_ice = aux.ref_state.ρq_ice
    end
end

function driver_configuration(
    FT,
    temp_profile,
    numerical_flux,
    solver_type,
    config_type,
    domain_height,
    domain_width,
    polyorder_vert,
    polyorder_horz,
    nelem_vert,
    nelem_horz;
    compressibility = Anelastic1D(),
)
    model = AtmosModel{FT}(
        typeof(config_type),
        param_set;
        problem = AtmosProblem(
            # # Changing the boundary conditions increases instability.
            # boundaryconditions = (
            #     AtmosBC(
            #         energy =
            #             PrescribedTemperature((state, aux, t) -> temp_sfc),
            #     ),
            #     AtmosBC(),
            # ),
            init_state_prognostic = init_to_ref_state!,
        ),
        ref_state = HydrostaticState(
            temp_profile,
            FT(0); # relative humidity
            subtract_off = polyorder_vert == 0 ? false : true,
        ),
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
        radiation = RRTMGPModel(),
        source = compressibility isa Compressible ? (Gravity(),) : (),
        compressibility = compressibility,
    )

    if config_type isa SingleStackConfigType
        config_fun = ClimateMachine.SingleStackConfiguration
        config_args = (nelem_vert, domain_height, param_set, model)
        config_kwargs = (hmax = domain_width,)
    elseif config_type isa AtmosLESConfigType
        config_fun = ClimateMachine.AtmosLESConfiguration
        npoints_horz = nelem_horz * polyorder_horz
        npoints_vert =
            polyorder_vert == 0 ? nelem_vert : nelem_vert * polyorder_vert
        resolution = (
            domain_width / npoints_horz,
            domain_width / npoints_horz,
            domain_height / npoints_vert,
        )
        config_args = (
            resolution,
            domain_width, domain_width, domain_height,
            param_set,
            nothing,
        )
        config_kwargs = (model = model,)
    else # config_type isa AtmosGCMConfigType
        config_fun = ClimateMachine.AtmosGCMConfiguration
        config_args =
            ((nelem_horz, nelem_vert), domain_height, param_set, nothing)
        config_kwargs = (model = model,)
    end

    return config_fun(
        "balanced state",
        (polyorder_horz, polyorder_vert),
        config_args...;
        config_kwargs...,
        solver_type = solver_type,
        numerical_flux_first_order = numerical_flux,
        fv_reconstruction = HBFVReconstruction(model, FVLinear()),
    )
end

function solve_and_plot!(
    driver_config,
    FT,
    rrtmgp_type,
    timestart,
    timeend,
    timestep,
)
    grid = driver_config.grid

    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        ode_dt = timestep,
        Courant_number = FT(0.5), # irrelevant; lets us use LSRKEulerMethod
        timeend_dt_adjust = false, # prevents dt from getting modified
        modeldata = (RRTMGPdata = RRTMGPData(grid, rrtmgp_type),),
    )

    solver = solver_config.solver
    dg = solver_config.dg
    prog = solver_config.Q
    gradflux = dg.state_gradient_flux
    progvars = vars_state(driver_config.bl, Prognostic(), FT)
    gradfluxvars = vars_state(driver_config.bl, GradientFlux(), FT)

    max_outputs = 11
    elem_stack_avgs = []
    elem_stack_vars = []
    # elem_stack_vals = []
    debug_array_avgs = []
    times = []
    merged_dicts(f) =
        merge(f(grid, prog, progvars), f(grid, gradflux, gradfluxvars))
    function push_elem_stack_data()
        push!(elem_stack_avgs, merged_dicts(get_horizontal_mean))
        push!(elem_stack_vars, merged_dicts(get_horizontal_variance))
        # push!(elem_stack_vals, merged_dicts(get_vars_from_nodal_stack))
        push!(debug_array_avgs, debug_arrays(prog, dg))
        push!(times, gettime(solver))
    end
    push_elem_stack_data()
    cb_diagnostics = GenericCallbacks.EveryXSimulationSteps(
        push_elem_stack_data,
        max(1, solver_config.numberofsteps ÷ (max_outputs - 1)),
    )

    # cb_print = GenericCallbacks.EveryXSimulationSteps(1) do 
    #     print_debug(prog, dg, gettime(solver), rrtmgp_type)
    # end
    # print_debug(prog, dg, gettime(solver), rrtmgp_type)
    
    progress = Progress(solver_config.numberofsteps)
    showvalues = () -> [(:simtime, gettime(solver)), (:normQ, norm(prog))]
    cb_progress = GenericCallbacks.EveryXWallTimeSeconds(0.3, prog.mpicomm) do
        update!(progress, getsteps(solver); showvalues = showvalues)
    end
    cb_progress_end = GenericCallbacks.AtInitAndFini() do
        getsteps(solver) == solver_config.numberofsteps && finish!(progress)
    end
    ClimateMachine.invoke!(
        solver_config;
        user_callbacks = (cb_diagnostics, cb_progress, cb_progress_end),
    )

    zs = get_z(grid; z_scale = 1e-3)
    for (vars, var_label, filename_prefix, skip_first) in (
        # (("ρ",), "density [kg/m^3]", "ρ", false),
        # (
        #     ("ρu[1]", "ρu[2]"),
        #     "momentum density's horizontal components [kg/s/m^2]",
        #     "ρuh", false,
        # ),
        # (
        #     ("ρu[3]",),
        #     "momentum density's vertical component [kg/s/m^2]",
        #     "ρuv", false,
        # ),
        (("energy.ρe",), "energy density [J/m^3]", "ρe", false),
        # (("moisture.ρq_tot",), "moisture density [kg/m^3]", "ρq", false),
        # (("moisture.ρq_liq",), "liq density [kg/m^3]", "ρql", false),
        # (("moisture.ρq_ice",), "ice density [kg/m^3]", "ρqi", false),
        # (
        #     ("energy.∇h_tot[1]", "energy.∇h_tot[2]"),
        #     "specific enthalpy gradient's horizontal components [J/kg/m]",
        #     "∇htoth", true,
        # ),
        (
            ("energy.∇h_tot[3]",),
            "specific enthalpy gradient's vertical component [J/kg/m]",
            "∇htotv", true,
        ),
    )
        for (data, data_label, filename_suffix) in (
            (elem_stack_avgs, "Horizontal average", "avg"),
            (elem_stack_vars, "Horizontal variance", "var"),
            # (elem_stack_vals, "Value at southwest node", "val"),
        )
            plot(
                xlabel = "$data_label of $var_label",
                ylabel = "z [km]",
                legend = :outertopright,
            )
            for var in vars, (index, time) in enumerate(times)
                index == 1 && skip_first && continue
                legend_entry = "$(round_year(time)) yr"
                length(vars) > 1 && (legend_entry = "$var, $legend_entry")
                plot!(data[index][var], zs, label = legend_entry)
            end
            savefig(joinpath(
                @__DIR__,
                "$(filename_prefix)_$(filename_suffix).png",
            ))
        end
    end
    for (array_num, var_label, filename_prefix) in (
        # (1, "pressure [Pa]", "P"),
        (2, "temperature [K]", "T"),
        (3, "net upward radiation energy flux [J/s/m^2]", "Fnet"),
        (4, "water vapor volume mixing ratio []", "h2o"),
    )
        plot(
            xlabel = "Horizontal average of $var_label",
            ylabel = "z [km]",
            legend = :outertopright,
        )
        for (index, time) in enumerate(times)
            legend_entry = "$(round_year(time)) yr"
            plot!(debug_array_avgs[index][array_num], zs, label = legend_entry)
        end
        savefig(joinpath(@__DIR__, "$(filename_prefix)_avg.png"))
    end

    return prog
end

function main()
    FT = Float64

    # max timesteps rounded down to multiples of 5:
    #     gray model: (0, 4, 120) - 85 hr; (4, 4, 24) - 35 hr
    #     full model: (0, 4, 120) - 80 hr; (4, 4, 24) - 35 hr

    rrtmgp_type = :full
    timestart = FT(0)
    timeend = FT(year_to_s * 1)
    timestep = FT(35 * 60 * 60)
    domain_height = FT(70e3)
    domain_width = FT(70) # Ignored for AtmosGCMConfigType
    polyorder_vert = 4
    polyorder_horz = 4
    nelem_vert = 24
    nelem_horz = 1 # Ignored for SingleStackConfigType

    @testset for config_type in (
        SingleStackConfigType(),
        # AtmosLESConfigType(),
        # AtmosGCMConfigType(),
    )
        @testset for solver_type in (
            ExplicitSolverType(; solver_method = LSRKEulerMethod),
            # ExplicitSolverType(; solver_method = LSRK54CarpenterKennedy),
            # ExplicitSolverType(; solver_method = LSRK144NiegemannDiehlBusch),
            # HEVISolverType(FT; solver_method = ARK2ImplicitExplicitMidpoint),
            # HEVISolverType(FT; solver_method = ARK548L2SA2KennedyCarpenter),
        )
            @testset for numerical_flux in (
                # CentralNumericalFluxFirstOrder(),
                RusanovNumericalFlux(),
                # RoeNumericalFlux(),
                # HLLCNumericalFlux(),
            )
                @testset for temp_profile in (
                    # IsothermalProfile(param_set, FT),
                    DecayingTemperatureProfile{FT}(param_set),
                )
                    driver_config = driver_configuration(
                        FT,
                        temp_profile,
                        numerical_flux,
                        solver_type,
                        config_type,
                        domain_height,
                        domain_width,
                        polyorder_vert,
                        polyorder_horz,
                        nelem_vert,
                        nelem_horz,
                    )

                    prog = solve_and_plot!(
                        driver_config,
                        FT,
                        rrtmgp_type,
                        timestart,
                        timeend,
                        timestep,
                    )

                    @test all(isfinite.(prog.data))
                end
            end
        end
    end
end

main()