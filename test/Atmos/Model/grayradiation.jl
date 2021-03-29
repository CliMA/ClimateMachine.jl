using StaticArrays
using LinearAlgebra
using ProgressMeter
using Plots
using Test
using NLsolve

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
using ClimateMachine.Topologies: SingleExponentialStretching
using ClimateMachine.Mesh.Elements: lglpoints
using ClimateMachine.Mesh.Filters: apply!, BoydVandevenFilter

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

####################### Adding to Preexisting Interface #######################

using UnPack
using ClimateMachine.BalanceLaws: AbstractEnergy, Flux, FirstOrder,
    SecondOrder, Σfluxes, Auxiliary, TendencyDef, Source,
    UpwardIntegrals, BalanceLaw
using ClimateMachine.DGMethods: SpaceDiscretization
using ClimateMachine.Orientations: vertical_unit_vector
using ClimateMachine.Atmos: Energy, nodal_update_auxiliary_state!
import ClimateMachine.BalanceLaws: eq_tends, vars_state, flux, prognostic_vars,
    source
import ClimateMachine.Atmos: atmos_energy_normal_boundary_flux_second_order!,
    update_auxiliary_state!

# Allow the RadiationModel to use a second-order flux for an Insulating bc.
eq_tends(pv, ::RadiationModel, tt) = ()
eq_tends(pv::AbstractEnergy, m::AtmosModel, tt::Flux{SecondOrder}) = (
    eq_tends(pv, m.energy, tt)...,
    eq_tends(pv, m.turbconv, tt)...,
    eq_tends(pv, m.hyperdiffusion, tt)...,
    eq_tends(pv, m.radiation, tt)..., # This line is new.
)
function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::Insulating,
    atmos,
    fluxᵀn,
    args,
)
    @unpack state⁻, aux⁻, t, n⁻ = args
    map(prognostic_vars(atmos.energy)) do prog
        fluxᵀn.energy.ρe += dot(n⁻, Σfluxes(
            prog,
            eq_tends(prog, atmos.radiation, Flux{SecondOrder}()),
            atmos,
            (; state = state⁻, aux = aux⁻, t),
        ))
    end
end

# Define a new type of radiation model that utilizes RRTMGP.
abstract type RRTMGPModel <: RadiationModel end
struct RRTMGPModelF1 <: RRTMGPModel end
struct RRTMGPModelF2 <: RRTMGPModel end
struct RRTMGPModelS <: RRTMGPModel end

# Allocate space in which to unload the energy fluxes calculated by RRTMGP.
vars_state(::RRTMGPModel, ::Auxiliary, FT) = @vars(flux::FT)
vars_state(::RRTMGPModelS, ::Auxiliary, FT) = @vars(flux::FT, div_flux::FT)

# Define a new type of tendency that adds those fluxes to the net energy flux.
struct Radiation{T} <: TendencyDef{T} end
eq_tends(::Energy, ::RRTMGPModelF1, ::Flux{FirstOrder}) =
    (Radiation{Flux{FirstOrder}}(),)
eq_tends(::Energy, ::RRTMGPModelF2, ::Flux{SecondOrder}) =
    (Radiation{Flux{SecondOrder}}(),)
function flux(::Energy, ::Radiation, bl, args)
    @unpack aux = args
    return aux.radiation.flux * vertical_unit_vector(bl, aux)
end
Radiation() = Radiation{Source}()
prognostic_vars(::Radiation) = (Energy(),)
function source(::Energy, ::Radiation, bl, args)
    @unpack aux = args
    return aux.radiation.div_flux
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
    q_vap_saturation, PhasePartition, vol_vapor_mixing_ratio,
    vapor_specific_humidity

const nstreams = 1
const ngpoints = 1
const ngaussangles = 1

const temp_sfc = 290 # Surface temperature; used by both gray and full optics
const temp_toa = 200 # Top-of-atmosphere temperature; only used by gray optics
const max_relhum = 0.6 # Maximum relative humidity; only used by full optics

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
    ds_lw = Dataset(joinpath(@__DIR__, "rrtmgp-data-lw-g256-2018-12-04.nc"))
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
    radiation_model::RRTMGPModel,
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
            parameter_set(m),
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

    if radiation_model isa RRTMGPModelS
        event = Event(device)
        event = kernel_flux_divergence!(device, (Nq[1], Nq[2]))(
            m,
            map(Val, Nq)...,
            Val(nvertelem),
            Val(varsindex(vars(spacedisc.state_auxiliary), :radiation, :flux)[1]),
            Val(varsindex(vars(spacedisc.state_auxiliary), :radiation, :div_flux)[1]),
            spacedisc.state_auxiliary.data,
            horzelems,
            grid;
            ndrange = (length(horzelems) * Nq[1], Nq[2]),
            dependencies = (event,),
        )
        wait(device, event)
    end
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

Base.@propagate_inbounds function lev_set!(
    bl,
    prog,
    aux,
    v,
    h,
    p_lev,
    p_lay,
    t_lev,
    t_lay,
    qvapmin,
)
    thermo_state = recover_thermo_state(bl, prog, aux)
    p_lev[v, h] = air_pressure(thermo_state)
    t_lev[v, h] = air_temperature(thermo_state)
    return qvapmin
end

Base.@propagate_inbounds function lev_set!(
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
    qvapmin::FT,
) where {FT}
    thermo_state = recover_thermo_state(bl, prog, aux)
    p_lev[v, h] = air_pressure(thermo_state)
    t_lev[v, h] = air_temperature(thermo_state)
    qvap = FT(max_relhum) * q_vap_saturation(thermo_state)
    if aux.coord[3] > FT(10000)
        if qvap < qvapmin
            qvapmin = qvap
        else
            qvap = qvapmin
        end
    end
    h2o_lev[v, h] =
        vol_vapor_mixing_ratio(parameter_set(bl), PhasePartition(qvap))
    return qvapmin
end

Base.@propagate_inbounds function lev_avg!(
    bl,
    prog,
    aux,
    v,
    h,
    p_lev,
    p_lay,
    t_lev,
    t_lay,
    qvapmin::FT,
) where {FT}
    thermo_state = recover_thermo_state(bl, prog, aux)
    p_lev[v, h] = (p_lev[v, h] + air_pressure(thermo_state)) * FT(0.5)
    t_lev[v, h] = (t_lev[v, h] + air_temperature(thermo_state)) * FT(0.5)
    return qvapmin
end

Base.@propagate_inbounds function lev_avg!(
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
    qvapmin::FT,
) where {FT}
    thermo_state = recover_thermo_state(bl, prog, aux)
    p_lev[v, h] = (p_lev[v, h] + air_pressure(thermo_state)) * FT(0.5)
    t_lev[v, h] = (t_lev[v, h] + air_temperature(thermo_state)) * FT(0.5)
    qvap = FT(max_relhum) * q_vap_saturation(thermo_state)
    if aux.coord[3] > FT(10000)
        if qvap < qvapmin
            qvapmin = qvap
        else
            qvap = qvapmin
        end
    end
    h2o_lev[v, h] = (
        h2o_lev[v, h] +
        vol_vapor_mixing_ratio(parameter_set(bl), PhasePartition(qvap))
    ) * FT(0.5)
    return qvapmin
end

Base.@propagate_inbounds function lev_to_lay!(
    v,
    h,
    p_lev::AbstractArray{FT},
    p_lay,
    t_lev,
    t_lay,
) where {FT}
    p_lay[v, h] = (p_lev[v, h] + p_lev[v + 1, h]) * FT(0.5)
    t_lay[v, h] = (t_lev[v, h] + t_lev[v + 1, h]) * FT(0.5)
end

Base.@propagate_inbounds function lev_to_lay!(
    v,
    h,
    p_lev::AbstractArray{FT},
    p_lay,
    t_lev,
    t_lay,
    h2o_lev,
    h2o_lay,
) where {FT}
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
            lev_to_lay!(v, h, rrtmgp_arrays...)
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
            lev_to_lay!(v, h, rrtmgp_arrays...)
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

@kernel function kernel_flux_divergence!(
    balance_law::BalanceLaw,
    ::Val{Nq1}, ::Val{Nq2}, ::Val{1}, ::Val{nvertelem}, ::Val{fluxindex},
    ::Val{divfluxindex},
    state_auxiliary::AbstractArray{FT},
    elems,
    grid,
) where {Nq1, Nq2, nvertelem, fluxindex, divfluxindex, FT}
    @uniform begin
        vgeo = grid.vgeo
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]

        for ev in 1:nvertelem
            e = nvertelem * (eh - 1) + ev
            ijk = Nq1 * (j - 1) + i

            error("Source radiation not implemented for FV")

            # if ev == 1
            #     ΔF⁺ = state_auxiliary[ijk, fluxindex, e + 1] -
            #         state_auxiliary[ijk, fluxindex, e]
            #     Δz⁺ = vgeo[ijk, Grids._x3, e + 1] - vgeo[ijk, Grids._x3, e]
            #     state_auxiliary[ijk, divfluxindex, e] = ΔF⁺/Δz⁺
            # elseif ev == nvertelem
            #     ΔF⁻ = state_auxiliary[ijk, fluxindex, e] -
            #         state_auxiliary[ijk, fluxindex, e - 1]
            #     Δz⁻ = vgeo[ijk, Grids._x3, e] - vgeo[ijk, Grids._x3, e - 1]
            #     state_auxiliary[ijk, divfluxindex, e] = ΔF⁻/Δz⁻
            # else
            #     ΔF⁻ = state_auxiliary[ijk, fluxindex, e] -
            #         state_auxiliary[ijk, fluxindex, e - 1]
            #     ΔF⁺ = state_auxiliary[ijk, fluxindex, e + 1] -
            #         state_auxiliary[ijk, fluxindex, e]
            #     Δz⁻ = vgeo[ijk, Grids._x3, e] - vgeo[ijk, Grids._x3, e - 1]
            #     Δz⁺ = vgeo[ijk, Grids._x3, e + 1] - vgeo[ijk, Grids._x3, e]
            #     state_auxiliary[ijk, divfluxindex, e] =
            #         Δz⁺/(Δz⁻ + Δz⁺) * ΔF⁻/Δz⁻ + Δz⁻/(Δz⁻ + Δz⁺) * ΔF⁺/Δz⁺
            # end
        end
    end
end

@kernel function kernel_flux_divergence!(
    balance_law::BalanceLaw,
    ::Val{Nq1}, ::Val{Nq2}, ::Val{Nq3}, ::Val{nvertelem}, ::Val{fluxindex},
    ::Val{divfluxindex},
    state_auxiliary::AbstractArray{FT},
    elems,
    grid,
) where {Nq1, Nq2, Nq3, nvertelem, fluxindex, divfluxindex, FT}
    @uniform begin
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        aux = Vars{vars_state(balance_law, Auxiliary(), FT)}(
            local_state_auxiliary,
        )
        vgeo = grid.vgeo
        D = grid.D[3]
        sgeo = grid.sgeo
        elemtobndy = grid.elemtobndy
        vmap⁻ = grid.vmap⁻
        vmap⁺ = grid.vmap⁺
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]

        for ev in 1:nvertelem
            e = nvertelem * (eh - 1) + ev

            @unroll for k in 1:Nq3
                ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
                state_auxiliary[ijk, divfluxindex, e] = zero(FT)
            end

            @unroll for k in 1:Nq3
                ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                M = vgeo[ijk, Grids._M, e]
                ζx1 = vgeo[ijk, Grids._ξ3x1, e]
                ζx2 = vgeo[ijk, Grids._ξ3x2, e]
                ζx3 = vgeo[ijk, Grids._ξ3x3, e]
                @unroll for s in 1:num_state_auxiliary
                    local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
                end
                zhat = vertical_unit_vector(balance_law, aux)
                F1, F2, F3 = state_auxiliary[ijk, fluxindex, e] * zhat
                Fv = M * (ζx1 * F1 + ζx2 * F2 + ζx3 * F3)
                @unroll for n in 1:Nq3
                    ijn = Nq1 * (Nq2 * (n - 1) + (j - 1)) + i
                    MI = vgeo[ijn, Grids._MI, e]
                    state_auxiliary[ijn, divfluxindex, e] += MI * D[k, n] * Fv
                end

                # if ev == 1 && k == 1
                #     ijk⁺ = Nq1 * (Nq2 * k + (j - 1)) + i
                #     e⁺ = e
                #     ΔF⁺ = state_auxiliary[ijk⁺, fluxindex, e⁺] -
                #         state_auxiliary[ijk, fluxindex, e]
                #     Δz⁺ = state_auxiliary[ijk⁺, zindex, e⁺] -
                #         state_auxiliary[ijk, zindex, e]
                #     state_auxiliary[ijk, divfluxindex, e] = ΔF⁺/Δz⁺
                # elseif ev == nvertelem && k == Nq3
                #     ijk⁻ = Nq1 * (Nq2 * (k - 2) + (j - 1)) + i
                #     e⁻ = e
                #     ΔF⁻ = state_auxiliary[ijk, fluxindex, e] -
                #         state_auxiliary[ijk⁻, fluxindex, e⁻]
                #     Δz⁻ = state_auxiliary[ijk, zindex, e] -
                #         state_auxiliary[ijk⁻, zindex, e⁻]
                #     state_auxiliary[ijk, divfluxindex, e] = ΔF⁻/Δz⁻
                # else
                #     if k == 1
                #         ijk⁻ = Nq1 * (Nq2 * (Nq3 - 2) + (j - 1)) + i
                #         e⁻ = e - 1
                #     else
                #         ijk⁻ = Nq1 * (Nq2 * (k - 2) + (j - 1)) + i
                #         e⁻ = e
                #     end
                #     if k == Nq3
                #         ijk⁺ = Nq1 * (Nq2 * 1 + (j - 1)) + i
                #         e⁺ = e + 1
                #     else
                #         ijk⁺ = Nq1 * (Nq2 * k + (j - 1)) + i
                #         e⁺ = e
                #     end
                #     ΔF⁻ = state_auxiliary[ijk, fluxindex, e] -
                #         state_auxiliary[ijk⁻, fluxindex, e⁻]
                #     ΔF⁺ = state_auxiliary[ijk⁺, fluxindex, e⁺] -
                #         state_auxiliary[ijk, fluxindex, e]
                #     Δz⁻ = state_auxiliary[ijk, zindex, e] -
                #         state_auxiliary[ijk⁻, zindex, e⁻]
                #     Δz⁺ = state_auxiliary[ijk⁺, zindex, e⁺] -
                #         state_auxiliary[ijk, zindex, e]
                #     state_auxiliary[ijk, divfluxindex, e] =
                #         Δz⁺/(Δz⁻ + Δz⁺) * ΔF⁻/Δz⁻ + Δz⁻/(Δz⁻ + Δz⁺) * ΔF⁺/Δz⁺
                # end
            end

            @unroll for f in (5, 6)
                n = Nq1 * (j - 1) + i
                normal_vector = SVector(
                    sgeo[Grids._n1, n, f, e],
                    sgeo[Grids._n2, n, f, e],
                    sgeo[Grids._n3, n, f, e],
                )
                sM = sgeo[Grids._sM, n, f, e]
                vMI = sgeo[Grids._vMI, n, f, e]
                bctag = elemtobndy[f, e]
                Np = Nq1 * Nq2 * Nq3
                e⁺ = bctag != 0 ? e : ((vmap⁺[n, f, e] - 1) ÷ Np) + 1
                vid = ((vmap⁻[n, f, e] - 1) % Np) + 1
                vid⁺ = bctag != 0 ? vid : ((vmap⁺[n, f, e] - 1) % Np) + 1
                flux_norm = state_auxiliary[vid, fluxindex, e]
                @unroll for s in 1:num_state_auxiliary
                    local_state_auxiliary[s] = state_auxiliary[vid, s, e]
                end
                flux = flux_norm * vertical_unit_vector(balance_law, aux)
                @unroll for s in 1:num_state_auxiliary
                    local_state_auxiliary[s] = state_auxiliary[vid⁺, s, e⁺]
                end
                flux⁺ = flux_norm * vertical_unit_vector(balance_law, aux)
                state_auxiliary[vid, divfluxindex, e] -= vMI * sM *
                    ((flux + flux⁺)' * (normal_vector / FT(2)))
            end
        end
    end
end

########################### Custom Energy Filtering ###########################

using ClimateMachine.Mesh.Filters: AbstractFilterTarget
using ClimateMachine.Orientations: gravitational_potential
using ClimateMachine.Thermodynamics: PhaseDry_ρp, PhaseEquil_ρpq, total_energy
import ClimateMachine.Mesh.Filters: vars_state_filtered,
    compute_filter_argument!, compute_filter_result!

struct EnergyPerturbation{M} <: AbstractFilterTarget
    atmos::M
end

vars_state_filtered(target::EnergyPerturbation, FT) = @vars(e_tot::FT)

ref_thermo_state(atmos::AtmosModel, aux::Vars) =
    ref_thermo_state(atmos, aux, atmos.moisture)
ref_thermo_state(atmos::AtmosModel, aux::Vars, ::DryModel) =
    PhaseDry_ρp(
        parameter_set(atmos),
        aux.ref_state.ρ,
        aux.ref_state.p,
    )
ref_thermo_state(atmos::AtmosModel, aux::Vars, ::Any) =
    PhaseEquil_ρpq(
        parameter_set(atmos),
        aux.ref_state.ρ,
        aux.ref_state.p,
        aux.ref_state.ρq_tot / aux.ref_state.ρ,
    )

function compute_filter_argument!(
    target::EnergyPerturbation,
    filter_state::Vars,
    state::Vars,
    aux::Vars,
)
    filter_state.e_tot = state.energy.ρe / state.ρ
    filter_state.e_tot -= total_energy(
        zero(eltype(aux)),
        gravitational_potential(target.atmos, aux),
        ref_thermo_state(target.atmos, aux),
    )
end

function compute_filter_result!(
    target::EnergyPerturbation,
    state::Vars,
    filter_state::Vars,
    aux::Vars,
)
    filter_state.e_tot += total_energy(
        zero(eltype(aux)),
        gravitational_potential(target.atmos, aux),
        ref_thermo_state(target.atmos, aux),
    )
    state.energy.ρe = state.ρ * filter_state.e_tot
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

function get_horizontal_mean_debug(Q, dg, rm_dupes)
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

    nvert = rm_dupes ? (Nq3 - 1) * nvertelem + 1 : Nq3 * nvertelem
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

    qvapmins = Array{FT}(undef, nhorz)
    dict = Dict(
        "P" => Array{FT}(undef, nvert),
        "T" => Array{FT}(undef, nvert),
        "F" => Array{FT}(undef, nvert),
        "qvap_sat" => Array{FT}(undef, nvert),
        "qvap" => Array{FT}(undef, nvert),
        "rel_hum" => Array{FT}(undef, nvert),
        "vmr_H2O" => Array{FT}(undef, nvert),
    )

    qvapmins .= FT(Inf)
    for array in values(dict)
        array .= FT(0)
    end

    if nhorz > 0
        for ev in 1:nvertelem
            for k in 1:Nq3
                v = rm_dupes ? (Nq3 - 1) * (ev - 1) + k : Nq3 * (ev - 1) + k
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
                    dict["P"][v] += air_pressure(thermo_state)
                    dict["T"][v] += air_temperature(thermo_state)
                    dict["F"][v] += aux.radiation.flux
                    qvap = FT(max_relhum) * q_vap_saturation(thermo_state)
                    h = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i
                    if aux.coord[3] > FT(10000)
                        if qvap < qvapmins[h]
                            qvapmins[h] = qvap
                        else
                            qvap = qvapmins[h]
                        end
                    end
                    dict["qvap_sat"][v] += q_vap_saturation(thermo_state)
                    dict["qvap"][v] += qvap
                    dict["rel_hum"][v] += qvap / q_vap_saturation(thermo_state)
                    dict["vmr_H2O"][v] += vol_vapor_mixing_ratio(
                        parameter_set(balance_law),
                        PhasePartition(qvap),
                    )
                end
                for array in values(dict)
                    if rm_dupes && ev > 1 && k == 1
                        array[v] /= 2 * nhorz
                    elseif !(rm_dupes && ev < nvertelem && k == Nq3)
                        array[v] /= nhorz
                    end
                end
            end
        end
    end

    @assert(dict["F"] ≈ get_horizontal_mean(
        grid,
        state_auxiliary,
        vars_state(balance_law, Auxiliary(), FT);
        interp = rm_dupes,
    )["radiation.flux"])
    
    return dict
end
function print_debug(Q, dg, t, optics_symbol, rm_dupes)
    zs = get_z(dg.grid; rm_dupes = rm_dupes)
    println("z($(cpad(t, 4, 1))) = $(cpad(zs, 5, 2))")
    dict = get_horizontal_mean_debug(Q, dg, rm_dupes)
    println("P($(cpad(t, 4, 1))) = $(cpad(dict["P"], 3, 4))")
    println("T($(cpad(t, 4, 1))) = $(cpad(dict["T"], 3, 4))")
    println("F($(cpad(t, 4, 1))) = $(cpad(dict["F"], 3, 4))")
    if optics_symbol == :full
        println("vmr_H2O($(cpad(t, 4, 1))) = $(cpad(h2o_avg, 3, 4))")
    end
end

const year_to_s = 60 * 60 * 24 * 365.25636
function time_string(x)
    x < 60 && return "$(round(x; sigdigits = 3)) s"
    x /= 60
    x < 60 && return "$(round(x; sigdigits = 3)) min"
    x /= 60
    x < 24 && return "$(round(x; sigdigits = 3)) hr"
    x /= 24
    x < 99.95 && return "$(round(x; sigdigits = 3)) days"
    x < 365.25636 && return "$(round(Int, x)) days"
    x /= 365.25636
    x < 99.95 && return "$(round(x; sigdigits = 3)) yr"
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
    moisture,
    radiation,
    numerical_flux,
    solver_type,
    config_type,
    domain_height,
    domain_width,
    polyorder_vert,
    polyorder_horz,
    nelem_vert,
    nelem_horz,
    stretch;
    compressibility = Anelastic1D(),
)
    source = ()
    compressibility isa Compressible && (source = (source..., Gravity()))
    radiation isa RRTMGPModelS && (source = (source..., Radiation()))
    model = AtmosModel{FT}(
        typeof(config_type),
        param_set;
        init_state_prognostic = init_to_ref_state!,
        ref_state = HydrostaticState(
            temp_profile,
            FT(max_relhum);
            subtract_off = polyorder_vert == 0 ? false : true,
        ),
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = moisture,
        radiation = radiation,
        source = source,
        compressibility = compressibility,
    )

    if config_type isa SingleStackConfigType
        config_fun = ClimateMachine.SingleStackConfiguration
        config_args = (nelem_vert, domain_height, param_set, model)
        config_kwargs = (hmax = domain_width,)
        nelem_horz == 1 || @warn "Only 1 element stack for SingleStack config"
        stretch == 0 || @warn "No grid stretching for SingleStack config"
    elseif config_type isa AtmosLESConfigType
        config_fun = ClimateMachine.AtmosLESConfiguration
        resolution = (
            domain_width / (nelem_horz * polyorder_horz),
            domain_width / (nelem_horz * polyorder_horz),
            domain_height / (nelem_vert * max(polyorder_vert, 1)),
        )
        config_args = (
            resolution,
            domain_width, domain_width, domain_height,
            param_set,
            nothing,
        )
        stretching_vert =
            stretch == 0 ? nothing : SingleExponentialStretching(stretch)
        config_kwargs = (
            model = model,
            grid_stretching = (nothing, nothing, stretching_vert),
        )
    else # config_type isa AtmosGCMConfigType
        config_fun = ClimateMachine.AtmosGCMConfiguration
        config_args =
            ((nelem_horz, nelem_vert), domain_height, param_set, nothing)
        stretching_vert =
            stretch == 0 ? nothing : SingleExponentialStretching(stretch)
        config_kwargs = (model = model, grid_stretching = (stretching_vert,))
        isnothing(domain_width) || @warn "No domain width for GCM config"
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

function setup_solver(
    driver_config,
    FT,
    optics_symbol,
    timestart,
    timeend,
    substep_duration,
    substeps_per_step,
)
    grid = driver_config.grid
    bl = driver_config.bl
    polyorder_vert = driver_config.polyorders[2]

    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        ode_dt = substep_duration,
        Courant_number = FT(0.5), # irrelevant; lets us use LSRKEulerMethod
        diffdir = VerticalDirection(),
        direction = VerticalDirection(),
        timeend_dt_adjust = false, # prevents dt from getting modified
        modeldata = (RRTMGPdata = RRTMGPData(grid, optics_symbol),),
    )

    solver = solver_config.solver
    Q = solver_config.Q
    state_auxiliary = solver_config.dg.state_auxiliary

    function cb_set_timestep()
        if getsteps(solver) == substeps_per_step
            updatedt!(solver, substeps_per_step * substep_duration)
        end
    end

    cb_filter = GenericCallbacks.EveryXSimulationSteps(1) do
        apply!(
            Q,
            EnergyPerturbation(bl),
            grid,
            BoydVandevenFilter(grid, 1, 4);
            state_auxiliary = state_auxiliary,
            direction = VerticalDirection(),
        )
    end
    
    progress = Progress(round(Int, timeend))
    showvalues = () -> [(:simtime, gettime(solver)), (:normQ, norm(Q))]
    cb_progress = GenericCallbacks.EveryXWallTimeSeconds(0.3, Q.mpicomm) do
        update!(progress, round(Int, gettime(solver)); showvalues = showvalues)
    end
    function cb_progress_finish() # needs to get run before @sprintf in invoke!
        gettime(solver) >= timeend && update!(progress, round(Int, timeend))
    end

    return solver_config, (
        cb_set_timestep,
        cb_filter,
        cb_progress,
        cb_progress_finish,
    )
end

function solve_and_plot_detailed!(
    driver_config,
    FT,
    optics_symbol,
    timestart,
    timeend,
    substep_duration,
    substeps_per_step,
    rm_dupes,
)
    elem_stack_avgs = []
    elem_stack_vars = []
    # elem_stack_vals = []
    elem_stack_debug_avgs = []
    times = []

    solver_config, callbacks = setup_solver(
        driver_config,
        FT,
        optics_symbol,
        timestart,
        timeend,
        substep_duration,
        substeps_per_step,
    )

    dg = solver_config.dg
    grid = dg.grid
    prog = solver_config.Q
    progvars = vars_state(dg.balance_law, Prognostic(), eltype(prog))
    gradflux = dg.state_gradient_flux
    gradfluxvars = vars_state(dg.balance_law, GradientFlux(), eltype(prog))
    solver = solver_config.solver
    merged_dicts(f) = merge(
        f(grid, prog, progvars; interp = rm_dupes),
        f(grid, gradflux, gradfluxvars; interp = rm_dupes),
    )
    function push_diagnostics_data()
        push!(elem_stack_avgs, merged_dicts(get_horizontal_mean))
        push!(elem_stack_vars, merged_dicts(get_horizontal_variance))
        # push!(elem_stack_vals, merged_dicts(get_vars_from_nodal_stack))
        push!(
            elem_stack_debug_avgs,
            get_horizontal_mean_debug(prog, dg, rm_dupes),
        )
        push!(times, gettime(solver))
    end
    next_push_step = 1
    function cb_diagnostics()
        if getsteps(solver) - substeps_per_step + 1 == next_push_step
            push_diagnostics_data()
            next_push_step *= 2
        end
    end

    push_diagnostics_data()
    ClimateMachine.invoke!(
        solver_config;
        user_callbacks = (callbacks..., cb_diagnostics),
    )
    push_diagnostics_data()

    zs = get_z(grid; z_scale = 1e-3, rm_dupes = rm_dupes)
    for (names, label, filename, skip_first) in (
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
        # (
        #     ("energy.∇h_tot[1]", "energy.∇h_tot[2]"),
        #     "specific enthalpy gradient's horizontal components [J/kg/m]",
        #     "∇htoth", true,
        # ),
        # (
        #     ("energy.∇h_tot[3]",),
        #     "specific enthalpy gradient's vertical component [J/kg/m]",
        #     "∇htotv", true,
        # ),
    )
        for (data, label_prefix, filename_suffix) in (
            (elem_stack_avgs, "", ""),
            (elem_stack_vars, "horizontal variance of ", "_var"),
            # (elem_stack_vals, "value at southwest node of ", "_val"),
        )
            plot(;
                legend = :outertopright,
                palette = :darkrainbow,
                xlabel = "$label_prefix$label",
                ylabel = "z [km]",
            )
            for name in names, (index, time) in enumerate(times)
                index == 1 && skip_first && continue
                id = time_string(time)
                plot!(
                    data[index][name],
                    zs;
                    seriescolor = (1 < index < length(times)) ? index : :black,
                    label = length(names) > 1 ? "$name, $id" : id,
                )
            end
            savefig(joinpath(@__DIR__, "$(filename)$(filename_suffix).png"))
        end
    end
    for (name, label, skip_first) in (
        # ("P", "pressure [Pa]", false),
        ("T", "temperature [K]", false),
        ("F", "net upward radiation energy flux [W/m^2]", true),
        ("qvap_sat", "water vapor specific humidity at saturation", false),
        # ("qvap", "water vapor specific humidity", false),
        ("rel_hum", "water vapor relative humidity", false),
        # ("vmr_H2O", "water vapor volume mixing ratio", false),
    )
        plot(;
            legend = :outertopright,
            palette = :darkrainbow,
            xlabel = label,
            ylabel = "z [km]",
        )
        for (index, time) in enumerate(times)
            index == 1 && skip_first && continue
            plot!(
                elem_stack_debug_avgs[index][name],
                zs;
                seriescolor = (1 < index < length(times)) ? index : :black,
                label = time_string(time),
            )
        end
        savefig(joinpath(@__DIR__, "$name.png"))
    end
end

function solve_and_plot_multiple!(
    driver_configs,
    FT,
    optics_symbol,
    timestart,
    timeend,
    substep_duration,
    substeps_per_step,
    ids,
    id_name,
    id_label,
    rm_dupes,
)
    data = Dict(
        "T_bot_final" => [],
        "T_top_final" => [],
        "T_avg_final" => [],
        "F_avg_final" => [],
        "F_init" => [],
        "T_final" => [],
        "rel_hum_final" => [],
        "nodal_z" => [],
        "elem_z" => [],
    )

    for driver_config in driver_configs
        solver_config, callbacks = setup_solver(
            driver_config,
            FT,
            optics_symbol,
            timestart,
            timeend,
            substep_duration,
            substeps_per_step,
        )
        zs = get_z(driver_config.grid; z_scale = 1e-3, rm_dupes = rm_dupes)
        push!(data["nodal_z"], zs)
        push!(
            data["elem_z"],
            zs[[1:driver_config.polyorders[2] + Int(!rm_dupes):end..., end]],
        )
        debug = get_horizontal_mean_debug(
            solver_config.Q,
            solver_config.dg,
            rm_dupes,
        )
        push!(data["F_init"], debug["F"])
        ClimateMachine.invoke!(solver_config; user_callbacks = callbacks)
        debug = get_horizontal_mean_debug(
            solver_config.Q,
            solver_config.dg,
            rm_dupes,
        )
        push!(data["T_bot_final"], debug["T"][1])
        push!(data["T_top_final"], debug["T"][end])
        push!(data["T_avg_final"], sum(debug["T"]) / length(debug["T"]))
        push!(data["F_avg_final"], sum(debug["F"]) / length(debug["F"]))
        push!(data["T_final"], debug["T"])
        push!(data["rel_hum_final"], debug["rel_hum"])
    end

    for (name, label) in (
        ("T_bot_final", "final temperature at bottom of domain [K]"),
        ("T_top_final", "final temperature at top of domain [K]"),
        ("T_avg_final", "final average temperature throughout domain [K]"),
        ("F_avg_final", "final average upward radiation energy flux [W/m^2]"),
    )
        plot(
            ids,
            data[name];
            legend = false,
            seriescolor = :black,
            xlabel = id_label,
            ylabel = label,
        )
        savefig(joinpath(@__DIR__, "$(name)s.png"))
    end

    for (scale, filename_suffix, select_func) in (
        (:identity, "", array -> array),
        (:log10, "_log", array -> array[2:end]),
    )
        for (name, label) in (
            ("F_init", "initial upward radiation energy flux [W/m^2]"),
            ("T_final", "final temperature [K]"),
            ("rel_hum_final", "final water vapor relative humidity"),
        )
            plot(;
                legend = :outertopright,
                palette = :darkrainbow,
                xlabel = label,
                ylabel = "z [km]",
                yscale = scale,
            )
            for (id_num, id) in enumerate(ids)
                plot!(
                    select_func(data[name][id_num]),
                    select_func(data["nodal_z"][id_num]);
                    label = "$id_name = $id",
                )
            end
            savefig(joinpath(@__DIR__, "$(name)s$filename_suffix.png"))
        end

        xfuncs = Array{Any}(map(id_num -> z -> id_num, 1:length(ids)))
        plot(
            xfuncs,
            map(select_func, data["nodal_z"]);
            legend = false,
            palette = :darkrainbow,
            seriestype = :scatter,
            markershape = :circle,
            markersize = 2,
            markerstrokewidth = 0,
            xlabel = id_label,
            xticks = (1:length(ids), ids),
            ylabel = "z [km]",
            yscale = scale,
        )
        plot!(
            xfuncs,
            map(select_func, data["elem_z"]);
            seriescolor = :black,
            seriestype = :scatter,
            markershape = :hline,
            markersize = 5,
        )
        savefig(joinpath(@__DIR__, "zs$filename_suffix.png"))
    end
end

function bottomΔz_stretch(
    nelem_vert,
    polyorder_vert,
    domain_height::FT,
    bottomΔz::FT,
    node_or_elem,
) where {FT}
    if node_or_elem == :elem || polyorder_vert == 0
        factor = FT(1)
    else
        factor = (lglpoints(FT, polyorder_vert)[1][2] + FT(1)) / FT(2)
    end
    function f!(F, x)
        stretch = x[1]
        if iszero(stretch)
            elemΔz = domain_height / nelem_vert
        else
            elemΔz = domain_height *
                expm1(stretch / nelem_vert) / expm1(stretch)
        end
        F[1] = bottomΔz - elemΔz * factor
    end
    return nlsolve(f!, FT[1]).zero[1]
end

function main()
    FT = Float64

    rm_dupes = true # Ignored for polyorder_vert = 0
    optics_symbol = :full
    moisture = DryModel()
    radiation = RRTMGPModelF1()
    timestart = FT(0)
    timeend = FT(year_to_s * 0.2)
    substep_duration = FT(5 * 60 * 60)
    substeps_per_step = 1
    domain_height = FT(50e3)
    domain_width = FT(50) # Ignored for AtmosGCMConfigType
    polyorder_vert = 4
    polyorder_horz = 4
    nelem_vert = 80
    nelem_horz = 1 # Ignored for SingleStackConfigType
    stretch = FT(0) # Ignored for SingleStackConfigType

    rm_dupes &= polyorder_vert > 0
    @testset for config_type in (
        # SingleStackConfigType(),
        AtmosLESConfigType(),
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
                CentralNumericalFluxFirstOrder(),
                # RusanovNumericalFlux(),
                # RoeNumericalFlux(),
                # HLLCNumericalFlux(),
            )
                @testset for temp_profile in (
                    # IsothermalProfile(param_set, FT),
                    DecayingTemperatureProfile{FT}(param_set),
                )
                    # driver_config = driver_configuration(
                    #     FT,
                    #     temp_profile,
                    #     moisture,
                    #     radiation,
                    #     numerical_flux,
                    #     solver_type,
                    #     config_type,
                    #     domain_height,
                    #     domain_width,
                    #     polyorder_vert,
                    #     polyorder_horz,
                    #     nelem_vert,
                    #     nelem_horz,
                    #     stretch,
                    # )
                    # solve_and_plot_detailed!(
                    #     driver_config,
                    #     FT,
                    #     optics_symbol,
                    #     timestart,
                    #     timeend,
                    #     substep_duration,
                    #     substeps_per_step,
                    #     rm_dupes,
                    # )

                    driver_configs = []
                    nelem_verts = Array(66:1:69)
                    for nelem_vert in nelem_verts
                        push!(driver_configs, driver_configuration(
                            FT,
                            temp_profile,
                            moisture,
                            radiation,
                            numerical_flux,
                            solver_type,
                            config_type,
                            domain_height,
                            domain_width,
                            polyorder_vert,
                            polyorder_horz,
                            nelem_vert,
                            nelem_horz,
                            # stretch,
                            bottomΔz_stretch(
                                nelem_vert,
                                polyorder_vert,
                                domain_height,
                                FT(100),
                                :elem,
                            ),
                        ))
                    end
                    solve_and_plot_multiple!(
                        driver_configs,
                        FT,
                        optics_symbol,
                        timestart,
                        timeend,
                        substep_duration,
                        substeps_per_step,
                        nelem_verts,
                        "N",
                        "number of vertical elements",
                        rm_dupes,
                    )

                    @test true
                end
            end
        end
    end
end

main()