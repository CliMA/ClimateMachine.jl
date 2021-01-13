using StaticArrays
using Test

using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.MPIStateArrays
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers: ManyColumnLU
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

####################### Adding to Preexisting Interface #######################

using UnPack
using ClimateMachine.BalanceLaws: Auxiliary, TendencyDef, Flux, FirstOrder,
    BalanceLaw, UpwardIntegrals
using ClimateMachine.DGMethods: DGModel
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
function flux(::Radiation{Energy}, atmos, args)
    @unpack aux = args
    return aux.radiation.flux * vertical_unit_vector(atmos, aux)
end

# Make update_auxiliary_state! also update data for radiation.
function update_auxiliary_state!(
    dg::DGModel,
    m::AtmosModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    FT = eltype(Q)
    state_auxiliary = dg.state_auxiliary

    if number_states(m, UpwardIntegrals()) > 0
        indefinite_stack_integral!(dg, m, Q, state_auxiliary, t, elems)
        reverse_indefinite_stack_integral!(dg, m, Q, state_auxiliary, t, elems)
    end

    update_auxiliary_state!(nodal_update_auxiliary_state!, dg, m, Q, t, elems)

    # TODO: Remove this hook. This hook was added for implementing
    # the first draft of EDMF, and should be removed so that we can
    # rely on a single vertical element traversal. This hook allows
    # us to compute globally vertical quantities specific to EDMF
    # until we're able to remove them or somehow incorporate them
    # into a higher level hierarchy.
    update_auxiliary_state!(dg, m.turbconv, m, Q, t, elems)

    println("Updating aux for $(length(elems)) elems at time $(cpad(t, 2, 3))")
    print("Update aux before: ")
    print_temps(Q, dg, t, elems)
    print("Update aux before: ")
    print_press(Q, dg, t, elems)

    # Update the radiation model's auxiliary state in a seperate traversal.
    update_auxiliary_state!(dg, m.radiation, m, Q, t, elems)

    print("Update aux after : ")
    print_temps(Q, dg, t, elems)
    print("Update aux after : ")
    print_press(Q, dg, t, elems)

    return true
end

# By default, don't do anything special for radiation.
function update_auxiliary_state!(
    dg::DGModel,
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
using RRTMGP.GrayBCs:                    GrayLwBCs
using RRTMGP.GrayAngularDiscretizations: AngularDiscretization
using RRTMGP.GrayAtmosphericStates:      GrayAtmosphericState
using RRTMGP.GrayFluxes:                 GrayFlux
using RRTMGP.GraySources:                source_func_longwave_gray_atmos
using RRTMGP.GrayAtmos:                  GrayRRTMGP
using RRTMGP.GrayRTESolver:              gray_atmos_lw!
using RRTMGP.GrayOptics:                 GrayOneScalar, GrayTwoStream
using ClimateMachine.BalanceLaws: number_states, Prognostic
using ClimateMachine.Thermodynamics: air_pressure, air_temperature

const ngaussangles = 1
const nstreams = 1

# Create and initialize the struct used by RRTMGP.
function create_rrtmgp(grid)
    topology = grid.topology
    elems = topology.realelems
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)
    nhorzelem = length(horzelems)
    dim = dimensionality(grid)
    Nq = polynomialorders(grid) .+ 1
    Nqj = dim == 2 ? 1 : Nq[2]

    vgeo = grid.vgeo
    FT = eltype(vgeo)
    DA = arraytype(grid)
    device = array_device(vgeo)

    # Allocate the vertically flattened arrays for the GrayRRTMGP struct.
    nvert = (Nq[dim] - 1) * nvertelem + 1
    nhorz = Nq[1] * Nqj * nhorzelem
    z = similar(DA, FT, nvert, nhorz)
    pressure = similar(DA, FT, nvert, nhorz)
    temperature = similar(DA, FT, nvert, nhorz)
    latitude = similar(DA, FT, nhorz)
    surface_emissivity = similar(DA, FT, nhorz)

    # Fill in the constant arrays: z, latitude, and surface_emissivity.
    # event = Event(device)
    # event = kernel_init_rrtmgp!(device, (Nq[1], Nqj))(
    #     Val(dim),
    #     Val(Nq),
    #     Val(nvertelem),
    #     z,
    #     latitude,
    #     surface_emissivity,
    #     grid.vgeo,
    #     horzelems;
    #     ndrange = (length(horzelems) * Nq[1], Nqj),
    #     dependencies = (event,),
    # )
    # wait(device, event)
    init_rrtmgp!(
        Val(dim),
        Val(Nq),
        Val(nvertelem),
        z,
        latitude,
        surface_emissivity,
        grid.vgeo,
        horzelems,
    )

    # Allocate the GrayRRTMGP struct.
    as = GrayAtmosphericState(nvert - 1, nhorz, pressure, temperature, z, latitude, DA)
    OPC = nstreams == 1 ? GrayOneScalar : GrayTwoStream
    optical_props = OPC(FT, nhorz, nvert - 1, DA)
    src = source_func_longwave_gray_atmos(FT, nhorz, nvert - 1, 1, OPC, DA)
    bcs = GrayLwBCs(DA, surface_emissivity)
    gray_flux = GrayFlux(nhorz, nvert - 1, nvert, FT, DA)
    ang_disc = AngularDiscretization(FT, ngaussangles, DA)

    # TODO: get rid of this after debugging
    fill!(as.p_lev, zero(FT))
    fill!(as.p_lay, zero(FT))
    fill!(as.t_lev, zero(FT))
    fill!(as.t_lay, zero(FT))
    fill!(as.t_lay, zero(FT))
    fill!(src.lay_source, zero(FT))
    fill!(src.lev_source_inc, zero(FT))
    fill!(src.lev_source_dec, zero(FT))
    fill!(src.sfc_source, zero(FT))
    fill!(src.source_up, zero(FT))
    fill!(src.source_dn, zero(FT))
    fill!(gray_flux.flux_up, zero(FT))
    fill!(gray_flux.flux_dn, zero(FT))
    fill!(gray_flux.flux_net, zero(FT))

    return GrayRRTMGP{
        FT,
        Int,
        DA{FT, 1},
        DA{FT, 2},
        DA{FT, 3},
        Bool,
        typeof(optical_props),
        typeof(src),
        typeof(bcs),
    }(as, optical_props, src, bcs, gray_flux, ang_disc)
end

# Update the struct used by RRTMGP, use it to calculate energy fluxes, and
# unload those fluxes into the auxiliary state.
function update_auxiliary_state!(
    dg::DGModel,
    ::RRTMGPModel,
    m::BalanceLaw,
    state_prognostic::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    RRTMGPstruct = dg.modeldata.RRTMGPstruct
    as = RRTMGPstruct.as

    device = array_device(state_prognostic)

    grid = dg.grid
    topology = grid.topology
    dim = dimensionality(grid)
    Nq = polynomialorders(grid) .+ 1
    Nqj = dim == 2 ? 1 : Nq[2]
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)

    # Update the pressure and temperature arrays in the GrayRRTMGP struct.
    println(
        "Load into before : T($(cpad(t, 2, 3))) = ",
        cpad(sum(as.t_lev; dims = 2) ./ size(as.t_lev, 2), 3, 6),
        "\nLoad into before : P($(cpad(t, 2, 3))) = ",
        cpad(sum(as.p_lev; dims = 2) ./ size(as.p_lev, 2), 3, 7),
        "\nLoad into before : e($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.bcs.sfc_emis) / size(RRTMGPstruct.bcs.sfc_emis, 1), 3, 7),
        "\nLoad into before :s1($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.src.lay_source; dims = 2) ./ size(RRTMGPstruct.src.lay_source, 2), 3, 7),
        "\nLoad into before :s2($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.src.lev_source_inc; dims = 2) ./ size(RRTMGPstruct.src.lev_source_inc, 2), 3, 7),
        "\nLoad into before :s3($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.src.lev_source_dec; dims = 2) ./ size(RRTMGPstruct.src.lev_source_dec, 2), 3, 7),
        "\nLoad into before :s4($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.src.sfc_source) / size(RRTMGPstruct.src.sfc_source, 1), 3, 7),
        "\nLoad into before :s5($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.src.source_up; dims = 2) ./ size(RRTMGPstruct.src.source_up, 2), 3, 7),
        "\nLoad into before :s6($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.src.source_dn; dims = 2) ./ size(RRTMGPstruct.src.source_dn, 2), 3, 7),
        "\nLoad into before : a($(cpad(t, 2, 3))) = ",
        RRTMGPstruct.angle_disc,
    )
    # event = Event(device)
    # event = kernel_load_into_rrtmgp!(device, (Nq[1], Nqj))(
    #     m,
    #     Val(dim),
    #     Val(Nq),
    #     Val(nvertelem),
    #     as.p_lev,
    #     as.p_lay,
    #     as.t_lev,
    #     as.t_lay,
    #     state_prognostic.data,
    #     dg.state_auxiliary.data,
    #     horzelems;
    #     ndrange = (length(horzelems) * Nq[1], Nqj),
    #     dependencies = (event,),
    # )
    # wait(device, event)
    load_into_rrtmgp!(
        m,
        Val(dim),
        Val(Nq),
        Val(nvertelem),
        as.p_lev,
        as.p_lay,
        as.t_lev,
        as.t_lay,
        state_prognostic.data,
        dg.state_auxiliary.data,
        horzelems,
    )
    println(
        "Load into after  : T($(cpad(t, 2, 3))) = ",
        cpad(sum(as.t_lev; dims = 2) ./ size(as.t_lev, 2), 3, 6),
        "\nLoad into after  : P($(cpad(t, 2, 3))) = ",
        cpad(sum(as.p_lev; dims = 2) ./ size(as.p_lev, 2), 3, 7),
        "\nLoad into after  : e($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.bcs.sfc_emis) / size(RRTMGPstruct.bcs.sfc_emis, 1), 3, 7),
        "\nLoad into after  :s1($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.src.lay_source; dims = 2) ./ size(RRTMGPstruct.src.lay_source, 2), 3, 7),
        "\nLoad into after  :s2($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.src.lev_source_inc; dims = 2) ./ size(RRTMGPstruct.src.lev_source_inc, 2), 3, 7),
        "\nLoad into after  :s3($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.src.lev_source_dec; dims = 2) ./ size(RRTMGPstruct.src.lev_source_dec, 2), 3, 7),
        "\nLoad into after  :s4($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.src.sfc_source) / size(RRTMGPstruct.src.sfc_source, 1), 3, 7),
        "\nLoad into after  :s5($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.src.source_up; dims = 2) ./ size(RRTMGPstruct.src.source_up, 2), 3, 7),
        "\nLoad into after  :s6($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.src.source_dn; dims = 2) ./ size(RRTMGPstruct.src.source_dn, 2), 3, 7),
        "\nLoad into after  : a($(cpad(t, 2, 3))) = ",
        RRTMGPstruct.angle_disc,
    )

    # Update the energy flux array in the GrayRRTMGP struct.
    println(
        "RRTMGP before  : F_u($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.flux.flux_up; dims = 2) ./ size(RRTMGPstruct.flux.flux_up, 2), 3, 6),
        "\nRRTMGP before  : F_d($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.flux.flux_dn; dims = 2) ./ size(RRTMGPstruct.flux.flux_dn, 2), 3, 6),
        "\nRRTMGP before  : F_n($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.flux.flux_net; dims = 2) ./ size(RRTMGPstruct.flux.flux_net, 2), 3, 6),
    )
    gray_atmos_lw!(RRTMGPstruct, max_threads = 256)
    println(
        "RRTMGP after   : F_u($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.flux.flux_up; dims = 2) ./ size(RRTMGPstruct.flux.flux_up, 2), 3, 6),
        "\nRRTMGP after   : F_d($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.flux.flux_dn; dims = 2) ./ size(RRTMGPstruct.flux.flux_dn, 2), 3, 6),
        "\nRRTMGP after   : F_n($(cpad(t, 2, 3))) = ",
        cpad(sum(RRTMGPstruct.flux.flux_net; dims = 2) ./ size(RRTMGPstruct.flux.flux_net, 2), 3, 6),
    )

    # Update the energy flux array in the auxiliary state.
    # event = Event(device)
    # event = kernel_unload_from_rrtmgp!(device, (Nq[1], Nqj))(
    #     Val(dim),
    #     Val(Nq),
    #     Val(nvertelem),
    #     Val(varsindex(vars(dg.state_auxiliary), :radiation, :flux)[1]),
    #     dg.state_auxiliary.data,
    #     RRTMGPstruct.flux.flux_net,
    #     horzelems;
    #     ndrange = (length(horzelems) * Nq[1], Nqj),
    #     dependencies = (event,),
    # )
    # wait(device, event)
    unload_from_rrtmgp!(
        Val(dim),
        Val(Nq),
        Val(nvertelem),
        Val(varsindex(vars(dg.state_auxiliary), :radiation, :flux)[1]),
        dg.state_auxiliary.data,
        RRTMGPstruct.flux.flux_net,
        horzelems,
    )
end

@kernel function kernel_init_rrtmgp!(
    ::Val{dim},
    ::Val{Nq},
    ::Val{nvertelem},
    z,
    latitude,
    surface_emissivity,
    vgeo,
    elems,
) where {dim, Nq, nvertelem}
    @uniform begin
        Nq1 = Nq[1]
        Nq2 = dim == 2 ? 1 : Nq[2]
        Nq3 = Nq[dim]

        FT = eltype(vgeo)
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]
        hindex = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i
        
        # Fill in all but the top z-coordinates.
        for ev in 1:nvertelem
            e = nvertelem * (eh - 1) + ev
            @unroll for k in 1:Nq3 - 1
                ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                vindex = (Nq3 - 1) * (ev - 1) + k
                z[vindex, hindex] = vgeo[ijk, Grids._x3, e]
            end
        end

        # Fill in the top z-coordinates (ev = nvertelem, k = Nq3).
        ijk = Nq1 * (Nq2 * (Nq3 - 1) + (j - 1)) + i
        vindex = (Nq3 - 1) * nvertelem + 1
        z[vindex, hindex] = vgeo[ijk, Grids._x3, nvertelem * eh]

        # TODO: get the latitudes from the Orientation struct.
        latitude[hindex] = FT(π) * FT(17) / FT(180)

        # TODO: get the surface emissivities from the boundary conditions.
        surface_emissivity[hindex] = FT(1)
    end
end

function init_rrtmgp!(
    ::Val{dim},
    ::Val{Nq},
    ::Val{nvertelem},
    z,
    latitude,
    surface_emissivity,
    vgeo,
    elems,
) where {dim, Nq, nvertelem}
    Nq1 = Nq[1]
    Nq2 = dim == 2 ? 1 : Nq[2]
    Nq3 = Nq[dim]

    FT = eltype(vgeo)

    for eh in elems, i in 1:Nq1, j in 1:Nq2
        hindex = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i
        
        # Fill in all but the top z-coordinates.
        for ev in 1:nvertelem
            e = nvertelem * (eh - 1) + ev
            for k in 1:Nq3 - 1
                ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                vindex = (Nq3 - 1) * (ev - 1) + k
                z[vindex, hindex] = vgeo[ijk, Grids._x3, e]
            end
        end

        # Fill in the top z-coordinates (ev = nvertelem, k = Nq3).
        ijk = Nq1 * (Nq2 * (Nq3 - 1) + (j - 1)) + i
        vindex = (Nq3 - 1) * nvertelem + 1
        z[vindex, hindex] = vgeo[ijk, Grids._x3, nvertelem * eh]

        # TODO: get the latitudes from the Orientation struct.
        latitude[hindex] = FT(π) * FT(17) / FT(180)

        # TODO: get the surface emissivities from the boundary conditions.
        surface_emissivity[hindex] = FT(1)
    end
end

@kernel function kernel_load_into_rrtmgp!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{Nq},
    ::Val{nvertelem},
    p_lev,
    p_lay,
    t_lev,
    t_lay,
    state_prognostic,
    state_auxiliary,
    elems,
) where {dim, Nq, nvertelem}
    @uniform begin
        Nq1 = Nq[1]
        Nq2 = dim == 2 ? 1 : Nq[2]
        Nq3 = Nq[dim]

        FT = eltype(state_prognostic)
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

    @inbounds begin
        eh = elems[_eh]
        hindex = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i

        # Fill in data from the bottom element (ev = 1).
        e = nvertelem * (eh - 1) + 1
        @unroll for k in 1:Nq3
            ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
            vindex = k
            @unroll for s in 1:num_state_prognostic
                local_state_prognostic[s] = state_prognostic[ijk, s, e]
            end
            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
            end
            thermo_state = recover_thermo_state(balance_law, prog, aux)
            p_lev[vindex, hindex] = air_pressure(thermo_state)
            t_lev[vindex, hindex] = air_temperature(thermo_state)
        end

        # Fill in data from the remaining elements.
        for ev in 2:nvertelem
            e = nvertelem * (eh - 1) + ev

            # Average duplicate data from the bottom point (k = 1) of the
            # current element. The data from the top point (k = Nq3) of the
            # element below has already been filled in.
            ijk = Nq1 * (j - 1) + i
            vindex = (Nq3 - 1) * (ev - 1) + 1
            @unroll for s in 1:num_state_prognostic
                local_state_prognostic[s] = state_prognostic[ijk, s, e]
            end
            @unroll for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
            end
            thermo_state = recover_thermo_state(balance_law, prog, aux)
            p_lev[vindex, hindex] += air_pressure(thermo_state)
            p_lev[vindex, hindex] /= FT(2)
            t_lev[vindex, hindex] += air_temperature(thermo_state)
            t_lev[vindex, hindex] /= FT(2)

            # Fill in data from the remaining points.
            @unroll for k in 2:Nq3
                ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                vindex = (Nq3 - 1) * (ev - 1) + k
                @unroll for s in 1:num_state_prognostic
                    local_state_prognostic[s] = state_prognostic[ijk, s, e]
                end
                @unroll for s in 1:num_state_auxiliary
                    local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
                end
                thermo_state = recover_thermo_state(balance_law, prog, aux)
                p_lev[vindex, hindex] = air_pressure(thermo_state)
                t_lev[vindex, hindex] = air_temperature(thermo_state)
            end
        end

        # Average level data to obtain layer data.
        for vindex in 1:(Nq3 - 1) * nvertelem
            p_lay[vindex, hindex] =
                (p_lev[vindex, hindex] + p_lev[vindex + 1, hindex]) / FT(2)
            t_lay[vindex, hindex] =
                (t_lev[vindex, hindex] + t_lev[vindex + 1, hindex]) / FT(2)
        end
    end
end

function load_into_rrtmgp!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{Nq},
    ::Val{nvertelem},
    p_lev,
    p_lay,
    t_lev,
    t_lay,
    state_prognostic,
    state_auxiliary,
    elems,
) where {dim, Nq, nvertelem}
    Nq1 = Nq[1]
    Nq2 = dim == 2 ? 1 : Nq[2]
    Nq3 = Nq[dim]

    FT = eltype(state_prognostic)
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

    for eh in elems, i in 1:Nq1, j in 1:Nq2
        hindex = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i

        # Fill in data from the bottom element (ev = 1).
        e = nvertelem * (eh - 1) + 1
        for k in 1:Nq3
            ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
            vindex = k
            for s in 1:num_state_prognostic
                local_state_prognostic[s] = state_prognostic[ijk, s, e]
            end
            for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
            end
            thermo_state = recover_thermo_state(balance_law, prog, aux)
            p_lev[vindex, hindex] = air_pressure(thermo_state)
            t_lev[vindex, hindex] = air_temperature(thermo_state)
        end

        # Fill in data from the remaining elements.
        for ev in 2:nvertelem
            e = nvertelem * (eh - 1) + ev

            # Average duplicate data from the bottom point (k = 1) of the
            # current element. The data from the top point (k = Nq3) of the
            # element below has already been filled in.
            ijk = Nq1 * (j - 1) + i
            vindex = (Nq3 - 1) * (ev - 1) + 1
            for s in 1:num_state_prognostic
                local_state_prognostic[s] = state_prognostic[ijk, s, e]
            end
            for s in 1:num_state_auxiliary
                local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
            end
            thermo_state = recover_thermo_state(balance_law, prog, aux)
            p_lev[vindex, hindex] += air_pressure(thermo_state)
            p_lev[vindex, hindex] /= FT(2)
            t_lev[vindex, hindex] += air_temperature(thermo_state)
            t_lev[vindex, hindex] /= FT(2)

            # Fill in data from the remaining points.
            for k in 2:Nq3
                ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                vindex = (Nq3 - 1) * (ev - 1) + k
                for s in 1:num_state_prognostic
                    local_state_prognostic[s] = state_prognostic[ijk, s, e]
                end
                for s in 1:num_state_auxiliary
                    local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
                end
                thermo_state = recover_thermo_state(balance_law, prog, aux)
                p_lev[vindex, hindex] = air_pressure(thermo_state)
                t_lev[vindex, hindex] = air_temperature(thermo_state)
            end
        end

        # Average level data to obtain layer data.
        for vindex in 1:(Nq3 - 1) * nvertelem
            p_lay[vindex, hindex] =
                (p_lev[vindex, hindex] + p_lev[vindex + 1, hindex]) / FT(2)
            t_lay[vindex, hindex] =
                (t_lev[vindex, hindex] + t_lev[vindex + 1, hindex]) / FT(2)
        end
    end
end

@kernel function kernel_unload_from_rrtmgp!(
    ::Val{dim},
    ::Val{Nq},
    ::Val{nvertelem},
    ::Val{fluxindex},
    state_auxiliary,
    flux,
    elems,
) where {dim, Nq, nvertelem, fluxindex}
    @uniform begin
        Nq1 = Nq[1]
        Nq2 = dim == 2 ? 1 : Nq[2]
        Nq3 = Nq[dim]
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]
        hindex = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i

        for ev in 1:nvertelem
            e = nvertelem * (eh - 1) + ev
            @unroll for k in 1:Nq3
                ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                vindex = (Nq3 - 1) * (ev - 1) + k
                state_auxiliary[ijk, fluxindex, e] = flux[vindex, hindex]
            end
        end
    end
end

function unload_from_rrtmgp!(
    ::Val{dim},
    ::Val{Nq},
    ::Val{nvertelem},
    ::Val{fluxindex},
    state_auxiliary,
    flux,
    elems,
) where {dim, Nq, nvertelem, fluxindex}
    Nq1 = Nq[1]
    Nq2 = dim == 2 ? 1 : Nq[2]
    Nq3 = Nq[dim]

    for eh in elems, i in 1:Nq1, j in 1:Nq2
        hindex = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i

        for ev in 1:nvertelem
            e = nvertelem * (eh - 1) + ev
            for k in 1:Nq3
                ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                vindex = (Nq3 - 1) * (ev - 1) + k
                state_auxiliary[ijk, fluxindex, e] = flux[vindex, hindex]
            end
        end
    end
end

########################## Temporary Debugging Stuff ##########################

# Center-pads a string or number. The value of intdigits should be positive.
cpad(v::Array, args...) = '[' * join(cpad.(v, args...), ',') * ']'
cpad(x::Any, padding) = cpad(string(x), padding)
function cpad(s::String, padding)
    d, r = divrem(padding - length(s), 2)
    return ' '^(d + r) * s * ' '^d
end
function cpad(x::Real, intdigits, decdigits, sign = true)
    if !isfinite(x)
        padding = intdigits + Int(sign)
        if decdigits > 0
            padding += decdigits + 1
        end
        return cpad(x, padding)
    end
    isneg = signbit(x)
    x = abs(x)
    if x >= 10^intdigits
        pow = iszero(x) ? 0 : floor(Int, log10(x))
        x *= 10.0^(-pow)
        s = (isneg ? '-' : (sign ? ' ' : "")) * string(trunc(Int, x)) * '.'
        pow = string(pow)
        newdecdigits = intdigits + Int(sign)
        if decdigits > 0
            newdecdigits += decdigits + 1
        end
        newdecdigits -= length(s) + 1 + length(pow)
        if newdecdigits > 0
            s *= string(trunc(Int, (x - trunc(x)) * 10^newdecdigits))
        end
        return s * 'e' * pow
    else
        s = (isneg ? '-' : (sign ? ' ' : "")) * lpad(trunc(Int, x), intdigits)
        if decdigits > 0
            s *= '.' *
                lpad(trunc(Int, (x - trunc(x)) * 10^decdigits), decdigits, '0')
        end
        return s
    end
end

function print_temps(Q, dg, t, elems)
    grid = dg.grid
    balance_law = dg.balance_law
    state_auxiliary = dg.state_auxiliary

    dim = dimensionality(grid)
    Nq = polynomialorders(grid) .+ 1
    Nq1 = Nq[1]
    Nq2 = dim == 2 ? 1 : Nq[2]
    Nq3 = Nq[3]

    topology = grid.topology
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)
    
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

    n = FT(Nq1 * Nq2 * length(horzelems))

    print("T($(cpad(t, 2, 3))) = [")
    if n > 0
        for ev in 1:nvertelem
            for k in 1:Nq3
                tsum = zero(FT)
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
                    tsum += air_temperature(thermo_state)
                end
                tmean = tsum / n
                print(cpad(tmean, 3, 6))
                if ev < nvertelem || k < Nq3
                    print(',')
                end
            end
        end
    end
    println(']')
end

function print_press(Q, dg, t, elems)
    grid = dg.grid
    balance_law = dg.balance_law
    state_auxiliary = dg.state_auxiliary

    dim = dimensionality(grid)
    Nq = polynomialorders(grid) .+ 1
    Nq1 = Nq[1]
    Nq2 = dim == 2 ? 1 : Nq[2]
    Nq3 = Nq[3]

    topology = grid.topology
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)
    
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

    n = FT(Nq1 * Nq2 * length(horzelems))

    print("P($(cpad(t, 2, 3))) = [")
    if n > 0
        for ev in 1:nvertelem
            for k in 1:Nq3
                tsum = zero(FT)
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
                    tsum += air_pressure(thermo_state)
                end
                tmean = tsum / n
                print(cpad(tmean, 3, 7))
                if ev < nvertelem || k < Nq3
                    print(',')
                end
            end
        end
    end
    println(']')
end

################################# Driver Code #################################

function init_to_ref_state!(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)
    state.ρ = aux.ref_state.ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = aux.ref_state.ρe
end

function config_balanced(
    FT,
    poly_order,
    temp_profile,
    numflux,
    (config_type, config_fun, config_args),
)
    model = AtmosModel{FT}(
        config_type,
        param_set;
        ref_state = HydrostaticState(temp_profile),
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
        radiation = RRTMGPModel(),
        source = (Gravity(),),
        init_state_prognostic = init_to_ref_state!,
    )

    config = config_fun(
        "balanced state",
        poly_order,
        config_args...,
        param_set,
        nothing;
        model = model,
        numerical_flux_first_order = numflux,
    )

    return config
end

function main()
    FT = Float64
    poly_order = 4

    timestart = FT(0)
    timeend = FT(1.5)
    domain_height = FT(70e3)

    LES_params = let
        LES_resolution = ntuple(_ -> domain_height / 3poly_order, 3)
        LES_domain = ntuple(_ -> domain_height, 3)
        (LES_resolution, LES_domain...)
    end
    GCM_params = let
        GCM_resolution = (3, 3)
        (GCM_resolution, domain_height)
    end

    LES = (AtmosLESConfigType, ClimateMachine.AtmosLESConfiguration, LES_params)
    GCM = (AtmosGCMConfigType, ClimateMachine.AtmosGCMConfiguration, GCM_params)

    explicit_solver_type = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK54CarpenterKennedy,
    )
    imex_solver_type = ClimateMachine.IMEXSolverType(
        splitting_type = HEVISplitting(),
        implicit_model = AtmosAcousticGravityLinearModel,
        implicit_solver = ManyColumnLU,
        solver_method = ARK2GiraldoKellyConstantinescu,
    )

    @testset for config in (LES,)# GCM)
        @testset for ode_solver_type in (
            explicit_solver_type,
            #imex_solver_type,
        )
            @testset for numflux in (
                CentralNumericalFluxFirstOrder(),
                #RoeNumericalFlux(),
                #HLLCNumericalFlux(),
            )
                @testset for temp_profile in (
                    #IsothermalProfile(param_set, FT),
                    DecayingTemperatureProfile{FT}(param_set),
                )
                    driver_config = config_balanced(
                        FT,
                        poly_order,
                        temp_profile,
                        numflux,
                        config,
                    )

                    solver_config = ClimateMachine.SolverConfiguration(
                        timestart,
                        timeend,
                        driver_config,
                        Courant_number = FT(0.1),
                        init_on_cpu = true,
                        ode_solver_type = ode_solver_type,
                        CFL_direction = EveryDirection(),
                        diffdir = HorizontalDirection(),
                        modeldata = (RRTMGPstruct = create_rrtmgp(driver_config.grid),),
                        skip_update_aux = true,
                    )

                    update_auxiliary_state!(
                        solver_config.dg,
                        RRTMGPModel(),
                        solver_config.dg.balance_law,
                        solver_config.Q,
                        zero(FT),
                        solver_config.dg.grid.topology.realelems,
                    )

                    # ClimateMachine.invoke!(solver_config)

                    @test all(isfinite.(solver_config.Q.data))
                end
            end
        end
    end
end

main()