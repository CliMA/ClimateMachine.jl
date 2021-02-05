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

using ClimateMachine.DGMethods: FVLinear, DGModel
using ClimateMachine.SystemSolvers: JacobianFreeNewtonKrylovSolver,
    BatchedGeneralizedMinimalResidual
using ClimateMachine.SingleStackUtils
using ClimateMachine.GenericCallbacks

const clima_dir = dirname(dirname(pathof(ClimateMachine)))
include(joinpath(clima_dir, "docs", "plothelpers.jl"))

####################### Adding to Preexisting Interface #######################

using UnPack
using ClimateMachine.BalanceLaws: Auxiliary, TendencyDef, Flux, FirstOrder,
    BalanceLaw, UpwardIntegrals, Source, SecondOrder, Gradient
using ClimateMachine.DGMethods: SpaceDiscretization
using ClimateMachine.Orientations: vertical_unit_vector, projection_tangential,
    projection_normal
using ClimateMachine.Atmos: nodal_update_auxiliary_state!
using ClimateMachine.TurbulenceConvection
using LinearAlgebra: Diagonal, tr
import ClimateMachine.BalanceLaws: vars_state, eq_tends, flux, source,
    prognostic_vars, compute_gradient_argument!, compute_gradient_flux!
import ClimateMachine.Atmos: update_auxiliary_state!, precompute

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
    if length(elems) > 0
        println(t, ", ", spacedisc.direction)
    end

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
    if spacedisc.direction isa EveryDirection
        update_auxiliary_state!(spacedisc, m.radiation, m, Q, t, elems)
    end

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

# Introduce Rayleigh friction towards a stationary profile.
struct RayleighFriction{PV <: Momentum, FT} <: TendencyDef{Source, PV}
    γ_h::FT # ~1e-2
    γ_v::FT # ~1e-1
end
RayleighFriction(γ_h::FT, γ_v::FT) where {FT} = RayleighFriction{Momentum, FT}(γ_h, γ_v)
function source(s::RayleighFriction{Momentum}, bl, args)
    @unpack state, aux = args
    return -s.γ_h * projection_tangential(bl, aux, state.ρu) -
        s.γ_v * projection_normal(bl, aux, state.ρu)
end

# Introduce divergence damping towards a stationary profile.
struct DivergenceDampingModel{FT} <: TurbulenceConvectionModel
    ν_h::FT # ~1e5
    ν_v::FT
end
struct DivergenceDamping{PV <: Momentum} <: TendencyDef{Flux{SecondOrder}, PV} end
prognostic_vars(::DivergenceDampingModel) = ()
vars_state(::DivergenceDampingModel, ::Gradient, FT) =
    @vars(ρu::SVector{3, FT})
vars_state(::DivergenceDampingModel, ::GradientFlux, FT) =
    @vars(ν∇D::SMatrix{3, 3, FT, 9})
eq_tends(pv, ::DivergenceDampingModel, tt) = ()
eq_tends(::PV, ::DivergenceDampingModel, ::Flux{SecondOrder}) where {PV <: Momentum} =
    (DivergenceDamping{PV}(),)
precompute(::DivergenceDampingModel, bl, args, ts, tend_type) = NamedTuple()
function update_auxiliary_state!(
    spacedisc::SpaceDiscretization,
    m::TurbulenceConvectionModel,
    bl::BalanceLaw,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    return nothing
end
function compute_gradient_argument!(
    ::DivergenceDampingModel,
    bl::BalanceLaw,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.turbconv.ρu = state.ρu
end
function compute_gradient_flux!(
    m::DivergenceDampingModel,
    bl::BalanceLaw,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ∇ρu = ∇transform.turbconv.ρu
    ẑ = vertical_unit_vector(bl, aux)
    ∇D_v = ẑ' * ∇ρu * ẑ
    ∇D_h = tr(∇ρu) - ∇D_v
    diffusive.turbconv.ν∇D =
        Diagonal(SVector(m.ν_h, m.ν_h, m.ν_v)) *
        Diagonal(SVector(∇D_h, ∇D_h, ∇D_v))
end
function flux(::DivergenceDamping{Momentum}, bl, args)
    @unpack diffusive = args
    return -diffusive.turbconv.ν∇D
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
const ngpoints = 1

# Create and initialize the struct used by RRTMGP.
function create_rrtmgp(grid)
    topology = grid.topology
    elems = topology.realelems
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)
    nhorzelem = length(horzelems)
    dim = dimensionality(grid)
    Nq = polynomialorders(grid) .+ 1
    Nq1 = Nq[1]
    Nq2 = dim == 2 ? 1 : Nq[2]
    Nq3 = Nq[dim]

    vgeo = grid.vgeo
    FT = eltype(vgeo)
    DA = arraytype(grid)
    device = array_device(vgeo)

    # Allocate the vertically flattened arrays for the GrayRRTMGP struct.
    nvert = Nq3 == 1 ? nvertelem : (Nq3 - 1) * nvertelem + 1
    nhorz = Nq1 * Nq2 * nhorzelem
    z = similar(DA, FT, nvert, nhorz)
    pressure = similar(DA, FT, nvert, nhorz)
    temperature = similar(DA, FT, nvert, nhorz)
    latitude = similar(DA, FT, nhorz)
    surface_emissivity = similar(DA, FT, nhorz)

    # Fill in the constant arrays: z, latitude, and surface_emissivity.
    event = Event(device)
    kernel = Nq3 == 1 ? kernel_init_rrtmgp_fv! : kernel_init_rrtmgp_dg!
    event = kernel(device, (Nq1, Nq2))(
        Val(dim),
        Val(Nq),
        Val(nvertelem),
        z,
        latitude,
        surface_emissivity,
        grid.vgeo,
        horzelems;
        ndrange = (length(horzelems) * Nq1, Nq2),
        dependencies = (event,),
    )
    wait(device, event)

    # Allocate the GrayRRTMGP struct.
    as = GrayAtmosphericState(nvert - 1, nhorz, pressure, temperature, z, latitude, DA)
    OPC = nstreams == 1 ? GrayOneScalar : GrayTwoStream
    op = OPC(FT, nhorz, nvert - 1, DA)
    src = source_func_longwave_gray_atmos(FT, nhorz, nvert - 1, ngpoints, OPC, DA)
    bcs = GrayLwBCs(DA, surface_emissivity)
    flux = GrayFlux(nhorz, nvert - 1, nvert, FT, DA)
    angle_disc = AngularDiscretization(FT, ngaussangles, DA)
    return GrayRRTMGP{
        FT,
        Int,
        DA{FT, 1},
        DA{FT, 2},
        DA{FT, 3},
        Bool,
        typeof(op),
        typeof(src),
        typeof(bcs),
    }(as, op, src, bcs, flux, angle_disc)
end

# Update the struct used by RRTMGP, use it to calculate energy fluxes, and
# unload those fluxes into the auxiliary state.
function update_auxiliary_state!(
    spacedisc::SpaceDiscretization,
    ::RRTMGPModel,
    m::BalanceLaw,
    state_prognostic::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    RRTMGPstruct = spacedisc.modeldata.RRTMGPstruct
    as = RRTMGPstruct.as

    device = array_device(state_prognostic)

    grid = spacedisc.grid
    topology = grid.topology
    dim = dimensionality(grid)
    Nq = polynomialorders(grid) .+ 1
    Nq1 = Nq[1]
    Nq2 = dim == 2 ? 1 : Nq[2]
    Nq3 = Nq[dim]
    nvertelem = topology.stacksize
    horzelems = fld1(first(elems), nvertelem):fld1(last(elems), nvertelem)

    # Update the pressure and temperature arrays in the GrayRRTMGP struct.
    event = Event(device)
    kernel = Nq3 == 1 ?
        kernel_load_into_rrtmgp_fv! : kernel_load_into_rrtmgp_dg!
    event = kernel(device, (Nq1, Nq2))(
        m,
        Val(dim),
        Val(Nq),
        Val(nvertelem),
        as.p_lev,
        as.p_lay,
        as.t_lev,
        as.t_lay,
        as.t_sfc,
        state_prognostic.data,
        spacedisc.state_auxiliary.data,
        horzelems;
        ndrange = (length(horzelems) * Nq1, Nq2),
        dependencies = (event,),
    )
    wait(device, event)

    if length(elems) > 0
        println("{$t, $(sum(as.t_lev[end, :]) / size(as.t_lev, 2))},")
    end

    # Update the energy flux array in the GrayRRTMGP struct.
    gray_atmos_lw!(RRTMGPstruct)

    # Update the energy flux array in the auxiliary state.
    event = Event(device)
    kernel = Nq3 == 1 ?
        kernel_unload_from_rrtmgp_fv! : kernel_unload_from_rrtmgp_dg!
    event = kernel(device, (Nq1, Nq2))(
        Val(dim),
        Val(Nq),
        Val(nvertelem),
        Val(varsindex(vars(spacedisc.state_auxiliary), :radiation, :flux)[1]),
        spacedisc.state_auxiliary.data,
        RRTMGPstruct.flux.flux_net,
        horzelems;
        ndrange = (length(horzelems) * Nq1, Nq2),
        dependencies = (event,),
    )
    wait(device, event)
end

@kernel function kernel_init_rrtmgp_fv!(
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

        FT = eltype(vgeo)
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]
        hindex = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i
        
        # Fill in the z-coordinates.
        for ev in 1:nvertelem
            e = nvertelem * (eh - 1) + ev
            ijk = Nq1 * (j - 1) + i
            z[ev, hindex] = vgeo[ijk, Grids._x3, e]
        end

        # TODO: get the latitudes from the Orientation struct.
        latitude[hindex] = FT(π) * FT(17) / FT(180)

        # TODO: get the surface emissivities from the boundary conditions.
        surface_emissivity[hindex] = FT(1)
    end
end

@kernel function kernel_init_rrtmgp_dg!(
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

@kernel function kernel_load_into_rrtmgp_fv!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{Nq},
    ::Val{nvertelem},
    p_lev,
    p_lay,
    t_lev,
    t_lay,
    t_sfc,
    state_prognostic,
    state_auxiliary,
    elems,
) where {dim, Nq, nvertelem}
    @uniform begin
        Nq1 = Nq[1]
        Nq2 = dim == 2 ? 1 : Nq[2]

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
            thermo_state = recover_thermo_state(balance_law, prog, aux)
            p_lev[ev, hindex] = air_pressure(thermo_state)
            t_lev[ev, hindex] = air_temperature(thermo_state)
        end

        # Average the level data to obtain the layer data.
        for ev in 1:nvertelem - 1
            p_lay[ev, hindex] =
                (p_lev[ev, hindex] + p_lev[ev + 1, hindex]) * FT(0.5)
            t_lay[ev, hindex] =
                (t_lev[ev, hindex] + t_lev[ev + 1, hindex]) * FT(0.5)
        end

        # Fill in the surface data.
        t_sfc[hindex] = t_lev[1, hindex]
    end
end

@kernel function kernel_load_into_rrtmgp_dg!(
    balance_law::BalanceLaw,
    ::Val{dim},
    ::Val{Nq},
    ::Val{nvertelem},
    p_lev,
    p_lay,
    t_lev,
    t_lay,
    t_sfc,
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
            p_lev[vindex, hindex] *= FT(0.5)
            t_lev[vindex, hindex] += air_temperature(thermo_state)
            t_lev[vindex, hindex] *= FT(0.5)

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

        # Average the level data to obtain the layer data.
        for vindex in 1:(Nq3 - 1) * nvertelem
            p_lay[vindex, hindex] =
                (p_lev[vindex, hindex] + p_lev[vindex + 1, hindex]) * FT(0.5)
            t_lay[vindex, hindex] =
                (t_lev[vindex, hindex] + t_lev[vindex + 1, hindex]) * FT(0.5)
        end

        # Fill in the surface data.
        t_sfc[hindex] = t_lev[1, hindex]
    end
end

@kernel function kernel_unload_from_rrtmgp_fv!(
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
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]
        hindex = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i

        for ev in 1:nvertelem
            e = nvertelem * (eh - 1) + ev
            ijk = Nq1 * (j - 1) + i
            state_auxiliary[ijk, fluxindex, e] = flux[ev, hindex]
        end
    end
end

@kernel function kernel_unload_from_rrtmgp_dg!(
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

function print_debug(Q, dg, t)
    grid = dg.grid
    balance_law = dg.balance_law
    state_auxiliary = dg.state_auxiliary

    vgeo = grid.vgeo

    dim = dimensionality(grid)
    Nq = polynomialorders(grid) .+ 1
    Nq1 = Nq[1]
    Nq2 = dim == 2 ? 1 : Nq[2]
    Nq3 = Nq[3]

    topology = grid.topology
    elems = topology.realelems
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

    nvert = Nq3 * nvertelem
    nhorz = FT(Nq1 * Nq2 * length(horzelems))

    zs = Array{FT}(undef, nvert)
    ps = Array{FT}(undef, nvert)
    ts = Array{FT}(undef, nvert)

    if nhorz > 0
        for ev in 1:nvertelem
            for k in 1:Nq3
                ivert = Nq3 * (ev - 1) + k
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

                    zs[ivert] += vgeo[ijk, Grids._x3, e]
                    ps[ivert] += air_pressure(thermo_state)
                    ts[ivert] += air_temperature(thermo_state)
                end
                zs[ivert] /= nhorz
                ps[ivert] /= nhorz
                ts[ivert] /= nhorz
            end
        end
    end
    
    println("z($(cpad(t, 4, 1))) = $(cpad(zs, 5, 2))")
    println("P($(cpad(t, 4, 1))) = $(cpad(ps, 3, 4))")
    println("T($(cpad(t, 4, 1))) = $(cpad(ts, 3, 4))")
end

################################# Driver Code #################################

function init_to_ref_state!(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)
    state.ρ = aux.ref_state.ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = aux.ref_state.ρe
end

function radfluxup(state, aux, t)
    return aux.radiation.flux
end

function radfluxdown(state, aux, t)
    return -aux.radiation.flux
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
        init_state_prognostic = init_to_ref_state!,
        # problem = AtmosProblem(
        #     boundaryconditions = (
        #         AtmosBC(energy = PrescribedEnergyFlux(radfluxup)),   # surface
        #         AtmosBC(energy = PrescribedEnergyFlux(radfluxdown)), # TOA
        #     ),
        #     init_state_prognostic = init_to_ref_state!,
        # ),
        ref_state = HydrostaticState(temp_profile),
        turbulence = ConstantDynamicViscosity(FT(0)),
        turbconv = DivergenceDampingModel(FT(0), FT(0)),
        moisture = DryModel(),
        # radiation = RRTMGPModel(),
        source = (Gravity(), RayleighFriction(FT(0), FT(0))),
    )

    config = config_fun(
        "balanced state",
        poly_order,
        config_args...,
        param_set,
        nothing;
        model = model,
        numerical_flux_first_order = numflux,
        fv_reconstruction = FVLinear(),
    )

    return config
end

function save_plots(data_avg, data_var, data_node, times, z, z_label, args...)
    l = length(times)
    n = 20
    if l <= n
        indices = 1:l
    else
        indices = [1; l - n + 2:l] # first data point and last n-1 data points
    end
    for arg in args
        export_plot(
            z,
            times[indices],
            data_avg[indices],
            arg[1],
            joinpath(@__DIR__, arg[2] * "_vs_time_avg.png");
            xlabel = "Horizontal average of " * arg[3],
            ylabel = z_label,
            legend = :outertopright,
        )
        export_plot(
            z,
            times[indices],
            data_var[indices],
            arg[1],
            joinpath(@__DIR__, arg[2] * "_vs_time_var.png");
            xlabel = "Horizontal variance of " * arg[3],
            ylabel = z_label,
            legend = :outertopright,
        )
        export_plot(
            z,
            times[indices],
            data_node[indices],
            arg[1],
            joinpath(@__DIR__, arg[2] * "_vs_time_node.png");
            xlabel = "Value at southwest node of " * arg[3],
            ylabel = z_label,
            legend = :outertopright,
        )
    end
end

function main()
    FT = Float64

    timestart = FT(0)
    timeend = FT(60) * FT(10000)
    timestep = FT(0.6)
    domain_height = FT(70e3)
    polyorder_horz = 4
    polyorder_vert = 2
    nelem_horz = 1
    nelem_vert = 20

    LES_params = let
        npoints_horz = polyorder_horz == 0 ? nelem_horz : nelem_horz * polyorder_horz
        npoints_vert = polyorder_vert == 0 ? nelem_vert : nelem_vert * polyorder_vert
        LES_resolution = (
            domain_height / npoints_horz,
            domain_height / npoints_horz,
            domain_height / npoints_vert,
        )
        LES_domain = (domain_height, domain_height, domain_height)
        (LES_resolution, LES_domain...)
    end
    GCM_params = ((nelem_horz, nelem_vert), domain_height)

    LES = (AtmosLESConfigType, ClimateMachine.AtmosLESConfiguration, LES_params)
    GCM = (AtmosGCMConfigType, ClimateMachine.AtmosGCMConfiguration, GCM_params)

    explicit_solver_type = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK54CarpenterKennedy,
    )
    hevi_solver_type = ClimateMachine.HEVISolverType(
        FT;
        #solver_method = ARK548L2SA2KennedyCarpenter,
        preconditioner_update_freq = 0,
    )

    @testset for config in (LES,)# GCM)
        @testset for ode_solver_type in (
            explicit_solver_type,
            #hevi_solver_type,
        )
            @testset for numflux in (
                CentralNumericalFluxFirstOrder(),
                #RusanovNumericalFlux(),
                #RoeNumericalFlux(),
                #HLLCNumericalFlux(),
            )
                @testset for temp_profile in (
                    #IsothermalProfile(param_set, FT),
                    DecayingTemperatureProfile{FT}(param_set),
                )
                    driver_config = config_balanced(
                        FT,
                        (polyorder_horz, polyorder_vert),
                        temp_profile,
                        numflux,
                        config,
                    )

                    solver_config = ClimateMachine.SolverConfiguration(
                        timestart,
                        timeend,
                        driver_config,
                        ode_solver_type = ode_solver_type,
                        ode_dt = timestep,
                        modeldata = (RRTMGPstruct = create_rrtmgp(driver_config.grid),),
                    )

                    data_avg = Dict[]
                    data_var = Dict[]
                    data_node = Dict[]
                    times = FT[]
                    function push_data()
                        state_vars_avg = get_horizontal_mean(
                            driver_config.grid,
                            solver_config.Q,
                            vars_state(driver_config.bl, Prognostic(), FT),
                        )
                        state_vars_var = get_horizontal_variance(
                            driver_config.grid,
                            solver_config.Q,
                            vars_state(driver_config.bl, Prognostic(), FT),
                        )
                        state_vars_node = get_vars_from_nodal_stack(
                            driver_config.grid,
                            solver_config.Q,
                            vars_state(driver_config.bl, Prognostic(), FT),
                            i = 1,
                            j = 1,
                        )
                        push!(data_avg, state_vars_avg)
                        push!(data_var, state_vars_var)
                        push!(data_node, state_vars_node)
                        push!(times, gettime(solver_config.solver))
                        return nothing
                    end
                    push_data()
                    callback = GenericCallbacks.EveryXSimulationSteps(
                        push_data,
                        1,
                    )

                    print_debug(solver_config.Q, solver_config.dg, timestart)
                    ClimateMachine.invoke!(solver_config)#; user_callbacks = (callback,))
                    print_debug(solver_config.Q, solver_config.dg, timeend)

                    save_plots(
                        data_avg,
                        data_var,
                        data_node,
                        times,
                        get_z(driver_config.grid; z_scale = 1e-3),
                        "z [km]",
                        (("ρu[1]", "ρu[2]"), "ρuh", "horizontal components of ρu [m/s]"),
                        (("ρu[3]",), "ρuv", "vertical component of ρu [m/s]"),
                        (("ρe",), "ρe", "ρe [J]"),
                        (("ρ",), "ρ", "ρ [kg/m^3]"),
                    )

                    @test all(isfinite.(solver_config.Q.data))
                end
            end
        end
    end
end

main()