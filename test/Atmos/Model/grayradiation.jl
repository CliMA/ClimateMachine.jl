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

    # Update the radiation model's auxiliary state in a seperate traversal.
    update_auxiliary_state!(dg, m.radiation, m, Q, t, elems)

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
    event = Event(device)
    event = kernel_init_rrtmgp!(device, (Nq[1], Nqj))(
        Val(dim),
        Val(Nq),
        Val(nvertelem),
        z,
        latitude,
        surface_emissivity,
        grid.vgeo,
        horzelems;
        ndrange = (length(horzelems) * Nq[1], Nqj),
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
    event = Event(device)
    event = kernel_load_into_rrtmgp!(device, (Nq[1], Nqj))(
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
        dg.state_auxiliary.data,
        horzelems;
        ndrange = (length(horzelems) * Nq[1], Nqj),
        dependencies = (event,),
    )
    wait(device, event)

    # Update the energy flux array in the GrayRRTMGP struct.
    gray_atmos_lw!(RRTMGPstruct)

    # Update the energy flux array in the auxiliary state.
    event = Event(device)
    event = kernel_unload_from_rrtmgp!(device, (Nq[1], Nqj))(
        Val(dim),
        Val(Nq),
        Val(nvertelem),
        Val(varsindex(vars(dg.state_auxiliary), :radiation, :flux)[1]),
        dg.state_auxiliary.data,
        RRTMGPstruct.flux.flux_net,
        horzelems;
        ndrange = (length(horzelems) * Nq[1], Nqj),
        dependencies = (event,),
    )
    wait(device, event)
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

@kernel function kernel_load_into_rrtmgp!(
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

        # Average level data to obtain layer data.
        for vindex in 1:(Nq3 - 1) * nvertelem
            p_lay[vindex, hindex] =
                (p_lev[vindex, hindex] + p_lev[vindex + 1, hindex]) * FT(0.5)
            t_lay[vindex, hindex] =
                (t_lev[vindex, hindex] + t_lev[vindex + 1, hindex]) * FT(0.5)
        end

        # Set the surface data.
        t_sfc[hindex] = t_lev[1, hindex]
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

function radiationflux(state, aux, t)
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
        problem = AtmosProblem(
            boundaryconditions = (
                AtmosBC(energy = PrescribedEnergyFlux(radiationflux)),
                AtmosBC(energy = PrescribedEnergyFlux(radiationflux)),
            ),
            init_state_prognostic = init_to_ref_state!,
        ),
        ref_state = HydrostaticState(temp_profile),
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
        radiation = RRTMGPModel(),
        source = (Gravity(),),
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
    timeend = FT(81)
    domain_height = FT(70e3)

    LES_params = let
        LES_resolution = ntuple(_ -> domain_height / 20poly_order, 3)
        LES_domain = (domain_height / 20, domain_height / 20, domain_height)
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
                        #Courant_number = FT(0.1),
                        init_on_cpu = true,
                        ode_solver_type = ode_solver_type,
                        #CFL_direction = EveryDirection(),
                        #diffdir = HorizontalDirection(),
                        modeldata = (RRTMGPstruct = create_rrtmgp(driver_config.grid),),
                    )

                    print_debug(solver_config.Q, solver_config.dg, timestart)
                    ClimateMachine.invoke!(solver_config)
                    print_debug(solver_config.Q, solver_config.dg, timeend)

                    @test all(isfinite.(solver_config.Q.data))
                end
            end
        end
    end
end

main()