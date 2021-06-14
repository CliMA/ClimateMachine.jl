##################### Loading Modules and Defining Consts #####################

using Dierckx
using LinearAlgebra
using NCDatasets
using NLsolve
using Plots
using ProgressMeter
using Serialization
using StaticArrays
using Test
using UnPack

using RRTMGP
using RRTMGP.AngularDiscretizations
using RRTMGP.AtmosphericStates
using RRTMGP.BCs
using RRTMGP.Fluxes
using RRTMGP.RTESolver
using RRTMGP.LookUpTables
using RRTMGP.Optics
using RRTMGP.RTE
using RRTMGP.Sources
using RRTMGP.Vmrs
using KernelAbstractions
using KernelAbstractions.Extras: @unroll

using CLIMAParameters
using CLIMAParameters.Planet
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
# Prevent the ClimateMachine updates from messing up the progress bar.
ClimateMachine.init(show_updates = "never")

using ClimateMachine.VariableTemplates
using ClimateMachine.Thermodynamics
using ClimateMachine.TemperatureProfiles
using ClimateMachine.MPIStateArrays
using ClimateMachine.Mesh
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Topologies
using ClimateMachine.BalanceLaws
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Orientations
using ClimateMachine.SingleStackUtils
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.TurbulenceClosures
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes

# non-exported values
using RRTMGP.RTESolver: rte_lw_noscat_solve!, rte_lw_2stream_solve!
using ClimateMachine.DGMethods.FVReconstructions: FVLinear
using ClimateMachine.Atmos: Energy, nodal_update_auxiliary_state!,
    AbstractFilterTarget

# overwritten values
import ClimateMachine.Mesh.Filters: vars_state_filtered,
    compute_filter_argument!, compute_filter_result!
import ClimateMachine.BalanceLaws: eq_tends, vars_state, flux, prognostic_vars,
    source
import ClimateMachine.Atmos: atmos_energy_normal_boundary_flux_second_order!,
    update_auxiliary_state!, new_thermo_state, new_thermo_state_anelastic

# parameters for RRTMGP
const ngaussangles = 1
const max_threads = 256

# dataset paths
const ds_lw_path = joinpath(@__DIR__, "rrtmgp-data-lw-g256-2018-12-04.nc")
const ds_sw_path = joinpath(@__DIR__, "rrtmgp-data-sw-g224-2018-12-04.nc")
const ds_input_path = joinpath(
    @__DIR__,
    "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc",
)

# constants for initialization
const site_index = 28 # index of site used to initialize simulation
const expt_index = 1 # index of experiment used to initialize simulation

# This commented-out code generates all site and experiment indices which
# satisfy the following criteria:
#     - the surface pressure is no less than the surface pressure of the
#       decaying temperature profile (allows interpolation of surface values)
#     - the solar zenith angle is less than 90 degrees (allows use of shortwave
#       fluxes)
#     - the temperature profile is monotonically decreasing with height up to
#       at least 20 kPa (ensures that the profile has no significant
#       measurement artifacts)
# It also shows the maximum difference between the temperature profiles of any
# two experiments at the same site.
#
# ds_input = Dataset(ds_input_path)
# ps = Array{Float64}(ds_input["pres_level"])
# Ts = Array{Float64}(ds_input["temp_level"])
# zeniths = Array{Float64}(ds_input["solar_zenith_angle"])
# close(ds_input)
# pmax = DecayingTemperatureProfile{Float64}(param_set)(param_set, Float64(0))[2]
# site_indices = findall(
#     site_index -> ps[end, site_index] >= pmax && zeniths[site_index] < 90,
#     1:size(Ts, 2),
# )
# function is_T_monotonic(site_index, expt_index)
#     level_indices = findall(p -> p >= 20000, ps[:, site_index])
#     return all(
#         Ts[level_indices, site_index, expt_index] .>=
#         Ts[level_indices .- 1, site_index, expt_index]
#     )
# end
# for site_index in site_indices
#     expt_indices = findall(
#         expt_index -> is_T_monotonic(site_index, expt_index),
#         1:size(Ts, 3),
#     )
#     if length(expt_indices) > 0
#         print("site $site_index, ")
#         if length(expt_indices) == size(Ts, 3)
#             print("all experiments")
#         else
#             print("experiments with indices $expt_indices")
#         end
#         max_diff = maximum(
#             level_index -> begin
#                 expt_Ts = Ts[level_index, site_index, expt_indices]
#                 max_rel_err =
#                     (maximum(expt_Ts) - minimum(expt_Ts)) / minimum(expt_Ts)
#                 return round(Int, max_rel_err * 100)
#             end,
#             1:size(Ts, 1),
#         )
#         println(" (maximum difference = $max_diff%)")
#     end
# end
#
# site 28, all experiments (maximum difference = 9%)
# site 55, all experiments (maximum difference = 8%)
# site 56, all experiments (maximum difference = 9%)
# site 70, all experiments (maximum difference = 9%)

# TOA altitude
const z_toa = 111 * 1000

################ Overriding Parts of the Preexisting Interface ################

# Disable all tendencies except for radiation.
eq_tends(pv::Momentum, m::AtmosModel, tt::Flux{FirstOrder}) = ()
eq_tends(::Energy, m::TotalEnergyModel, tt::Flux{FirstOrder}) = ()
eq_tends(pv::Momentum, m::AtmosModel, tt::Flux{SecondOrder}) = (
    eq_tends(pv, m.moisture, tt)...,
    eq_tends(pv, turbconv_model(m), tt)...,
    eq_tends(pv, hyperdiffusion_model(m), tt)...,
)
eq_tends(::Energy, m::TotalEnergyModel, tt::Flux{SecondOrder}) = ()
eq_tends(
    ::Union{Mass, Momentum, AbstractMoistureVariable},
    ::AbstractMoistureModel,
    ::Flux{SecondOrder},
) = ()

# Allow a RadiationModel to use a second-order flux for an Insulating BC.
eq_tends(pv, ::RadiationModel, tt) = ()
eq_tends(pv::AbstractEnergyVariable, m::AtmosModel, tt::Flux{SecondOrder}) = (
    eq_tends(pv, m.energy, tt)...,
    eq_tends(pv, turbconv_model(m), tt)...,
    eq_tends(pv, hyperdiffusion_model(m), tt)...,
    eq_tends(pv, radiation_model(m), tt)..., # This line is new.
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
            eq_tends(prog, radiation_model(atmos), Flux{SecondOrder}()),
            atmos,
            (; state = state⁻, aux = aux⁻, t),
        ))
    end
end

# Make update_auxiliary_state! set radiation data.
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
    update_auxiliary_state!(spacedisc, turbconv_model(m), m, Q, t, elems)

    # Update the radiation model's auxiliary state in a separate traversal.
    update_auxiliary_state!(spacedisc, radiation_model(m), m, Q, t, elems)

    return true
end

# By default, don't do anything for radiation.
function update_auxiliary_state!(
    spacedisc::SpaceDiscretization,
    ::RadiationModel,
    m::BalanceLaw,
    state_prognostic::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
end

################ Defining Linear Interpolations for Layer Data ################

# The variable implied by the ideal gas law in a layer interpolation scheme;
# all other variables are linearly interpolated, so that variable f is
# approximated by f(z) = (f2 - f1) / (z2 - z1) * (z - z1) + f1.
abstract type ImpliedVariable end
struct Dens <: ImpliedVariable end # ρ(z) = M * p(z) / (R * T(z))
struct Temp <: ImpliedVariable end # T(z) = M * p(z) / (R * ρ(z))
struct Pres <: ImpliedVariable end # p(z) = R * ρ(z) * T(z) / M

abstract type LayerInterpolation{V <: ImpliedVariable} end

# z_lay = (z1 + z2) / 2
struct GeometricCenter{V} <: LayerInterpolation{V} end
# const Centroid = GeometricCenter

# z_lay = (z1 * p1 + z2 * p2) / (p1 + p2)
struct WeightedMean{V} <: LayerInterpolation{V} end

# z_lay = (∫_z1^z2 ρ(z) * z dz) / (∫_z1^z2 ρ(z) dz)
struct CenterOfMass{V} <: LayerInterpolation{V} end

# z_lay = (∫_z1^z2 p(z) * z dz) / (∫_z1^z2 p(z) dz)
# Note: this is not actually the center of pressure
struct CenterOfPres{V} <: LayerInterpolation{V} end

lay_pres(::GeometricCenter{<:Union{Dens, Temp}}, p1, p2, t1, t2) =
    (p1 + p2) / 2
lay_pres(::GeometricCenter{Pres}, p1, p2, t1, t2) =
    (t1 + t2) * (p1 / t1 + p2 / t2) / 4
lay_pres(::WeightedMean{<:Union{Dens, Temp}}, p1, p2, t1, t2) =
    (p1^2 + p2^2) / (p1 + p2)
lay_pres(::WeightedMean{Pres}, p1, p2, t1, t2) =
    (p1 * t1 + p2 * t2) * (p1^2 / t1 + p2^2 / t2) / (p1 + p2)^2
function lay_pres(::CenterOfMass{Dens}, p1, p2, t1, t2)
    FT = typeof(p1)
    t1 = BigFloat(t1) # Avoid errors when t1 ≈ t2.
    t2 = BigFloat(t2)
    return FT(
        (p1 * t2 - p2 * t1) / (t2 - t1) +
        (p2 - p1) * (p1 + p2) * (t2 - t1) / (2 * (
            (p2 - p1) * (t2 - t1) +
            (p1 * t2 - p2 * t1) * (log(t2) - log(t1))
        ))
    )
end
function lay_pres(::CenterOfMass{Temp}, p1, p2, t1, t2)
    C1 = p1 / t1
    C2 = p2 / t2
    return (p1 * (2C1 + C2) + p2 * (C1 + 2C2)) / (3 * (C1 + C2))
end
function lay_pres(::CenterOfMass{Pres}, p1, p2, t1, t2)
    C1 = p1 / t1
    C2 = p2 / t2
    return 2 * (t1 * (2C1 + C2) + t2 * (C1 + 2C2)) * (C1^2 + C1 * C2 + C2^2) /
        (9 * (C1 + C2)^2)
end
lay_pres(::CenterOfPres{<:Union{Dens, Temp}}, p1, p2, t1, t2) =
    2 * (p1^2 + p1 * p2 + p2^2) / (3 * (p1 + p2))
function lay_pres(::CenterOfPres{Pres}, p1, p2, t1, t2)
    C1 = p1 / t1
    C2 = p2 / t2
    return (t1^2 * (3C1 + C2) + 2 * t1 * t2 * (C1 + C2) + t2^2 * (C1 + 3C2)) *
        (C1^2 * (3t1 + t2) + 2 * C1 * C2 * (t1 + t2) + C2^2 * (t1 + 3t2)) /
        (2 * (t1 * (2C1 + C2) + t2 * (C1 + 2C2)))^2
end

lay_temp(::GeometricCenter{<:Union{Dens, Pres}}, p1, p2, t1, t2) =
    (t1 + t2) / 2
lay_temp(::GeometricCenter{Temp}, p1, p2, t1, t2) =
    (p1 + p2) / (p1 / t1 + p2 / t2)
lay_temp(::WeightedMean{<:Union{Dens, Pres}}, p1, p2, t1, t2) =
    (p1 * t1 + p2 * t2) / (p1 + p2)
lay_temp(::WeightedMean{Temp}, p1, p2, t1, t2) =
    (p1^2 + p2^2) / (p1^2 / t1 + p2^2 / t2)
function lay_temp(::CenterOfMass{Dens}, p1, p2, t1, t2)
    FT = typeof(p1)
    t1 = BigFloat(t1) # Avoid errors when t1 ≈ t2.
    t2 = BigFloat(t2)
    return FT(
        (p1 + p2) * (t2 - t1)^2 / (2 * (
            (p2 - p1) * (t2 - t1) +
            (p1 * t2 - p2 * t1) * (log(t2) - log(t1))
        )
    ))
end
function lay_temp(::CenterOfMass{Temp}, p1, p2, t1, t2)
    C1 = p1 / t1
    C2 = p2 / t2
    return (p1 * (2C1 + C2) + p2 * (C1 + 2C2)) / (2 * (C1^2 + C1 * C2 + C2^2))
end
function lay_temp(::CenterOfMass{Pres}, p1, p2, t1, t2)
    C1 = p1 / t1
    C2 = p2 / t2
    return (t1 * (2C1 + C2) + t2 * (C1 + 2C2)) / (3 * (C1 + C2))
end
lay_temp(::CenterOfPres{Dens}, p1, p2, t1, t2) =
    (t1 * (2p1 + p2) + t2 * (p1 + 2p2)) / (3 * (p1 + p2))
function lay_temp(::CenterOfPres{Temp}, p1, p2, t1, t2)
    C1 = p1 / t1
    C2 = p2 / t2
    return 2 * (p1^2 + p1 * p2 + p2^2) / (p1 * (2C1 + C2) + p2 * (C1 + 2C2))
end
function lay_temp(::CenterOfPres{Pres}, p1, p2, t1, t2)
    C1 = p1 / t1
    C2 = p2 / t2
    return (t1^2 * (3C1 + C2) + 2 * t1 * t2 * (C1 + C2) + t2^2 * (C1 + 3C2)) /
        (2 * (t1 * (2C1 + C2) + t2 * (C1 + 2C2)))
end

lay_var(::GeometricCenter, p1, p2, t1, t2, v1, v2) = (v1 + v2) / 2
lay_var(::WeightedMean, p1, p2, t1, t2, v1, v2) =
    (p1 * v1 + p2 * v2) / (p1 + p2)
function lay_var(::CenterOfMass{Dens}, p1, p2, t1, t2, v1, v2)
    FT = typeof(p1)
    t1 = BigFloat(t1) # Avoid errors when t1 ≈ t2.
    t2 = BigFloat(t2)
    return FT(
        (v1 * t2 - v2 * t1) / (t2 - t1) +
        (v2 - v1) * (p1 + p2) * (t2 - t1) / (2 * (
            (p2 - p1) * (t2 - t1) +
            (p1 * t2 - p2 * t1) * (log(t2) - log(t1))
        ))
    )
end
function lay_var(::CenterOfMass{<:Union{Temp, Pres}}, p1, p2, t1, t2, v1, v2)
    C1 = p1 / t1
    C2 = p2 / t2
    return (v1 * (2C1 + C2) + v2 * (C1 + 2C2)) / (3 * (C1 + C2))
end
lay_var(::CenterOfPres{<:Union{Dens, Temp}}, p1, p2, t1, t2, v1, v2) =
    (v1 * (2p1 + p2) + v2 * (p1 + 2p2)) / (3 * (p1 + p2))
function lay_var(::CenterOfPres{Pres}, p1, p2, t1, t2, v1, v2)
    C1 = p1 / t1
    C2 = p2 / t2
    return (
        v1 * (t1 * (3C1 + C2) + t2 * (C1 + C2)) +
        v2 * (t1 * (C1 + C2) + t2 * (C1 + 3C2))
    ) / (2 * (t1 * (2C1 + C2) + t2 * (C1 + 2C2)))
end

################## Defining the Radiation Model and Tendency ##################

# Define different upper boundary conditions for RRTMGP.
abstract type RRTMGPUpperBC end
struct DefaultUpperRRTMGPBC <: RRTMGPUpperBC end
struct ConstantUpperRRTMGPBC <: RRTMGPUpperBC
    ext_nlay::Int
    change_ext_layer_init::Bool
end
struct IsothermalUpperRRTMGPBC <: RRTMGPUpperBC
    ext_nlay::Int
    change_ext_layer_init::Bool
end
struct UpwardRRTMGPDomainExtension{FTN} <: RRTMGPUpperBC
    ext_nlay::Int
    change_ext_layer_init::Bool
    update_bottom_ext_layer::Bool
    ∂T∂p::FTN
end

# Define a new radiation model that utilizes RRTMGP.
struct RRTMGPModel{TT, BCT, LIT, PT, IT, FT} <: RadiationModel
    tendency_type::TT
    upper_bc::BCT
    layer_interp::LIT
    optics_symb::Symbol
    optical_props_symb::Symbol
    pressure_profile::PT
    init_rrtmgp_extension::IT
    vmr_co2_override::FT
end

# Allocate space in which to unload the net energy flux calculated by RRTMGP.
# If necessary, also allocate space for the derivative of the flux.
vars_state(::RRTMGPModel, ::Auxiliary, FT) = @vars(flux::FT)
vars_state(::RRTMGPModel{Source}, ::Auxiliary, FT) = @vars(
    flux::FT,
    src_dg::FT, src_deriv2::FT, src_deriv4::FT, src_deriv6::FT,
)

# Define a new tendency for radiation.
struct Radiation{TT} <: TendencyDef{TT} end

# Add the radiation flux to the net energy flux.
eq_tends(::Energy, ::RRTMGPModel{Flux{OT}}, ::Flux{OT}) where {OT} =
    (Radiation{Flux{OT}}(),)
function flux(::Energy, ::Radiation, bl, args)
    @unpack aux = args
    return aux.radiation.flux * vertical_unit_vector(bl, aux)
end

# Add the radiation source to the net energy source.
Radiation() = Radiation{Source}()
prognostic_vars(::Radiation) = (Energy(),)
function source(::Energy, ::Radiation, bl, args)
    @unpack aux = args
    return aux.radiation.src_deriv6
end

####################### Defining Simple Moisture Models #######################

# Define new moisture models for debugging RRTMGP.
struct CappedMonotonicRelativeHumidity{FT} <: AbstractMoistureModel
    max_relhum::FT
    z_cutoff::FT
end
struct CappedConstantSpecificHumidity <: AbstractMoistureModel end

# Allocate space to store the specific humidity for the model in which it is
# constant.
vars_state(::CappedConstantSpecificHumidity, ::Prognostic, FT) =
    @vars(q_vap::FT)

# Do not create prognostic variable tendencies for the new moisture models.
prognostic_vars(::AbstractMoistureModel) = ()

# Treat each of the new models as a DryModel for thermodynamics calculations.
new_thermo_state(atmos, energy, ::AbstractMoistureModel, state, aux) =
    new_thermo_state(atmos, energy, DryModel(), state, aux)
new_thermo_state_anelastic(atmos, energy, ::AbstractMoistureModel, state, aux) =
    new_thermo_state_anelastic(atmos, energy, DryModel(), state, aux)

# Define a function to get the volume mixing ratio of water vapor from any
# moisture model. For CappedMonotonicRelativeHumidity, this function modifies
# min_q_vap.
function vmr_h2o!(
    m::AbstractMoistureModel,
    param_set,
    thermo_state,
    prog,
    aux,
    min_q_vap,
)
    return vol_vapor_mixing_ratio(param_set, PhasePartition(thermo_state))
end
function vmr_h2o!(
    m::CappedMonotonicRelativeHumidity,
    param_set,
    thermo_state,
    prog,
    aux,
    min_q_vap,
)
    q_vap = m.max_relhum * q_vap_saturation(thermo_state)
    if aux.coord[3] > m.z_cutoff
        if q_vap < min_q_vap[1]
            min_q_vap[1] = q_vap
        else
            q_vap = min_q_vap[1]
        end
    end
    return vol_vapor_mixing_ratio(param_set, PhasePartition(q_vap))
end
function vmr_h2o!(
    m::CappedConstantSpecificHumidity,
    param_set,
    thermo_state,
    prog,
    aux,
    min_q_vap,
)
    q_vap = prog.moisture.q_vap
    qvap_sat = q_vap_saturation(thermo_state)
    if q_vap > qvap_sat
        q_vap = qvap_sat
    end
    return vol_vapor_mixing_ratio(param_set, PhasePartition(q_vap))
end


###################### Initializing Modeldata for RRTMGP ######################

# Define the modeldata for RRTMGP.
struct RRTMGPData{ST <: Solver, MT}
    solver::ST
    misc::MT
end
Base.getproperty(data::RRTMGPData, name::Symbol) = name == :solver ?
    getfield(data, :solver) : getproperty(getfield(data, :misc), name)

# Initialize everything in the atmospheric state that does not vary with
# elevation.
function atmospheric_state(DA, FT, ds_input, nlev, ncol, zs)
    as = GrayAtmosphericState(
        similar(DA, FT, nlev - 1, ncol), # p_lay
        similar(DA, FT, nlev, ncol),     # p_lev
        similar(DA, FT, nlev - 1, ncol), # t_lay
        similar(DA, FT, nlev, ncol),     # t_lev
        similar(DA, FT, nlev, ncol),     # z_lev
        similar(DA, FT, ncol),           # t_sfc
        FT(3.5),                         # α
        similar(DA, FT, ncol),           # d0
        nlev - 1,                        # nlay
        ncol,                            # ncol
    )

    # Initialize the surface temperature array.
    # TODO: Get this value from the land model.
    t_sfc = FT(ds_input["surface_temperature"][site_index, expt_index])
    as.t_sfc .= t_sfc

    # Initialize the optical thickness parameter array.
    as.d0 .= (t_sfc / FT(200))^4 - FT(1)

    # Initialize the z-coordinate array.
    as.z_lev .= zs

    # Note that arrays related to air temperature and pressure (as.p_lay,
    # as.p_lev, as.t_lay, and as.t_lev) are not initialized.

    # TODO: remove after debugging.
    as.p_lay .= FT(NaN)
    as.p_lev .= FT(NaN)
    as.t_lay .= FT(NaN)
    as.t_lev .= FT(NaN)

    return as
end
function atmospheric_state(
    DA,
    FT,
    ds_input,
    nlev,
    ncol,
    ngases,
    idx_gases,
    pressure_profile,
    zs,
    vmr_co2_override,
)
    as = AtmosphericState(
        similar(DA, FT, ncol),               # lon
        similar(DA, FT, ncol),               # lat
        similar(DA, FT, nlev - 1, ncol),     # p_lay
        similar(DA, FT, nlev, ncol),         # p_lev
        similar(DA, FT, nlev - 1, ncol),     # t_lay
        similar(DA, FT, nlev, ncol),         # t_lev
        similar(DA, FT, ncol),               # t_sfc
        similar(DA, FT, nlev - 1, ncol),     # col_dry
        Vmr(
            similar(DA, FT, nlev - 1, ncol), # vmr_h2o
            similar(DA, FT, nlev - 1, ncol), # vmr_o3
            similar(DA, FT, ngases),         # vmr
        ),
        nlev - 1,                            # nlay
        ncol,                                # ncol
        ngases,                              # ngas
    )

    # Initialize the longitude and latitude arrays.
    # TODO: Get these values from the orientation model.
    as.lon .= FT(ds_input["lon"][site_index])
    as.lat .= FT(ds_input["lat"][site_index])

    # Initialize the surface temperature array.
    # TODO: Get this value from the land model.
    as.t_sfc .= FT(ds_input["surface_temperature"][site_index, expt_index])

    # Initialize the volume mixing ratio scalars.
    # Unused input gases:
    #     "methyl_bromide_GM", "cfc113_GM", "hfc245fa_GM", "hcfc142b_GM",
    #     "cfc12eq_GM", "sf6_GM", "hfc365mfc_GM", "halon2402_GM",
    #     "hfc4310mee_GM", "halon1301_GM", "hfc152a_GM", "cfc115_GM",
    #     "methyl_chloride_GM", "c7f16_GM", "c4f10_GM", "nf3_GM",
    #     "hfc134aeq_GM", "so2f2_GM", "hfc236fa_GM", "ch2cl2_GM", "c_c4f8_GM",
    #     "cfc114_GM", "halon1211_GM", "c8f18_GM", "hfc227ea_GM", "ch3ccl3_GM",
    #     "c5f12_GM", "c3f8_GM", "c6f14_GM", "hcfc141b_GM", "chcl3_GM",
    #     "c2f6_GM", "cfc11eq_GM",
    as.vmr.vmr .= FT(0)
    for (lookup_gas_name, input_gas_name) in (
        # ("h2o", "water_vapor"),            # overwritten by vmr_h2o
        ("co2", "carbon_dioxide_GM"),
        # ("o3", "ozone"),                   # overwritten by vmr_o3
        ("n2o", "nitrous_oxide_GM"),
        ("co", "carbon_monoxide_GM"),
        ("ch4", "methane_GM"),
        ("o2", "oxygen_GM"),
        ("n2", "nitrogen_GM"),
        ("ccl4", "carbon_tetrachloride_GM"),
        ("cfc11", "cfc11_GM"),
        ("cfc12", "cfc12_GM"),
        ("cfc22", "hcfc22_GM"),
        ("hfc143a", "hfc143a_GM"),
        ("hfc125", "hfc125_GM"),
        ("hfc23", "hfc23_GM"),
        ("hfc32", "hfc32_GM"),
        ("hfc134a", "hfc134a_GM"),
        ("cf4", "cf4_GM"),
        # ("no2", nothing),                  # not available in input dataset
    )
        as.vmr.vmr[idx_gases[lookup_gas_name]] =
            FT(ds_input[input_gas_name][expt_index]) *
            parse(FT, ds_input[input_gas_name].attrib["units"])
    end

    if !isnothing(vmr_co2_override)
        as.vmr.vmr[idx_gases["co2"]] = vmr_co2_override
    end

    ps = Array{FT}(ds_input["pres_layer"])[:, site_index]
    o3s = Array{FT}(ds_input["ozone"])[:, site_index, expt_index]
    vmr_o3_interp = Spline1D(ps, o3s; k = 1, bc = "extrapolate", s = FT(0))

    p_lev = map(pressure_profile, zs)
    p_lay = (p_lev[1:end - 1] .+ p_lev[2:end]) .* FT(0.5)
    as.vmr.vmr_o3 .= map(vmr_o3_interp, p_lay)

    # Note that arrays related to air temperature, pressure, and water content
    # (as.p_lay, as.p_lev, as.t_lay, as.t_lev, as.col_dry, and as.vmr.vmr_h2o)
    # are not initialized.

    # TODO: remove after debugging.
    as.p_lay .= FT(NaN)
    as.p_lev .= FT(NaN)
    as.t_lay .= FT(NaN)
    as.t_lev .= FT(NaN)
    as.col_dry .= FT(NaN)
    as.vmr.vmr_h2o .= FT(NaN)

    return as
end

# Initialize everything in the RRTMGP boundary conditions except for the
# incoming longwave flux.
function rrtmgp_bcs(DA, FT, ds_input, ncol, ngpt_lw, change_inc_flux)
    bcs_lw = LwBCs(
        similar(DA, FT, ncol),                                      # sfc_emis
        change_inc_flux ? similar(DA, FT, ncol, ngpt_lw) : nothing, # inc_flux
    )
    bcs_sw = SwBCs(
        DA,
        FT,
        similar(DA, FT, ncol), # zenith
        similar(DA, FT, ncol), # toa_flux
        similar(DA, FT, ncol), # sfc_alb_direct
        nothing,               # inc_flux_diffuse
        # nothing,               # sfc_alb_diffuse
        similar(DA, FT, ncol), # sfc_alb_diffuse
    )

    # Initialize the boundary conditions.
    # TODO: Get these values from Insolation.jl and the land model.
    bcs_lw.sfc_emis .= FT(ds_input["surface_emissivity"][site_index])
    bcs_sw.zenith .=
        FT(π) / FT(180) * FT(ds_input["solar_zenith_angle"][site_index])
    bcs_sw.toa_flux .= FT(ds_input["total_solar_irradiance"][site_index])
    # bcs_sw.zenith .= FT(0)
    # bcs_sw.toa_flux .= FT(700)
    bcs_sw.sfc_alb_direct .= FT(ds_input["surface_albedo"][site_index])

    # TODO: Temporary fix
    bcs_sw.sfc_alb_diffuse .= FT(ds_input["surface_albedo"][site_index])

    # Note that bcs_lw.inc_flux is not initialized.

    return bcs_lw, bcs_sw
end

# Initialize everything in the RRTMGP modeldata that does not vary with time.
function RRTMGPData(atmos::AtmosModel, grid::AbstractGrid)
    radiation = radiation_model(atmos)
    info = basic_grid_info(grid)

    return RRTMGPData(
        arraytype(grid),
        eltype(grid.vgeo),
        get_z(grid; rm_dupes = true),
        info.Nqh * info.nhorzelem,
        radiation.upper_bc,
        radiation.layer_interp,
        radiation.optics_symb,
        radiation.optical_props_symb,
        radiation.pressure_profile,
        radiation.init_rrtmgp_extension,
        radiation.vmr_co2_override,
        parameter_set(atmos),
    )
end
function RRTMGPData(
    DA,
    FT,
    zs,
    ncol,
    upper_bc,
    layer_interp,
    optics_symb,
    optical_props_symb,
    pressure_profile,
    init_rrtmgp_extension,
    vmr_co2_override,
    param_set,
)
    if upper_bc isa UpwardRRTMGPDomainExtension
        @assert zs[end] < FT(z_toa)
        ext_zs = range(zs[end], FT(z_toa); length = upper_bc.ext_nlay + 1)
        zs = [zs..., ext_zs[2:end]...]
    end

    nlev = size(zs, 1)

    # TODO: Remove after debugging.
    misc = (z_lev = zs, z_lay = (zs[1:end - 1] .+ zs[2:end]) .* FT(0.5))

    ds_input = Dataset(ds_input_path)

    if optics_symb == :Gray
        as = atmospheric_state(DA, FT, ds_input, nlev, ncol, zs)
        ngpt_lw = 1
        ngpt_sw = 1
        fluxb_lw = nothing
        fluxb_sw = nothing
    else
        ds_lw = Dataset(ds_lw_path)
        lookup_lw, idx_gases_lw = LookUpLW(ds_lw, Int, FT, DA)
        close(ds_lw)
        ds_sw = Dataset(ds_sw_path)
        lookup_sw, idx_gases_sw = LookUpSW(ds_sw, Int, FT, DA)
        close(ds_sw)
        @assert idx_gases_lw == idx_gases_sw
        as = atmospheric_state(
            DA,
            FT,
            ds_input,
            nlev,
            ncol,
            lookup_lw.n_gases,
            idx_gases_lw,
            pressure_profile,
            zs,
            vmr_co2_override,
        )
        ngpt_lw = lookup_lw.n_gpt
        ngpt_sw = lookup_sw.n_gpt
        fluxb_lw = FluxLW(ncol, nlev - 1, FT, DA)
        fluxb_sw = init_flux_sw(ncol, nlev - 1, FT, DA, optical_props_symb)
        misc = (
            misc...,
            lookup_lw = lookup_lw,
            lookup_sw = lookup_sw,
            vmr_h2o_lev = similar(DA, FT, nlev, ncol),
        )
    end

    # All layers must be initialized for TwoStream with change_inc_flux = true.
    if upper_bc isa UpwardRRTMGPDomainExtension
        init_extension!(
            as,
            misc,
            init_rrtmgp_extension,
            zs,
            nlev - 1, # upper_bc.ext_nlay,
            upper_bc.change_ext_layer_init,
            layer_interp,
        )
    end

    change_inc_flux =
        upper_bc isa ConstantUpperRRTMGPBC ||
        upper_bc isa IsothermalUpperRRTMGPBC
    bcs_lw, bcs_sw =
        rrtmgp_bcs(DA, FT, ds_input, ncol, ngpt_lw, change_inc_flux)

    # Initialize the incoming longwave flux array and modify the incoming
    # shortwave flux array, if needed.
    if change_inc_flux
        ext_upper_bc = UpwardRRTMGPDomainExtension(
            upper_bc.ext_nlay,
            upper_bc.change_ext_layer_init,
            false, # this value is irrelevant here
            nothing,
        )

        ext_rrtmgp_data = RRTMGPData(
            DA,
            FT,
            zs,
            ncol,
            ext_upper_bc,
            layer_interp,
            optics_symb,
            optical_props_symb,
            pressure_profile,
            init_rrtmgp_extension,
            vmr_co2_override,
            param_set,
        )

        update_col_dry!(ext_rrtmgp_data.solver.as, param_set)

        if upper_bc isa ConstantUpperRRTMGPBC
            for igpt in 1:ngpt_lw
                unload_flux_dn_lw!(
                    ext_rrtmgp_data,
                    bcs_lw.inc_flux,
                    igpt,
                    nlev,
                )
            end
        else # IsothermalUpperRRTMGPBC
            ext_τ_lw = similar(DA, FT, ncol, ngpt_lw)
            ext_τ_lw .= FT(0)
            for igpt in 1:ngpt_lw
                update_τ_lw!(ext_rrtmgp_data, igpt)
                for ext_ilay in 1:upper_bc.ext_nlay
                    ext_τ_lw[:, igpt] .+=
                        ext_rrtmgp_data.solver.op.τ[nlev - 1 + ext_ilay, :]
                end
            end
            misc = (misc..., ext_τ_lw = ext_τ_lw)
        end

        ext_τ_sw = similar(DA, FT, ncol)
        transmittance = ext_τ_sw # add an alias for clarity
        for igpt in 1:ngpt_sw
            update_τ_sw!(ext_rrtmgp_data, igpt)
            ext_τ_sw .= FT(0)
            for ext_ilay in 1:upper_bc.ext_nlay
                ext_τ_sw .+=
                    ext_rrtmgp_data.solver.op.τ[nlev - 1 + ext_ilay, :]
            end
            transmittance .= exp.(-ext_τ_sw ./ cos.(bcs_sw.zenith))
            if optics_symb == :Gray
                bcs_sw.toa_flux .*= transmittance
            else
                # TODO: This won't work if there is horizontal variance.
                lookup_sw.solar_src_scaled[igpt] *= sum(transmittance) / ncol
            end
        end
    end

    close(ds_input)
    
    solver = Solver(
        as,
        init_optical_props(optical_props_symb, FT, DA, ncol, nlev - 1),
        source_func_longwave(FT, ncol, nlev - 1, ngpt_lw, optical_props_symb, DA),
        source_func_shortwave(FT, ncol, nlev - 1, optical_props_symb, DA),
        bcs_lw,
        bcs_sw,
        AngularDiscretization(optical_props_symb, FT, ngaussangles, DA),
        fluxb_lw,
        fluxb_sw,
        FluxLW(ncol, nlev - 1, FT, DA),
        init_flux_sw(ncol, nlev - 1, FT, DA, optical_props_symb),
    )

    return RRTMGPData(solver, misc)
end

######################### Helpful Wrappers for RRTMGP #########################

# TODO: Clean up this entire section.

# A quick and dirty replacement for init_state_prognostic within RRTMGP.
function init_extension!(
    as,
    misc,
    init_rrtmgp_extension,
    zs,
    ext_nlay,
    change_ext_layer_init,
    layer_interp,
)
    FT = eltype(zs)
    nlev = length(zs)
    for ilev in (nlev - ext_nlay):nlev
        z_lev = zs[ilev]
        lev_data = init_rrtmgp_extension(z_lev)
        as.p_lev[ilev, :] .= lev_data.p
        as.t_lev[ilev, :] .= lev_data.T
        if as isa AtmosphericState
            misc.vmr_h2o_lev[ilev, :] .= lev_data.vmr_h2o
        end
    end
    for ilay in (nlev - ext_nlay):nlev - 1
        if change_ext_layer_init
            # p_lay = (as.p_lev[ilay, 1] + as.p_lev[ilay + 1, 1]) * FT(0.5)

            p_lay = map(
                (args...) -> lay_pres(layer_interp, args...),
                as.p_lev[ilay, :], as.p_lev[ilay + 1, :],
                as.t_lev[ilay, :], as.t_lev[ilay + 1, :],
            )
            as.p_lay[ilay, :] .= p_lay
            as.t_lay[ilay, :] .= map(init_rrtmgp_extension.T_interp, p_lay)
            if as isa AtmosphericState
                as.vmr.vmr_h2o[ilay, :] .=
                    map(init_rrtmgp_extension.vmr_h2o_interp, p_lay)
            end
        else
            # as.p_lay[ilay, :] .=
            #     (as.p_lev[ilay, :] .+ as.p_lev[ilay + 1, :]) .* FT(0.5)
            # as.t_lay[ilay, :] .=
            #     (as.t_lev[ilay, :] .+ as.t_lev[ilay + 1, :]) .* FT(0.5)
            # if as isa AtmosphericState
            #     as.vmr.vmr_h2o[ilay, :] .= (
            #         misc.vmr_h2o_lev[ilay, :] .+
            #         misc.vmr_h2o_lev[ilay + 1, :]
            #     ) .* FT(0.5)
            # end
            as.p_lay[ilay, :] .= map(
                (args...) -> lay_pres(layer_interp, args...),
                as.p_lev[ilay, :], as.p_lev[ilay + 1, :],
                as.t_lev[ilay, :], as.t_lev[ilay + 1, :],
            )
            as.t_lay[ilay, :] .= map(
                (args...) -> lay_temp(layer_interp, args...),
                as.p_lev[ilay, :], as.p_lev[ilay + 1, :],
                as.t_lev[ilay, :], as.t_lev[ilay + 1, :],
            )
            if as isa AtmosphericState
                as.vmr.vmr_h2o[ilay, :] .= map(
                    (args...) -> lay_var(layer_interp, args...),
                    as.p_lev[ilay, :], as.p_lev[ilay + 1, :],
                    as.t_lev[ilay, :], as.t_lev[ilay + 1, :],
                    misc.vmr_h2o_lev[ilay, :], misc.vmr_h2o_lev[ilay + 1, :],
                )
            end
        end
    end
end

# Update the value of col_dry after modifying vmr_h2o.
function update_col_dry!(as::GrayAtmosphericState, param_set) end
update_col_dry!(as::AtmosphericState, param_set) =
    compute_col_dry!(
        as.p_lev,
        as.t_lay,
        as.col_dry,
        param_set;
        vmr_h2o = as.vmr.vmr_h2o,
        lat = as.lat,
        max_threads = max_threads,
    )

# Update the longwave optical depth of each layer.
update_τ_lw!(rrtmgp_data, igpt) =
    update_τ_lw!(rrtmgp_data.solver.as, rrtmgp_data, igpt)
update_τ_lw!(::GrayAtmosphericState, rrtmgp_data, igpt) =
    compute_optical_props!(
        rrtmgp_data.solver.as,
        rrtmgp_data.solver.op;
        islw = true,
        sf = rrtmgp_data.solver.src_lw,
    )
update_τ_lw!(::AtmosphericState, rrtmgp_data, igpt) =
    compute_optical_props!(
        rrtmgp_data.solver.as,
        rrtmgp_data.lookup_lw,
        rrtmgp_data.solver.op,
        igpt;
        islw = true,
        sf = rrtmgp_data.solver.src_lw,
        max_threads = max_threads,
    )

# Update the shortwave optical depth of each layer.
update_τ_sw!(rrtmgp_data, igpt) =
    update_τ_sw!(rrtmgp_data.solver.as, rrtmgp_data, igpt)
update_τ_sw!(::GrayAtmosphericState, rrtmgp_data, igpt) =
    compute_optical_props!(
        rrtmgp_data.solver.as,
        rrtmgp_data.solver.op;
        islw = false,
        sf = rrtmgp_data.solver.src_sw,
    )
update_τ_sw!(::AtmosphericState, rrtmgp_data, igpt) =
    compute_optical_props!(
        rrtmgp_data.solver.as,
        rrtmgp_data.lookup_sw,
        rrtmgp_data.solver.op,
        igpt;
        islw = false,
        sf = rrtmgp_data.solver.src_sw,
        max_threads = max_threads,
    )

# This is only used for ConstantUpperRRTMGPBC.
function unload_flux_dn_lw!(rrtmgp_data, inc_flux, igpt, ilev)
    isgray = rrtmgp_data.solver.as isa GrayAtmosphericState
    update_τ_lw!(rrtmgp_data.solver.as, rrtmgp_data, igpt)
    if rrtmgp_data.solver.op isa OneScalar
        rte_lw_noscat_solve!(
            rrtmgp_data.solver,
            isgray = isgray,
            igpt = igpt,
            max_threads = max_threads,
        )
    else
        rte_lw_2stream_solve!(
            rrtmgp_data.solver,
            isgray = isgray,
            igpt = igpt,
            max_threads = max_threads,
        )
    end
    if isgray
        flux_dn_lw = rrtmgp_data.solver.flux_lw.flux_dn
    else
        flux_dn_lw = rrtmgp_data.solver.fluxb_lw.flux_dn
    end
    inc_flux[:, igpt] .= flux_dn_lw[ilev, :]
end

# Update the longwave and shortwave fluxes. Use a custom wrapper for the
# longwave solver for when the upper boundary condition requires modifying the
# incoming longwave flux for each g-point.
update_fluxes!(rrtmgp_data, upper_bc) =
    update_fluxes!(rrtmgp_data.solver.as, rrtmgp_data, upper_bc)
function update_fluxes!(
    as::GrayAtmosphericState,
    rrtmgp_data,
    upper_bc,
)
    solve_lw_bc_wrapper!(rrtmgp_data, upper_bc)
    solve_sw!(rrtmgp_data.solver; max_threads = max_threads)
end
function update_fluxes!(
    as::AtmosphericState,
    rrtmgp_data,
    upper_bc,
)
    solve_lw_bc_wrapper!(rrtmgp_data, upper_bc)
    solve_sw!(
        rrtmgp_data.solver,
        rrtmgp_data.lookup_sw;
        max_threads = max_threads,
    )
end

function solve_lw_bc_wrapper!(rrtmgp_data, ::RRTMGPUpperBC)
    if rrtmgp_data.solver.as isa GrayAtmosphericState
        solve_lw!(rrtmgp_data.solver; max_threads = max_threads)
    else
        solve_lw!(
            rrtmgp_data.solver,
            rrtmgp_data.lookup_lw;
            max_threads = max_threads,
        )
    end
end
function solve_lw_bc_wrapper!(rrtmgp_data, ::IsothermalUpperRRTMGPBC)
    isgray = rrtmgp_data.solver.as isa GrayAtmosphericState
    if isgray
        ngpt_lw = 1
    else
        ngpt_lw = rrtmgp_data.lookup_lw.n_gpt
        set_flux_to_zero!(rrtmgp_data.solver.flux_lw)
    end
    for igpt = 1:ngpt_lw
        update_τ_lw!(rrtmgp_data.solver.as, rrtmgp_data, igpt)
        d = rrtmgp_data.solver.angle_disc.gauss_Ds[1]
        FT = typeof(d)
        w = FT(2) * FT(π) * rrtmgp_data.solver.angle_disc.gauss_wts[1]
        rrtmgp_data.solver.bcs_lw.inc_flux[:, igpt] .=
            w .* (FT(1) .- exp.(-rrtmgp_data.ext_τ_lw[:, igpt] .* d)) .*
            rrtmgp_data.solver.src_lw.lev_source_inc[end, :]
        if rrtmgp_data.solver.op isa OneScalar
            rte_lw_noscat_solve!(
                rrtmgp_data.solver,
                isgray = isgray,
                igpt = igpt,
                max_threads = max_threads,
            )
        else
            rte_lw_2stream_solve!(
                rrtmgp_data.solver,
                isgray = isgray,
                igpt = igpt,
                max_threads = max_threads,
            )
        end
        if !isgray
            add_to_flux!(
                rrtmgp_data.solver.flux_lw,
                rrtmgp_data.solver.fluxb_lw,
            )
        end
    end
end

######################### Kernel Launchers for RRTMGP #########################

# Update the RRTMGP data, use it to calculate radiation energy fluxes, and
# unload those fluxes into the auxiliary state.
function update_auxiliary_state!(
    spacedisc::SpaceDiscretization,
    radiation::RRTMGPModel,
    atmos::BalanceLaw,
    state_prognostic::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    rrtmgp_data = spacedisc.modeldata.rrtmgp_data
    FT = eltype(state_prognostic)
    device = array_device(state_prognostic)
    grid = spacedisc.grid
    info = basic_grid_info(grid)
    horzelems =
        fld1(first(elems), info.nvertelem):fld1(last(elems), info.nvertelem)

    # Load the new temperature, pressure, and moisture into RRTMGP.
    lev_arrays = (rrtmgp_data.solver.as.p_lev, rrtmgp_data.solver.as.t_lev)
    lay_arrays = (rrtmgp_data.solver.as.p_lay, rrtmgp_data.solver.as.t_lay)
    if rrtmgp_data.solver.as isa AtmosphericState
        lev_arrays = (lev_arrays..., rrtmgp_data.vmr_h2o_lev)
        lay_arrays = (lay_arrays..., rrtmgp_data.solver.as.vmr.vmr_h2o)
    end
    event = Event(device)
    event = kernel_load_into_rrtmgp!(device, (info.Nq[1], info.Nq[2]))(
        atmos,
        map(Val, info.Nq)...,
        Val(info.nvertelem),
        state_prognostic.data,
        spacedisc.state_auxiliary.data,
        horzelems,
        lev_arrays,
        lay_arrays,
        radiation.layer_interp;
        ndrange = (info.nhorzelem * info.Nq[1], info.Nq[2]),
        dependencies = (event,),
    )
    wait(device, event)

    # TODO: Move this into the above kernel call.
    if (
        radiation.upper_bc isa UpwardRRTMGPDomainExtension &&
        radiation.upper_bc.update_bottom_ext_layer
    )
        ilay_ext_bot =
            rrtmgp_data.solver.as.nlay - radiation.upper_bc.ext_nlay + 1
        if !isnothing(radiation.upper_bc.∂T∂p)
            for ilev in ilay_ext_bot + 1:rrtmgp_data.solver.as.nlay + 1
                rrtmgp_data.solver.as.t_lev[ilev, :] .=
                    rrtmgp_data.solver.as.t_lev[ilev - 1, :] .-
                    radiation.upper_bc.∂T∂p .* (
                        rrtmgp_data.solver.as.p_lev[ilev - 1, :] .-
                        rrtmgp_data.solver.as.p_lev[ilev, :]
                    )
            end
        end
        layer_interp = radiation.layer_interp
        p_lev = rrtmgp_data.solver.as.p_lev
        t_lev = rrtmgp_data.solver.as.t_lev
        rrtmgp_data.solver.as.p_lay[ilay_ext_bot, :] .= map(
            (args...) -> lay_pres(layer_interp, args...),
            p_lev[ilay_ext_bot, :], p_lev[ilay_ext_bot + 1, :],
            t_lev[ilay_ext_bot, :], t_lev[ilay_ext_bot + 1, :],
        )
        rrtmgp_data.solver.as.t_lay[ilay_ext_bot, :] .= map(
            (args...) -> lay_temp(layer_interp, args...),
            p_lev[ilay_ext_bot, :], p_lev[ilay_ext_bot + 1, :],
            t_lev[ilay_ext_bot, :], t_lev[ilay_ext_bot + 1, :],
        )
        if rrtmgp_data.solver.as isa AtmosphericState
            vmr_h2o_lev = rrtmgp_data.vmr_h2o_lev
            rrtmgp_data.solver.as.vmr.vmr_h2o[ilay_ext_bot, :] .= map(
                (args...) -> lay_var(layer_interp, args...),
                p_lev[ilay_ext_bot, :], p_lev[ilay_ext_bot + 1, :],
                t_lev[ilay_ext_bot, :], t_lev[ilay_ext_bot + 1, :],
                vmr_h2o_lev[ilay_ext_bot, :], vmr_h2o_lev[ilay_ext_bot + 1, :],
            )
        end
        # for i in 1:length(lay_arrays)
        #     lay_arrays[i][ilay_ext_bot, :] .= (
        #         lev_arrays[i][ilay_ext_bot, :] .+
        #         lev_arrays[i][ilay_ext_bot + 1, :]
        #     ) .* FT(0.5)
        # end
    end

    # TODO: Maybe also move this into the above kernel call. Also, avoid
    # recomputing constant values for UpwardRRTMGPDomainExtension.
    update_col_dry!(rrtmgp_data.solver.as, parameter_set(atmos))
    update_fluxes!(rrtmgp_data, radiation.upper_bc)

    aux_vars = vars(spacedisc.state_auxiliary)

    # Update the radiation energy flux array in the auxiliary state.
    event = Event(device)
    event = kernel_unload_from_rrtmgp!(device, (1, info.Nq[1], info.Nq[2]))(
        map(Val, info.Nq)...,
        Val(info.nvertelem),
        Val(varsindex(aux_vars, :radiation, :flux)[1]),
        spacedisc.state_auxiliary.data,
        rrtmgp_data.solver.flux_lw,
        rrtmgp_data.solver.flux_sw,
        horzelems;
        ndrange = (info.nvertelem, info.nhorzelem * info.Nq[1], info.Nq[2]),
        dependencies = (event,),
    )
    wait(device, event)

    # Use the radiation flux to compute a radiation source term.
    if radiation.tendency_type isa Source
        event = Event(device)
        event = kernel_flux_source!(device, (info.Nq[1], info.Nq[2]))(
            atmos,
            map(Val, info.Nq)...,
            Val(info.nvertelem),
            Val(varsindex(aux_vars, :radiation, :flux)[1]),
            Val(varsindex(aux_vars, :radiation, :src_dg)[1]),
            Val(varsindex(aux_vars, :radiation, :src_deriv2)[1]),
            Val(varsindex(aux_vars, :radiation, :src_deriv4)[1]),
            Val(varsindex(aux_vars, :radiation, :src_deriv6)[1]),
            spacedisc.state_auxiliary.data,
            horzelems,
            grid;
            ndrange = (info.nhorzelem * info.Nq[1], info.Nq[2]),
            dependencies = (event,),
        )
        wait(device, event)
    end
end

######################### Kernel Functions for RRTMGP #########################

function rrtmgp_nodal_data!(::NTuple{2}, atmos, prog, aux, min_q_vap)
    thermo_state = recover_thermo_state(atmos, prog, aux)
    return (p = air_pressure(thermo_state), T = air_temperature(thermo_state))
end
function rrtmgp_nodal_data!(::NTuple{3}, atmos, prog, aux, min_q_vap)
    thermo_state = recover_thermo_state(atmos, prog, aux)
    return (
        p = air_pressure(thermo_state),
        T = air_temperature(thermo_state),
        vmr_h2o = vmr_h2o!(
            atmos.moisture,
            parameter_set(atmos),
            thermo_state,
            prog,
            aux,
            min_q_vap,
        ),
    )
end

@Base.propagate_inbounds function lev_set!(nd, v, h, p_lev, t_lev)
    p_lev[v, h] = nd.p
    t_lev[v, h] = nd.T
end
@Base.propagate_inbounds function lev_set!(nd, v, h, p_lev, t_lev, vmr_h2o_lev)
    p_lev[v, h] = nd.p
    t_lev[v, h] = nd.T
    vmr_h2o_lev[v, h] = nd.vmr_h2o
end

@Base.propagate_inbounds function lev_avg!(FT, nd, v, h, p_lev, t_lev)
    p_lev[v, h] = (p_lev[v, h] + nd.p) * FT(0.5)
    t_lev[v, h] = (t_lev[v, h] + nd.T) * FT(0.5)
end
@Base.propagate_inbounds function lev_avg!(FT, nd, v, h, p_lev, t_lev, vmr_h2o_lev)
    p_lev[v, h] = (p_lev[v, h] + nd.p) * FT(0.5)
    t_lev[v, h] = (t_lev[v, h] + nd.T) * FT(0.5)
    vmr_h2o_lev[v, h] = (vmr_h2o_lev[v, h] + nd.vmr_h2o) * FT(0.5)
end

@Base.propagate_inbounds function lev_to_lay!(FT, v, h, lev_arrays, lay_arrays, layer_interp)
    # for i in 1:length(lev_arrays)
    #     lay_arrays[i][v, h] =
    #         (lev_arrays[i][v, h] + lev_arrays[i][v + 1, h]) * FT(0.5)
    # end
    p_lev = lev_arrays[1]
    t_lev = lev_arrays[2]
    lay_arrays[1][v, h] = lay_pres(
        layer_interp,
        p_lev[v, h], p_lev[v + 1, h],
        t_lev[v, h], t_lev[v + 1, h],
    )
    lay_arrays[2][v, h] = lay_temp(
        layer_interp,
        p_lev[v, h], p_lev[v + 1, h],
        t_lev[v, h], t_lev[v + 1, h],
    )
    if length(lev_arrays) == 3
        vmr_h2o_lev = lev_arrays[3]
        lay_arrays[3][v, h] = lay_var(
            layer_interp,
            p_lev[v, h], p_lev[v + 1, h],
            t_lev[v, h], t_lev[v + 1, h],
            vmr_h2o_lev[v, h], vmr_h2o_lev[v + 1, h],
        )
    end
end

@kernel function kernel_load_into_rrtmgp!(
    atmos::BalanceLaw,
    ::Val{Nq1}, ::Val{Nq2}, ::Val{1}, ::Val{nvertelem},
    state_prognostic,
    state_auxiliary,
    elems,
    lev_arrays,
    lay_arrays,
    layer_interp,
) where {Nq1, Nq2, nvertelem}
    @uniform begin
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(atmos, Prognostic())
        num_state_auxiliary = number_states(atmos, Auxiliary())
        local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        prog = Vars{vars_state(atmos, Prognostic(), FT)}(
            local_state_prognostic,
        )
        aux = Vars{vars_state(atmos, Auxiliary(), FT)}(
            local_state_auxiliary,
        )
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)
    min_q_vap = @private FT (1)

    @inbounds begin
        min_q_vap[1] = FT(Inf)

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
            nd = rrtmgp_nodal_data!(lev_arrays, atmos, prog, aux, min_q_vap)
            lev_set!(nd, ev, h, lev_arrays...)
        end

        # Average the level data to obtain the layer data.
        for v in 1:nvertelem - 1
            lev_to_lay!(FT, v, h, lev_arrays, lay_arrays, layer_interp)
        end
    end
end

@kernel function  kernel_load_into_rrtmgp!(
    atmos::BalanceLaw,
    ::Val{Nq1}, ::Val{Nq2}, ::Val{Nq3}, ::Val{nvertelem},
    state_prognostic,
    state_auxiliary,
    elems,
    lev_arrays,
    lay_arrays,
    layer_interp,
) where {Nq1, Nq2, Nq3, nvertelem}
    @uniform begin
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(atmos, Prognostic())
        num_state_auxiliary = number_states(atmos, Auxiliary())
        local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        prog = Vars{vars_state(atmos, Prognostic(), FT)}(
            local_state_prognostic,
        )
        aux = Vars{vars_state(atmos, Auxiliary(), FT)}(
            local_state_auxiliary,
        )
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)
    min_q_vap = @private FT (1)

    @inbounds begin
        min_q_vap[1] = FT(Inf)
        
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
            nd = rrtmgp_nodal_data!(lev_arrays, atmos, prog, aux, min_q_vap)
            lev_set!(nd, k, h, lev_arrays...)
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
            nd = rrtmgp_nodal_data!(lev_arrays, atmos, prog, aux, min_q_vap)
            lev_avg!(FT, nd, (Nq3 - 1) * (ev - 1) + 1, h, lev_arrays...)

            # Fill in data from the remaining points.
            @unroll for k in 2:Nq3
                ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                @unroll for s in 1:num_state_prognostic
                    local_state_prognostic[s] = state_prognostic[ijk, s, e]
                end
                @unroll for s in 1:num_state_auxiliary
                    local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
                end
                nd =
                    rrtmgp_nodal_data!(lev_arrays, atmos, prog, aux, min_q_vap)
                lev_set!(nd, (Nq3 - 1) * (ev - 1) + k, h, lev_arrays...)
            end
        end

        # Average the level data to obtain the layer data.
        for v in 1:(Nq3 - 1) * nvertelem
            lev_to_lay!(FT, v, h, lev_arrays, lay_arrays, layer_interp)
        end
    end
end

@kernel function kernel_unload_from_rrtmgp!(
    ::Val{Nq1}, ::Val{Nq2}, ::Val{1}, ::Val{nvertelem}, ::Val{fluxindex},
    state_auxiliary,
    flux_lw,
    flux_sw,
    elems,
) where {Nq1, Nq2, nvertelem, fluxindex}
    ev, _eh, _ = @index(Group, NTuple)
    _, i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]
        h = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i
        e = nvertelem * (eh - 1) + ev
        ijk = Nq1 * (j - 1) + i

        if flux_sw isa FluxSWNoScat
            flux_net_sw = -flux_sw.flux_dn_dir[ev, h]
        else
            flux_net_sw = flux_sw.flux_up[ev, h] - flux_sw.flux_dn[ev, h]
        end

        state_auxiliary[ijk, fluxindex, e] =
            flux_lw.flux_net[ev, h] + flux_net_sw
    end
end

@kernel function kernel_unload_from_rrtmgp!(
    ::Val{Nq1}, ::Val{Nq2}, ::Val{Nq3}, ::Val{nvertelem}, ::Val{fluxindex},
    state_auxiliary,
    flux_lw,
    flux_sw,
    elems,
) where {Nq1, Nq2, Nq3, nvertelem, fluxindex}
    ev, _eh, _ = @index(Group, NTuple)
    _, i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]
        h = Nq1 * (Nq2 * (eh - 1) + (j - 1)) + i
        e = nvertelem * (eh - 1) + ev
        @unroll for k in 1:Nq3
            ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
            v = (Nq3 - 1) * (ev - 1) + k

            if flux_sw isa FluxSWNoScat
                flux_net_sw = -flux_sw.flux_dn_dir[v, h]
            else
                flux_net_sw = flux_sw.flux_up[v, h] - flux_sw.flux_dn[v, h]
            end

            state_auxiliary[ijk, fluxindex, e] =
                flux_lw.flux_net[v, h] + flux_net_sw
        end
    end
end

@kernel function kernel_flux_source!(
    balance_law::BalanceLaw,
    ::Val{Nq1}, ::Val{Nq2}, ::Val{1}, ::Val{nvertelem}, ::Val{iF},
    ::Val{i∂F∂zDG}, ::Val{i∂F∂z2}, ::Val{i∂F∂z4}, ::Val{i∂F∂z6},
    state_auxiliary,
    elems,
    grid,
) where {Nq1, Nq2, nvertelem, iF, i∂F∂zDG, i∂F∂z2, i∂F∂z4, i∂F∂z6}
    error("Source term radiation not implemented for FV")
end

@kernel function kernel_flux_source!(
    balance_law::BalanceLaw,
    ::Val{Nq1}, ::Val{Nq2}, ::Val{Nq3}, ::Val{nvertelem}, ::Val{iF},
    ::Val{i∂F∂zDG}, ::Val{i∂F∂z2}, ::Val{i∂F∂z4}, ::Val{i∂F∂z6},
    state_auxiliary,
    elems,
    grid,
) where {Nq1, Nq2, Nq3, nvertelem, iF, i∂F∂zDG, i∂F∂z2, i∂F∂z4, i∂F∂z6}
    @uniform begin
        FT = eltype(state_auxiliary)
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

        z = MArray{Tuple{7}, FT}(undef)
        F = MArray{Tuple{7}, FT}(undef)
    end

    _eh = @index(Group, Linear)
    i, j = @index(Local, NTuple)

    @inbounds begin
        eh = elems[_eh]

        for ev in 1:nvertelem
            e = nvertelem * (eh - 1) + ev

            # Emulation of DG derivative calculation
            @unroll for k in 1:Nq3
                ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))
                state_auxiliary[ijk, i∂F∂zDG, e] = zero(FT)
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
                F1, F2, F3 = state_auxiliary[ijk, iF, e] * zhat
                Fv = M * (ζx1 * F1 + ζx2 * F2 + ζx3 * F3)
                @unroll for n in 1:Nq3
                    ijn = Nq1 * (Nq2 * (n - 1) + (j - 1)) + i
                    MI = vgeo[ijn, Grids._MI, e]
                    state_auxiliary[ijn, i∂F∂zDG, e] += MI * D[k, n] * Fv
                end
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
                flux_norm = state_auxiliary[vid, iF, e]
                @unroll for s in 1:num_state_auxiliary
                    local_state_auxiliary[s] = state_auxiliary[vid, s, e]
                end
                flux = flux_norm * vertical_unit_vector(balance_law, aux)
                @unroll for s in 1:num_state_auxiliary
                    local_state_auxiliary[s] = state_auxiliary[vid⁺, s, e⁺]
                end
                flux⁺ = flux_norm * vertical_unit_vector(balance_law, aux)
                state_auxiliary[vid, i∂F∂zDG, e] -= vMI * sM *
                    ((flux + flux⁺)' * (normal_vector / FT(2)))
            end

            # Second-order accurate derivative calculation
            @unroll for k in 1:Nq3
                is_bottom = ev == 1 && k == 1
                is_top = ev == nvertelem && k == Nq3

                if is_bottom
                    e′ = 1
                    k′ = 1
                    index_range = 3:-1:1
                elseif is_top
                    e′ = nvertelem
                    k′ = Nq3
                    for Δk in 1:2
                        k′ -= 1
                        if k′ == 0
                            e′ -= 1
                            k′ = Nq3 - 1
                        end
                    end
                    index_range = 1:3
                else
                    e′ = e
                    k′ = k - 1
                    if k′ == 0
                        e′ -= 1
                        k′ = Nq3 - 1
                    end
                    index_range = 1:3
                end

                for index in index_range
                    ijk′ = Nq1 * (Nq2 * (k′ - 1) + (j - 1)) + i
                    z[index] = vgeo[ijk′, Grids._x3, e′]
                    F[index] = state_auxiliary[ijk′, iF, e′]
                    k′ += 1
                    if k′ == Nq3 + 1
                        e′ += 1
                        k′ = 2
                    end
                end

                Δz₁₂ = z[2] - z[1]
                Δz₁₃ = z[3] - z[1]
                Δz₂₃ = z[3] - z[2]

                if is_bottom || is_top
                    ∂F∂z =
                        F[1] * Δz₂₃/(Δz₁₂*Δz₁₃) +
                        F[2] * -Δz₁₃/(Δz₁₂*Δz₂₃) +
                        F[3] * (1/Δz₁₃ + 1/Δz₂₃)
                else
                    ∂F∂z =
                        F[1] * -Δz₂₃/(Δz₁₂*Δz₁₃) +
                        F[2] * (1/Δz₁₂ - 1/Δz₂₃) +
                        F[3] * Δz₁₂/(Δz₁₃*Δz₂₃)
                end

                ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                state_auxiliary[ijk, i∂F∂z2, e] = -∂F∂z
            end

            # Fourth-order accurate derivative calculation
            @unroll for k in 1:Nq3
                is_bottom1 = ev == 1 && k == 1
                is_bottom2 = ev == 1 && k == 2 ||
                    Nq3 == 2 && ev == 2 && k == 1
                is_top1 = ev == nvertelem && k == Nq3
                is_top2 = ev == nvertelem && k == Nq3 - 1 ||
                    Nq3 == 2 && ev == nvertelem - 1 && k == 2
                
                if is_bottom1 || is_bottom2
                    e′ = 1
                    k′ = 1
                    index_range = 5:-1:1
                elseif is_top1 || is_top2
                    e′ = nvertelem
                    k′ = Nq3
                    for Δk in 1:4
                        k′ -= 1
                        if k′ == 0
                            e′ -= 1
                            k′ = Nq3 - 1
                        end
                    end
                    index_range = 1:5
                else
                    e′ = e
                    k′ = k
                    for Δk in 1:2
                        k′ -= 1
                        if k′ == 0
                            e′ -= 1
                            k′ = Nq3 - 1
                        end
                    end
                    index_range = 1:5
                end

                for index in index_range
                    ijk′ = Nq1 * (Nq2 * (k′ - 1) + (j - 1)) + i
                    z[index] = vgeo[ijk′, Grids._x3, e′]
                    F[index] = state_auxiliary[ijk′, iF, e′]
                    k′ += 1
                    if k′ == Nq3 + 1
                        e′ += 1
                        k′ = 2
                    end
                end

                Δz₁₂ = z[2] - z[1]
                Δz₁₃ = z[3] - z[1]
                Δz₁₄ = z[4] - z[1]
                Δz₁₅ = z[5] - z[1]
                Δz₂₃ = z[3] - z[2]
                Δz₂₄ = z[4] - z[2]
                Δz₂₅ = z[5] - z[2]
                Δz₃₄ = z[4] - z[3]
                Δz₃₅ = z[5] - z[3]
                Δz₄₅ = z[5] - z[4]

                if is_bottom1 || is_top1
                    ∂F∂z =
                        F[1] * (Δz₂₅*Δz₃₅*Δz₄₅)/(Δz₁₂*Δz₁₃*Δz₁₄*Δz₁₅) +
                        F[2] * -(Δz₁₅*Δz₃₅*Δz₄₅)/(Δz₁₂*Δz₂₃*Δz₂₄*Δz₂₅) +
                        F[3] * (Δz₁₅*Δz₂₅*Δz₄₅)/(Δz₁₃*Δz₂₃*Δz₃₄*Δz₃₅) +
                        F[4] * -(Δz₁₅*Δz₂₅*Δz₃₅)/(Δz₁₄*Δz₂₄*Δz₃₄*Δz₄₅) +
                        F[5] * (1/Δz₁₅ + 1/Δz₂₅ + 1/Δz₃₅ + 1/Δz₄₅)
                elseif is_bottom2 || is_top2
                    ∂F∂z =
                        F[1] * -(Δz₂₄*Δz₃₄*Δz₄₅)/(Δz₁₂*Δz₁₃*Δz₁₄*Δz₁₅) +
                        F[2] * (Δz₁₄*Δz₃₄*Δz₄₅)/(Δz₁₂*Δz₂₃*Δz₂₄*Δz₂₅) +
                        F[3] * -(Δz₁₄*Δz₂₄*Δz₄₅)/(Δz₁₃*Δz₂₃*Δz₃₄*Δz₃₅) +
                        F[4] * (1/Δz₁₄ + 1/Δz₂₄ + 1/Δz₃₄ - 1/Δz₄₅) +
                        F[5] * (Δz₁₄*Δz₂₄*Δz₃₄)/(Δz₁₅*Δz₂₅*Δz₃₅*Δz₄₅)
                else
                    ∂F∂z =
                        F[1] * (Δz₂₃*Δz₃₄*Δz₃₅)/(Δz₁₂*Δz₁₃*Δz₁₄*Δz₁₅) +
                        F[2] * -(Δz₁₃*Δz₃₄*Δz₃₅)/(Δz₁₂*Δz₂₃*Δz₂₄*Δz₂₅) +
                        F[3] * (1/Δz₁₃ + 1/Δz₂₃ - 1/Δz₃₄ - 1/Δz₃₅) +
                        F[4] * (Δz₁₃*Δz₂₃*Δz₃₅)/(Δz₁₄*Δz₂₄*Δz₃₄*Δz₄₅) +
                        F[5] * -(Δz₁₃*Δz₂₃*Δz₃₄)/(Δz₁₅*Δz₂₅*Δz₃₅*Δz₄₅)
                end

                ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                state_auxiliary[ijk, i∂F∂z4, e] = -∂F∂z
            end

            # Sixth-order accurate derivative calculation
            @unroll for k in 1:Nq3
                is_bottom1 = ev == 1 && k == 1
                is_bottom2 = ev == 1 && k == 2 ||
                    Nq3 == 2 && ev == 2 && k == 1
                is_bottom3 = ev == 1 && k == 3 ||
                    Nq3 == 2 && ev == 2 && k == 2 ||
                    Nq3 == 2 && ev == 3 && k == 1 ||
                    Nq3 == 3 && ev == 2 && k == 1
                is_top1 = ev == nvertelem && k == Nq3
                is_top2 = ev == nvertelem && k == Nq3 - 1 ||
                    Nq3 == 2 && ev == nvertelem - 1 && k == 2
                is_top3 = ev == nvertelem && k == Nq3 - 2 ||
                    Nq3 == 2 && ev == nvertelem - 1 && k == 1 ||
                    Nq3 == 2 && ev == nvertelem - 2 && k == 2 ||
                    Nq3 == 3 && ev == nvertelem - 1 && k == 3

                if is_bottom1 || is_bottom2 || is_bottom3
                    e′ = 1
                    k′ = 1
                    index_range = 7:-1:1
                elseif is_top1 || is_top2 || is_top3
                    e′ = nvertelem
                    k′ = Nq3
                    for Δk in 1:6
                        k′ -= 1
                        if k′ == 0
                            e′ -= 1
                            k′ = Nq3 - 1
                        end
                    end
                    index_range = 1:7
                else
                    e′ = e
                    k′ = k
                    for Δk in 1:3
                        k′ -= 1
                        if k′ == 0
                            e′ -= 1
                            k′ = Nq3 - 1
                        end
                    end
                    index_range = 1:7
                end

                for index in index_range
                    ijk′ = Nq1 * (Nq2 * (k′ - 1) + (j - 1)) + i
                    z[index] = vgeo[ijk′, Grids._x3, e′]
                    F[index] = state_auxiliary[ijk′, iF, e′]
                    k′ += 1
                    if k′ == Nq3 + 1
                        e′ += 1
                        k′ = 2
                    end
                end

                Δz₁₂ = z[2] - z[1]
                Δz₁₃ = z[3] - z[1]
                Δz₁₄ = z[4] - z[1]
                Δz₁₅ = z[5] - z[1]
                Δz₁₆ = z[6] - z[1]
                Δz₁₇ = z[7] - z[1]
                Δz₂₃ = z[3] - z[2]
                Δz₂₄ = z[4] - z[2]
                Δz₂₅ = z[5] - z[2]
                Δz₂₆ = z[6] - z[2]
                Δz₂₇ = z[7] - z[2]
                Δz₃₄ = z[4] - z[3]
                Δz₃₅ = z[5] - z[3]
                Δz₃₆ = z[6] - z[3]
                Δz₃₇ = z[7] - z[3]
                Δz₄₅ = z[5] - z[4]
                Δz₄₆ = z[6] - z[4]
                Δz₄₇ = z[7] - z[4]
                Δz₅₆ = z[6] - z[5]
                Δz₅₇ = z[7] - z[5]
                Δz₆₇ = z[7] - z[6]

                if is_bottom1 || is_top1
                    ∂F∂z =
                        F[1] * (Δz₂₇*Δz₃₇*Δz₄₇*Δz₅₇*Δz₆₇)/(Δz₁₂*Δz₁₃*Δz₁₄*Δz₁₅*Δz₁₆*Δz₁₇) +
                        F[2] * -(Δz₁₇*Δz₃₇*Δz₄₇*Δz₅₇*Δz₆₇)/(Δz₁₂*Δz₂₃*Δz₂₄*Δz₂₅*Δz₂₆*Δz₂₇) +
                        F[3] * (Δz₁₇*Δz₂₇*Δz₄₇*Δz₅₇*Δz₆₇)/(Δz₁₃*Δz₂₃*Δz₃₄*Δz₃₅*Δz₃₆*Δz₃₇) +
                        F[4] * -(Δz₁₇*Δz₂₇*Δz₃₇*Δz₅₇*Δz₆₇)/(Δz₁₄*Δz₂₄*Δz₃₄*Δz₄₅*Δz₄₆*Δz₄₇) +
                        F[5] * (Δz₁₇*Δz₂₇*Δz₃₇*Δz₄₇*Δz₆₇)/(Δz₁₅*Δz₂₅*Δz₃₅*Δz₄₅*Δz₅₆*Δz₅₇) +
                        F[6] * -(Δz₁₇*Δz₂₇*Δz₃₇*Δz₄₇*Δz₅₇)/(Δz₁₆*Δz₂₆*Δz₃₆*Δz₄₆*Δz₅₆*Δz₆₇) +
                        F[7] * (1/Δz₁₇ + 1/Δz₂₇ + 1/Δz₃₇ + 1/Δz₄₇ + 1/Δz₅₇ + 1/Δz₆₇)
                elseif is_bottom2 || is_top2
                    ∂F∂z =
                        F[1] * -(Δz₂₆*Δz₃₆*Δz₄₆*Δz₅₆*Δz₆₇)/(Δz₁₂*Δz₁₃*Δz₁₄*Δz₁₅*Δz₁₆*Δz₁₇) +
                        F[2] * (Δz₁₆*Δz₃₆*Δz₄₆*Δz₅₆*Δz₆₇)/(Δz₁₂*Δz₂₃*Δz₂₄*Δz₂₅*Δz₂₆*Δz₂₇) +
                        F[3] * -(Δz₁₆*Δz₂₆*Δz₄₆*Δz₅₆*Δz₆₇)/(Δz₁₃*Δz₂₃*Δz₃₄*Δz₃₅*Δz₃₆*Δz₃₇) +
                        F[4] * (Δz₁₆*Δz₂₆*Δz₃₆*Δz₅₆*Δz₆₇)/(Δz₁₄*Δz₂₄*Δz₃₄*Δz₄₅*Δz₄₆*Δz₄₇) +
                        F[5] * -(Δz₁₆*Δz₂₆*Δz₃₆*Δz₄₆*Δz₆₇)/(Δz₁₅*Δz₂₅*Δz₃₅*Δz₄₅*Δz₅₆*Δz₅₇) +
                        F[6] * (1/Δz₁₆ + 1/Δz₂₆ + 1/Δz₃₆ + 1/Δz₄₆ + 1/Δz₅₆ - 1/Δz₆₇) +
                        F[7] * (Δz₁₆*Δz₂₆*Δz₃₆*Δz₄₆*Δz₅₆)/(Δz₁₇*Δz₂₇*Δz₃₇*Δz₄₇*Δz₅₇*Δz₆₇)
                elseif is_bottom3 || is_top3
                    ∂F∂z =
                        F[1] * (Δz₂₅*Δz₃₅*Δz₄₅*Δz₅₆*Δz₅₇)/(Δz₁₂*Δz₁₃*Δz₁₄*Δz₁₅*Δz₁₆*Δz₁₇) +
                        F[2] * -(Δz₁₅*Δz₃₅*Δz₄₅*Δz₅₆*Δz₅₇)/(Δz₁₂*Δz₂₃*Δz₂₄*Δz₂₅*Δz₂₆*Δz₂₇) +
                        F[3] * (Δz₁₅*Δz₂₅*Δz₄₅*Δz₅₆*Δz₅₇)/(Δz₁₃*Δz₂₃*Δz₃₄*Δz₃₅*Δz₃₆*Δz₃₇) +
                        F[4] * -(Δz₁₅*Δz₂₅*Δz₃₅*Δz₅₆*Δz₅₇)/(Δz₁₄*Δz₂₄*Δz₃₄*Δz₄₅*Δz₄₆*Δz₄₇) +
                        F[5] * (1/Δz₁₅ + 1/Δz₂₅ + 1/Δz₃₅ + 1/Δz₄₅ - 1/Δz₅₆ - 1/Δz₅₇) +
                        F[6] * (Δz₁₅*Δz₂₅*Δz₃₅*Δz₄₅*Δz₅₇)/(Δz₁₆*Δz₂₆*Δz₃₆*Δz₄₆*Δz₅₆*Δz₆₇) +
                        F[7] * -(Δz₁₅*Δz₂₅*Δz₃₅*Δz₄₅*Δz₅₆)/(Δz₁₇*Δz₂₇*Δz₃₇*Δz₄₇*Δz₅₇*Δz₆₇)
                else
                    ∂F∂z =
                        F[1] * -(Δz₂₄*Δz₃₄*Δz₄₅*Δz₄₆*Δz₄₇)/(Δz₁₂*Δz₁₃*Δz₁₄*Δz₁₅*Δz₁₆*Δz₁₇) +
                        F[2] * (Δz₁₄*Δz₃₄*Δz₄₅*Δz₄₆*Δz₄₇)/(Δz₁₂*Δz₂₃*Δz₂₄*Δz₂₅*Δz₂₆*Δz₂₇) +
                        F[3] * -(Δz₁₄*Δz₂₄*Δz₄₅*Δz₄₆*Δz₄₇)/(Δz₁₃*Δz₂₃*Δz₃₄*Δz₃₅*Δz₃₆*Δz₃₇) +
                        F[4] * (1/Δz₁₄ + 1/Δz₂₄ + 1/Δz₃₄ - 1/Δz₄₅ - 1/Δz₄₆ - 1/Δz₄₇) +
                        F[5] * (Δz₁₄*Δz₂₄*Δz₃₄*Δz₄₆*Δz₄₇)/(Δz₁₅*Δz₂₅*Δz₃₅*Δz₄₅*Δz₅₆*Δz₅₇) +
                        F[6] * -(Δz₁₄*Δz₂₄*Δz₃₄*Δz₄₅*Δz₄₇)/(Δz₁₆*Δz₂₆*Δz₃₆*Δz₄₆*Δz₅₆*Δz₆₇) +
                        F[7] * (Δz₁₄*Δz₂₄*Δz₃₄*Δz₄₅*Δz₄₆)/(Δz₁₇*Δz₂₇*Δz₃₇*Δz₄₇*Δz₅₇*Δz₆₇)
                end
                ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                state_auxiliary[ijk, i∂F∂z6, e] = -∂F∂z
            end
        end
    end
end

###################### Initializing the Prognostic State ######################

# Add a thermodynamic state initializer that is absent.
function PhaseNonEquil_pTq(
    param_set,
    p::FT,
    T::FT,
    q_pt::PhasePartition{FT},
) where {FT <: Real}
    ρ = air_density(param_set, T, p, q_pt)
    e_int = internal_energy(param_set, T, q_pt)
    return PhaseNonEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_pt)
end

# Create an object for initializing either RRTMGP or the prognostic state to
# interpolated file data, or to a fixed temperature and volume mixing ratio of
# water.
struct SimpleInitializer{PT, TT, VT}
    p_prof::PT
    T_interp::TT
    vmr_h2o_interp::VT
end
function SimpleInitializer(temp_prof::TemperatureProfile{FT}) where {FT}
    p_prof = z -> temp_prof(param_set, z)[2]

    ds_input = Dataset(ds_input_path)

    ps = Array{FT}(ds_input["pres_level"])[:, site_index]
    Ts = Array{FT}(ds_input["temp_level"])[:, site_index, expt_index]
    T_interp = Spline1D(ps, Ts; k = 1, bc = "extrapolate", s = FT(0))
    
    ps = Array{FT}(ds_input["pres_layer"])[:, site_index]
    h2os = Array{FT}(ds_input["water_vapor"])[:, site_index, expt_index]
    vmr_h2o_interp = Spline1D(ps, h2os; k = 1, bc = "extrapolate", s = FT(0))

    close(ds_input)

    return SimpleInitializer(p_prof, T_interp, vmr_h2o_interp)
end
function SimpleInitializer(
    temp_prof::TemperatureProfile{FT},
    vmr_h2o::FT,
) where {FT}
    p_prof = z -> temp_prof(param_set, z)[2]

    ds_input = Dataset(ds_input_path)

    ps = Array{FT}(ds_input["pres_level"])[:, site_index]
    Ts = Array{FT}(ds_input["temp_level"])[:, site_index, expt_index]
    T_interp = Spline1D(ps, Ts; k = 1, bc = "extrapolate", s = FT(0))

    vmr_h2o_interp = _ -> FT(vmr_h2o)

    close(ds_input)

    return SimpleInitializer(p_prof, T_interp, vmr_h2o_interp)
end
function SimpleInitializer(
    temp_prof::TemperatureProfile{FT},
    T::FT,
    vmr_h2o::FT,
) where {FT}
    p_prof = z -> temp_prof(param_set, z)[2]
    T_interp = _ -> FT(T)
    vmr_h2o_interp = _ -> FT(vmr_h2o)
    return SimpleInitializer(p_prof, T_interp, vmr_h2o_interp)
end
function SimpleInitializer(
    temp_prof::TemperatureProfile{FT},
    ref_filename::AbstractString,
) where {FT}
    p_prof = z -> temp_prof(param_set, z)[2]
    _, ref_dg = deserialize(joinpath(@__DIR__, ref_filename))

    ps = reverse(horz_mean(ref_dg.modeldata.rrtmgp_data.solver.as.p_lev))
    Ts = reverse(horz_mean(ref_dg.modeldata.rrtmgp_data.solver.as.t_lev))
    T_interp = Spline1D(ps, Ts; k = 1, bc = "extrapolate", s = FT(0))

    ps = reverse(horz_mean(ref_dg.modeldata.rrtmgp_data.solver.as.p_lay))
    h2os =
        reverse(horz_mean(ref_dg.modeldata.rrtmgp_data.solver.as.vmr.vmr_h2o))
    vmr_h2o_interp = Spline1D(ps, h2os; k = 1, bc = "extrapolate", s = FT(0))

    return SimpleInitializer(p_prof, T_interp, vmr_h2o_interp)
end
function (si::SimpleInitializer)(z)
    p = si.p_prof(z)
    return (;
        p = p,
        T = si.T_interp(p),
        vmr_h2o = si.vmr_h2o_interp(p),
    )
end
function (si::SimpleInitializer)(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)
    ρ = aux.ref_state.ρ
    z = localgeo.coord[3]
    p, T, vmr_h2o = si(z)
    param_set = parameter_set(bl)
    e_kin = FT(0)
    e_pot = FT(grav(param_set)) * z
    q_tot = vmr_h2o / (FT(molmass_ratio(param_set)) + vmr_h2o)

    @assert p ≈ aux.ref_state.p

    state.ρ = ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    if (
        bl.moisture isa DryModel ||
        bl.moisture isa CappedConstantSpecificHumidity ||
        bl.moisture isa CappedMonotonicRelativeHumidity
    )
        thermo_state = PhaseDry_pT(param_set, p, T)
        if bl.moisture isa CappedConstantSpecificHumidity
            state.moisture.q_vap = q_tot
        end
    else
        if bl.moisture isa EquilMoist
            thermo_state = PhaseEquil_pTq(param_set, p, T, q_tot)
            state.moisture.ρq_tot = ρ * q_tot
        else # NonEquilMoist
            thermo_state =
                PhaseNonEquil_pTq(param_set, p, T, PhasePartition(q_tot))
            state.moisture.ρq_tot = ρ * q_tot
            state.moisture.ρq_liq = FT(0)
            state.moisture.ρq_ice = FT(0)
        end
    end
    state.energy.ρe = ρ * total_energy(e_kin, e_pot, thermo_state)
end

# Create a function for initializing the prognostic state to the reference
# state.
function init_to_ref_state!(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)
    ρ = aux.ref_state.ρ
    ρq_tot = aux.ref_state.ρq_tot
    ρq_liq = aux.ref_state.ρq_liq
    ρq_ice = aux.ref_state.ρq_ice

    state.ρ = ρ
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.energy.ρe = aux.ref_state.ρe
    if bl.moisture isa CappedConstantSpecificHumidity
        state.moisture.q_vap = (ρq_tot - ρq_liq - ρq_ice) / ρ
    elseif bl.moisture isa EquilMoist
        state.moisture.ρq_tot = ρq_tot
    elseif bl.moisture isa NonEquilMoist
        state.moisture.ρq_tot = ρq_tot
        state.moisture.ρq_liq = ρq_liq
        state.moisture.ρq_ice = ρq_ice
    end
end

########################### Custom Energy Filtering ###########################

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

# Generate dictionaries with the means and variances of all prognositc,
# auxiliary, and gradient flux variables. Interpolate at element interfaces
# if needed.
function means_and_vars(Q, dg, interp)
    FT = eltype(Q)
    mean_dicts = []
    var_dicts = []
    for (f, array) in (
        (get_horizontal_mean, mean_dicts),
        (get_horizontal_variance, var_dicts),
    )
        for (state, state_type) in (
            (Q, Prognostic()),
            (dg.state_auxiliary, Auxiliary()),
            (dg.state_gradient_flux, GradientFlux()),
        )
            vars = vars_state(dg.balance_law, state_type, FT)
            push!(array, f(dg.grid, state, vars; interp = interp))
        end
    end
    return merge(mean_dicts...), merge(var_dicts...)
end

# Average an array across its second dimension.
horz_mean(array) = dropdims(sum(array; dims = 2); dims = 2) ./ size(array, 2)

# Dictionaries used to determine what is plotted.
function debug_info(dg)
    xinfo = (
        # ("ρ", ("density", " [kg/m^3]", "node", false)),
        ("ρe", ("energy density", " [J/m^3]", "node", false)),
        ("∇htotv", (
            "specific enthalpy gradient's vertical component",
            " [J/kg/m]",
            "node",
            true,
        )),
        # ("∇htotv_var", (
        #     "variance of specific enthalpy gradient's vertical component",
        #     "",
        #     "node",
        #     true,
        # )),
        ("p", ("pressure", " [Pa]", "lev", false)),
        ("p_lay", ("layer pressure", " [Pa]", "lay", false)),
        ("T", ("temperature", " [K]", "lev", false)),
        ("T_lay", ("layer temperature", " [K]", "lay", false)),
        ("F_net", ("net radiation energy flux", " [W/m^2]", "lev", false)),
        ("F_up_lw", (
            "upward longwave radiation energy flux",
            " [W/m^2]",
            "lev",
            false,
        )),
        ("F_dn_lw", (
            "downward longwave radiation energy flux",
            " [W/m^2]",
            "lev",
            false,
        )),
        ("F_up_sw", (
            "upward shortwave radiation energy flux",
            " [W/m^2]",
            "lev",
            false,
        )),
        ("F_dn_sw", (
            "downward shortwave radiation energy flux",
            " [W/m^2]",
            "lev",
            false,
        )),
    )
    if radiation_model(dg.balance_law).tendency_type isa Source
        for (name, label) in (
            # ("dg", "DG"),
            # ("deriv2", "2ⁿᵈ-order"),
            # ("deriv4", "4ᵗʰ-order"),
            ("deriv6", "6ᵗʰ-order"),
        )
            xinfo = (
                xinfo...,
                # ("src_$name", (
                #     "$label source due to radiation",
                #     " [W/m^3]",
                #     "node",
                #     false,
                # )),
                ("heat_rate_$name", (
                    "$label heating rate due to radiation",
                    " [K/day]",
                    "node",
                    false,
                )),
            )
        end
    end
    if dg.modeldata.rrtmgp_data.solver.as isa AtmosphericState
        xinfo = (
            xinfo...,
            ("vmr_h2o", ("volume mixing ratio of water", "", "lev", false)),
            # ("q_vap", ("specific humidity", "", "lev", false)),
            ("rel_hum", ("relative humidity", "", "node", false)),
            # ("vmr_h2o_lay", (
            #     "layer volume mixing ratio of water",
            #     "",
            #     "lay",
            #     false,
            # )),
            ("vmr_o3_lay", (
                "layer volume mixing ratio of ozone",
                "",
                "lay",
                false,
            )),
        )
    end

    yinfo = (
        ("alt", ("altitude", " [km]", false)),
        # ("pres", ("pressure", " [kPa]", true)),
    )

    return Dict(xinfo), Dict(yinfo)
end

# Values that correspond to debug_info(dg).
function debug_values(Q, dg, interp)
    FT = eltype(Q)
    rrtmgp_data = dg.modeldata.rrtmgp_data
    param_set = parameter_set(dg.balance_law)
    mean_dict, var_dict = means_and_vars(Q, dg, interp)

    if interp
        interp_mean_dict = mean_dict
    else
        interp_mean_dict = means_and_vars(Q, dg, true)[1]
    end
    nnode = length(interp_mean_dict["ρ"])

    if rrtmgp_data.solver.flux_sw isa FluxSWNoScat
        F_net = horz_mean(
            rrtmgp_data.solver.flux_lw.flux_up .-
            rrtmgp_data.solver.flux_lw.flux_dn .-
            rrtmgp_data.solver.flux_sw.flux_dn_dir
        )
    else
        F_net = horz_mean(
            rrtmgp_data.solver.flux_lw.flux_up .-
            rrtmgp_data.solver.flux_lw.flux_dn .+
            rrtmgp_data.solver.flux_sw.flux_up .-
            rrtmgp_data.solver.flux_sw.flux_dn
        )
    end
    @assert F_net[1:nnode] ≈ interp_mean_dict["radiation.flux"]

    xvalues = (
        ("ρ", mean_dict["ρ"]),
        ("ρe", mean_dict["energy.ρe"]),
        ("∇htotv", mean_dict["energy.∇h_tot[3]"]),
        ("∇htotv_var", var_dict["energy.∇h_tot[3]"]),
        ("p", horz_mean(rrtmgp_data.solver.as.p_lev)),
        ("p_lay", horz_mean(rrtmgp_data.solver.as.p_lay)),
        ("T", horz_mean(rrtmgp_data.solver.as.t_lev)),
        ("T_lay", horz_mean(rrtmgp_data.solver.as.t_lay)),
        ("F_net", F_net),
        ("F_up_lw", horz_mean(rrtmgp_data.solver.flux_lw.flux_up)),
        ("F_dn_lw", -horz_mean(rrtmgp_data.solver.flux_lw.flux_dn)),
    )

    if rrtmgp_data.solver.flux_sw isa FluxSWNoScat
        xvalues = (
            xvalues...,
            ("F_up_sw", zeros(FT, rrtmgp_data.solver.as.nlay + 1)),
            ("F_dn_sw", -horz_mean(rrtmgp_data.solver.flux_sw.flux_dn_dir)),
        )
    else
        xvalues = (
            xvalues...,
            ("F_up_sw", horz_mean(rrtmgp_data.solver.flux_sw.flux_up)),
            ("F_dn_sw", -horz_mean(rrtmgp_data.solver.flux_sw.flux_dn)),
        )
    end
    
    if radiation_model(dg.balance_law).tendency_type isa Source
        for name in ("dg", "deriv2", "deriv4", "deriv6")
            src = mean_dict["radiation.src_$name"]
            # TODO: This is inaccurate when there is horizontal variance.
            heat_rate = FT(60 * 60 * 24) .* src ./  mean_dict["ρ"] ./
                CLIMAParameters.Planet.cv_d(param_set)
            xvalues = (
                xvalues...,
                ("src_$name", src),
                ("heat_rate_$name", heat_rate),
            )
        end
    end
    if rrtmgp_data.solver.as isa AtmosphericState
        vmr_h2o = rrtmgp_data.vmr_h2o_lev
        q_vap = vmr_h2o ./ (FT(molmass_ratio(param_set)) .+ vmr_h2o)
        if dg.balance_law.moisture isa EquilMoist
            phase_type = PhaseEquil
        elseif dg.balance_law.moisture isa NonEquilMoist
            phase_type = PhaseNonEquil
        else
            phase_type = PhaseDry
        end
        # TODO: This is inaccurate when there is horizontal variance.
        q_vap_sat = map(
            (T, ρ, q_vap) -> q_vap_saturation(
                param_set,
                T,
                ρ,
                phase_type,
                PhasePartition(q_vap),
            ),
            rrtmgp_data.solver.as.t_lev[1:nnode, :],
            repeat(interp_mean_dict["ρ"], 1, rrtmgp_data.solver.as.ncol),
            q_vap[1:nnode, :],
        )
        xvalues = (
            xvalues...,
            ("vmr_h2o", horz_mean(vmr_h2o)),
            ("q_vap", horz_mean(q_vap)),
            ("rel_hum", horz_mean(q_vap[1:nnode, :] ./ q_vap_sat)),
            ("vmr_h2o_lay", horz_mean(rrtmgp_data.solver.as.vmr.vmr_h2o)),
            ("vmr_o3_lay", horz_mean(rrtmgp_data.solver.as.vmr.vmr_o3)),
        )
    end
    
    yvalues = (
        ("alt_node", mean_dict["coord[3]"] ./ FT(1000)),
        ("alt_lev", rrtmgp_data.z_lev ./ FT(1000)),
        ("alt_lay", rrtmgp_data.z_lay ./ FT(1000)),
        ("pres_node", mean_dict["ref_state.p"] ./ FT(1000)),
        ("pres_lev", horz_mean(rrtmgp_data.solver.as.p_lev) ./ FT(1000)),
        ("pres_lay", horz_mean(rrtmgp_data.solver.as.p_lay) ./ FT(1000)),
    )

    return Dict(xvalues), Dict(yvalues)
end

join_palettes(symbs...) =
    PlotUtils.ColorSchemes.ColorScheme(vcat(
        map(symb -> PlotUtils.get_colorscheme(symb).colors, symbs)...,
    ))

# TODO: Merge these two functions.
function plot_multiple(
    filename,
    xs,
    xlabel,
    ys,
    ylabel,
    yflip,
    labels;
    skip_first = false,
    highlight_first = false,
    highlight_last = false,
)
    plot(;
        legend = :outertopright,
        palette = join_palettes(:seaborn_bright, :seaborn_dark),
        xlabel = xlabel,
        ylabel = ylabel,
        yflip = yflip,
    )
    for (i, label) in enumerate(labels)
        i == 1 && skip_first && continue
        plot!(
            xs[i],
            ys[i] isa AbstractArray ? ys[i] : ys;
            seriescolor = (
                i == 1 && highlight_first ||
                i == length(labels) && highlight_last
            ) ? :black : (highlight_first ? i - 1 : i),
            label = label,
        )
    end
    savefig(joinpath(@__DIR__, "$filename.png"))
end
function plot_multiple_views(
    xlims_list,
    xticks_list,
    filename,
    xs,
    xlabel,
    ys,
    ylabel,
    yflip,
    labels;
    skip_first = false,
    highlight_first = false,
    highlight_last = false,
)
    plots = []
    for view_index in 1:length(xlims_list)
        legend = view_index == length(xlims_list) ? :outertopright : :none
        kwargs = (;
            palette = join_palettes(:seaborn_bright, :seaborn_dark),
            xlims = xlims_list[view_index],
            xticks = xticks_list[view_index],
            yflip,
            legend,
        )
        if view_index == 1
            kwargs = (; kwargs..., ylabel = ylabel)
        else
            kwargs = (; kwargs..., yformatter = _ -> "")
        end
        push!(plots, plot(; kwargs...))
        for (i, label) in enumerate(labels)
            i == 1 && skip_first && continue
            plot!(
                xs[i],
                ys[i] isa AbstractArray ? ys[i] : ys;
                seriescolor = (
                    i == 1 && highlight_first ||
                    i == length(labels) && highlight_last
                ) ? :black : (highlight_first ? i - 1 : i),
                label = label,
            )
        end
    end
    plot(
        plot(plots...; layout = (1, length(plots))),
        plot(;
            annotations = (0.44, 0, Plots.text(xlabel, 12)),
            framestyle = :none,
        );
        layout = Plots.grid(2, 1; heights = [0.99, 0.01])
    )
    savefig(joinpath(@__DIR__, "$filename.png"))
end

function get_horizontal_mean_debug(Q, dg, interp)
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

    nlev = interp ? (Nq3 - 1) * nvertelem + 1 : Nq3 * nvertelem
    ncol = Nq1 * Nq2 * length(horzelems)
    
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

    keys = ("F",) # ("p", "T", "q_vap_sat", "F")
    has_source = radiation_model(balance_law).tendency_type isa Source
    if has_source
        keys = (keys..., "src_dg", "src_deriv2", "src_deriv4", "src_deriv6")
    end
    aux_dict = Dict(map(key -> (key, zeros(FT, nlev)), keys))

    if ncol > 0
        for ev in 1:nvertelem
            for k in 1:Nq3
                v = interp ? (Nq3 - 1) * (ev - 1) + k : Nq3 * (ev - 1) + k
                for eh in horzelems, i in 1:Nq1, j in 1:Nq2
                    e = nvertelem * (eh - 1) + ev
                    ijk = Nq1 * (Nq2 * (k - 1) + (j - 1)) + i
                    for s in 1:num_state_prognostic
                        local_state_prognostic[s] = Q[ijk, s, e]
                    end
                    for s in 1:num_state_auxiliary
                        local_state_auxiliary[s] = state_auxiliary[ijk, s, e]
                    end
                    # thermo_state = recover_thermo_state(balance_law, prog, aux)
                    # aux_dict["p"][v] += air_pressure(thermo_state)
                    # aux_dict["T"][v] += air_temperature(thermo_state)
                    # aux_dict["q_vap_sat"][v] += q_vap_saturation(thermo_state)
                    aux_dict["F"][v] += aux.radiation.flux
                    if has_source
                        aux_dict["src_dg"][v] += aux.radiation.src_dg
                        aux_dict["src_deriv2"][v] += aux.radiation.src_deriv2
                        aux_dict["src_deriv4"][v] += aux.radiation.src_deriv4
                        aux_dict["src_deriv6"][v] += aux.radiation.src_deriv6
                    end
                end
                if interp && ev > 1 && k == 1
                    for array in values(aux_dict)
                        array[v] /= 2 * ncol
                    end
                elseif !(interp && ev < nvertelem && k == Nq3)
                    for array in values(aux_dict)
                        array[v] /= ncol
                    end
                end
            end
        end
    end

    @assert(aux_dict["F"] ≈ get_horizontal_mean(
        grid,
        state_auxiliary,
        vars_state(balance_law, Auxiliary(), FT);
        interp = interp,
    )["radiation.flux"])

    as = dg.modeldata.rrtmgp_data.solver.as
    keys = ("p_lev", "t_lev", "p_lay", "t_lay")
    rrtmgp_dict =
        Dict(map(key -> (key, sum(getfield(as, Symbol(key)), dim = 2) ./ as.ncol), keys))
    
    return dict
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

function driver_configuration(
    FT,
    compressibility,
    radiation,
    moisture,
    temp_profile,
    init_state_prognostic,
    numerical_flux,
    solver_type,
    config_type,
    domain_height,
    domain_width,
    polyorder_vert,
    polyorder_horz,
    nelem_vert,
    nelem_horz,
    stretch,
    label,
)
    source = ()
    compressibility isa Compressible && (source = (source..., Gravity()))
    radiation.tendency_type isa Source && (source = (source..., Radiation()))
    model = AtmosModel{FT}(
        typeof(config_type),
        param_set;
        init_state_prognostic = init_state_prognostic,
        # problem = AtmosProblem(
        #     boundaryconditions = (
        #         AtmosBC(),
        #         AtmosBC(energy = PrescribedEnergyFlux(
        #             (state, aux, t) ->
        #             aux.radiation.flux_dn_lw + aux.radiation.flux_dn_dir_sw,
        #         )),
        #     ),
        #     init_state_prognostic = init_state_prognostic,
        # ),
        ref_state = HydrostaticState(
            temp_profile,
            moisture isa CappedMonotonicRelativeHumidity ?
                moisture.max_relhum : FT(0);
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
        stretching_vert = stretch == 0 ?
            nothing : Topologies.SingleExponentialStretching(stretch)
        config_kwargs = (
            model = model,
            grid_stretching = (nothing, nothing, stretching_vert),
        )
    else # config_type isa AtmosGCMConfigType
        config_fun = ClimateMachine.AtmosGCMConfiguration
        config_args =
            ((nelem_horz, nelem_vert), domain_height, param_set, nothing)
        stretching_vert = stretch == 0 ?
            nothing : Topologies.SingleExponentialStretching(stretch)
        config_kwargs = (model = model, grid_stretching = (stretching_vert,))
        isnothing(domain_width) || @warn "No domain width for GCM config"
    end

    return config_fun(
        label,
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
    timestart,
    timeend,
    substep_duration,
    substeps_per_step,
)
    grid = driver_config.grid
    bl = driver_config.bl

    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        ode_dt = substep_duration,
        Courant_number = 0.5, # irrelevant; lets us use LSRKEulerMethod
        diffdir = VerticalDirection(),
        direction = VerticalDirection(),
        timeend_dt_adjust = false, # prevents dt from getting modified
        modeldata = (rrtmgp_data = RRTMGPData(bl, grid),),
    )

    solver = solver_config.solver
    Q = solver_config.Q
    state_auxiliary = solver_config.dg.state_auxiliary

    function cb_set_timestep()
        if getsteps(solver) == substeps_per_step
            updatedt!(solver, substeps_per_step * substep_duration)
        end
    end
    function cb_filter()
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
        # cb_filter,
        cb_progress,
        cb_progress_finish,
    )
end

function solve_and_plot_detailed(driver_config, interp, args...)
    solver_config, callbacks = setup_solver(driver_config, args...)
    xinfo, yinfo = debug_info(solver_config.dg)

    xvalue_arrays = Dict(map(key -> (key, []), collect(keys(xinfo))))
    labels = []

    function push_diagnostics_data()
        xvalues, _ = debug_values(solver_config.Q, solver_config.dg, interp)
        for key in keys(xvalue_arrays)
            push!(xvalue_arrays[key], xvalues[key])
        end
        push!(labels, time_string(gettime(solver_config.solver)))
    end
    next_push_step = 1
    function cb_diagnostics()
        if (
            getsteps(solver_config.solver) == # - substeps_per_step + 1 ==
            next_push_step
        )
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

    _, yvalues = debug_values(solver_config.Q, solver_config.dg, interp)
    for (yname, (ylabel, yunits, yflip)) in yinfo
        for (xname, (xlabel, xunits, xtype, skip_first)) in xinfo
            if occursin("heat_rate", xname)
                plot_multiple_views(
                    (:auto, (-1, 1)),
                    (:auto, :auto),
                    "$(xname)_$yname",
                    xvalue_arrays[xname],
                    "$xlabel$xunits",
                    yvalues["$(yname)_$xtype"],
                    "$ylabel$yunits",
                    yflip,
                    labels;
                    skip_first = skip_first,
                    highlight_first = true,
                    highlight_last = true,
                )
            else
                plot_multiple(
                    "$(xname)_$yname",
                    xvalue_arrays[xname],
                    "$xlabel$xunits",
                    yvalues["$(yname)_$xtype"],
                    "$ylabel$yunits",
                    yflip,
                    labels;
                    skip_first = skip_first,
                    highlight_first = true,
                    highlight_last = true,
                )
            end
        end
    end
end

function generate_and_save_reference(filename, driver_config, args...)
    solver_config, callbacks = setup_solver(driver_config, args...)
    ClimateMachine.invoke!(solver_config; user_callbacks = callbacks)
    serialize(
        joinpath(@__DIR__, filename),
        (solver_config.Q, solver_config.dg),
    )
end

extract_reference(filename::AbstractString, args...) =
    deserialize(joinpath(@__DIR__, filename))
function extract_reference(driver_config, args...)
    solver_config, callbacks = setup_solver(driver_config, args...)
    ClimateMachine.invoke!(solver_config; user_callbacks = callbacks)
    return solver_config.Q, solver_config.dg
end

function solve_and_plot_comparison(
    reference,
    driver_configs,
    labels,
    interp,
    args...,
)
    ref_Q, ref_dg = extract_reference(reference, args...)
    FT = eltype(ref_Q)
    xinfo, yinfo = debug_info(ref_dg)
    ref_xvalues, ref_yvalues = debug_values(ref_Q, ref_dg, interp)

    xvalue_arrays = Dict(map(key -> (key, []), collect(keys(ref_xvalues))))
    yvalue_arrays = Dict(map(key -> (key, []), collect(keys(ref_yvalues))))
    for i in 1:length(driver_configs)
        solver_config, callbacks = setup_solver(driver_configs[i], args...)
        ClimateMachine.invoke!(solver_config; user_callbacks = callbacks)
        xvalues, yvalues =
            debug_values(solver_config.Q, solver_config.dg, interp)
        for key in keys(xvalue_arrays)
            push!(xvalue_arrays[key], xvalues[key])
        end
        for key in keys(yvalue_arrays)
            push!(yvalue_arrays[key], yvalues[key])
        end
    end

    for (yname, (ylabel, yunits, yflip)) in yinfo
        for (xname, (xlabel, xunits, xtype, _)) in xinfo
            plot_multiple(
                "$(xname)_$yname",
                [ref_xvalues[xname], xvalue_arrays[xname]...],
                "$xlabel$xunits",
                [
                    ref_yvalues["$(yname)_$xtype"],
                    yvalue_arrays["$(yname)_$xtype"]...,
                ],
                "$ylabel$yunits",
                yflip,
                ["Reference", labels...];
                highlight_first = true,
            )
            if occursin(r"p|T|vmr|F|heat_rate", xname)
                transform = yflip ? reverse : identity
                xexact_interp = Spline1D(
                    transform(ref_yvalues["$(yname)_$xtype"]),
                    transform(ref_xvalues[xname]);
                    k = 1,
                    bc = "extrapolate",
                    s = FT(0),
                )
                plot_multiple(
                    "$(xname)_err_$yname",
                    map(
                        (x, y) -> begin
                            xexact = xexact_interp(y)
                            xerr = (x .- xexact) ./ abs.(xexact) .* FT(100)
                            xerr[x .== xexact .== FT(0)] .= FT(0)
                            xerr
                        end,
                        xvalue_arrays[xname],
                        yvalue_arrays["$(yname)_$xtype"],
                    ),
                    "relative error in $xlabel [%]",
                    yvalue_arrays["$(yname)_$xtype"],
                    "$ylabel$yunits",
                    yflip,
                    labels,
                )
            end
        end
    end
end

function solve_and_plot_multiple(
    driver_configs,
    FT,
    optics_symb,
    optical_props_symb,
    timestart,
    timeend,
    substep_duration,
    substeps_per_step,
    ids,
    id_name,
    id_label,
    interp,
    disable_shortwave,
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
            optics_symb,
            optical_props_symb,
            timestart,
            timeend,
            substep_duration,
            substeps_per_step,
            disable_shortwave,
        )
        zs = get_z(driver_config.grid; z_scale = 1e-3, rm_dupes = interp)
        push!(data["nodal_z"], zs)
        push!(
            data["elem_z"],
            zs[[1:driver_config.polyorders[2] + Int(!interp):end..., end]],
        )
        debug = get_horizontal_mean_debug(
            solver_config.Q,
            solver_config.dg,
            interp,
        )
        push!(data["F_init"], debug["F"])
        ClimateMachine.invoke!(solver_config; user_callbacks = callbacks)
        debug = get_horizontal_mean_debug(
            solver_config.Q,
            solver_config.dg,
            interp,
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
    
    temp_profile = DecayingTemperatureProfile{FT}(param_set)
    pressure_profile = z -> temp_profile(param_set, z)[2] # only used for ozone
    init_state_prognostic = SimpleInitializer(temp_profile)
    init_rrtmgp_extension = SimpleInitializer(temp_profile)

    vmr_co2_override = nothing # FT(0.000365) # current global mean
    upper_bc = DefaultUpperRRTMGPBC()
    layer_interp = GeometricCenter{Dens}()

    compressibility = Anelastic1D()
    radiation = RRTMGPModel(
        Source(),
        upper_bc,
        layer_interp,
        :Full,
        :TwoStream,
        pressure_profile,
        init_rrtmgp_extension,
        vmr_co2_override,
    )
    moisture = CappedConstantSpecificHumidity()

    numerical_flux = CentralNumericalFluxFirstOrder()
    solver_type = ExplicitSolverType(; solver_method = LSRKEulerMethod)
    timestart = FT(0)
    timeend = FT(year_to_s * 0)
    substep_duration = FT(15 * 60 * 60)
    substeps_per_step = 1

    config_type = AtmosLESConfigType()
    domain_height = FT(50e3)
    domain_width = FT(50) # Ignored for AtmosGCMConfigType
    polyorder_vert = 4
    polyorder_horz = 4
    nelem_vert = 10
    nelem_horz = 1 # Ignored for SingleStackConfigType
    stretch = FT(0) # Ignored for SingleStackConfigType

    interp = true

    # stretch = bottomΔz_stretch(
    #     nelem_vert,
    #     polyorder_vert,
    #     domain_height,
    #     FT(200),
    #     :node,
    # )
    # println("Stretch: $stretch")
    # driver_config = driver_configuration(
    #     FT,
    #     compressibility,
    #     radiation,
    #     moisture,
    #     temp_profile,
    #     init_state_prognostic,
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
    #     "Simulation",
    # )
    # solve_and_plot_detailed(
    #     driver_config,
    #     interp,
    #     timestart,
    #     timeend,
    #     substep_duration,
    #     substeps_per_step,
    # )

    ref_driver_config = driver_configuration(
        FT,
        compressibility,
        radiation,
        moisture,
        temp_profile,
        init_state_prognostic,
        numerical_flux,
        solver_type,
        config_type,
        FT(z_toa),
        domain_width,
        polyorder_vert,
        polyorder_horz,
        111,
        nelem_horz,
        stretch,
        "Reference State",
    )
    # generate_and_save_reference(
    #     "ref_1yr_5hr_default_CO2_capped_constant_q_centroid_dens_site_56_2stream.bin",
    #     ref_driver_config,
    #     timestart,
    #     timeend,
    #     substep_duration,
    #     substeps_per_step,
    # )
    # return

    ref_filename = "ref_1yr_5hr_default_CO2_capped_constant_q_centroid_dens_site_28_2stream.bin"
    driver_configs = []
    labels = []
    # for site in (28, 56, 70)
    #     radiation = RRTMGPModel(
    #         Source(),
    #         UpwardRRTMGPDomainExtension(20, true, false, nothing),
    #         layer_interp,
    #         :Full,
    #         :TwoStream,
    #         pressure_profile,
    #         SimpleInitializer(
    #             temp_profile,
    #             "ref_1yr_5hr_default_CO2_capped_constant_q_centroid_dens_site_$(site)_2stream.bin",
    #         ),
    #         vmr_co2_override,
    #     )
    #     label = "Site $site Reference"
    #     push!(
    #         driver_configs,
    #         driver_configuration(
    #             FT,
    #             compressibility,
    #             radiation,
    #             moisture,
    #             temp_profile,
    #             init_state_prognostic,
    #             numerical_flux,
    #             solver_type,
    #             config_type,
    #             domain_height,
    #             domain_width,
    #             polyorder_vert,
    #             polyorder_horz,
    #             nelem_vert,
    #             nelem_horz,
    #             stretch,
    #             label,
    #         )
    #     )
    #     push!(labels, label)
    # end
    # for domain_height_km in (30, 50)
    #     for (init_rrtmgp_ext, upper_bc, label) in (
    #         # (
    #         #     init_rrtmgp_extension,
    #         #     ConstantUpperRRTMGPBC(10, true),
    #         #     "10 Layer Constant Flux P-Interp.",
    #         # ),
    #         # (
    #         #     init_rrtmgp_extension,
    #         #     UpwardRRTMGPDomainExtension(10, true, false, nothing),
    #         #     "10 Layer Ext. File No Update P-Interp.",
    #         # ),
    #         # (
    #         #     SimpleInitializer(temp_profile, ref_filename),
    #         #     ConstantUpperRRTMGPBC(20, false),
    #         #     "$domain_height_km km; 20 precomp lay; const ref",
    #         # ),
    #         (
    #             SimpleInitializer(temp_profile, ref_filename),
    #             UpwardRRTMGPDomainExtension(1, true, false, nothing),
    #             "$domain_height_km km; 1 ghost lay; const ref PI",
    #         ),
    #         (
    #             SimpleInitializer(temp_profile, ref_filename),
    #             UpwardRRTMGPDomainExtension(2, true, false, nothing),
    #             "$domain_height_km km; 2 ghost lay; const ref PI",
    #         ),
    #         (
    #             SimpleInitializer(temp_profile, ref_filename),
    #             UpwardRRTMGPDomainExtension(20, false, false, nothing),
    #             "$domain_height_km km; 20 ghost lay; const ref",
    #         ),
    #         (
    #             SimpleInitializer(temp_profile, ref_filename),
    #             UpwardRRTMGPDomainExtension(20, true, false, nothing),
    #             "$domain_height_km km; 20 ghost lay; const ref PI",
    #         ),
    #         (
    #             SimpleInitializer(temp_profile, ref_filename),
    #             UpwardRRTMGPDomainExtension(20, false, true, nothing),
    #             "$domain_height_km km; 20 ghost lay; dynamic ref",
    #         ),
    #         # (
    #         #     SimpleInitializer(temp_profile, FT(0)),
    #         #     UpwardRRTMGPDomainExtension(ext_nlay, true, false, nothing),
    #         #     "$domain_height_km km; $ext_nlay ghost lay; const file PI",
    #         # ),
    #         # (
    #         #     init_rrtmgp_extension,
    #         #     UpwardRRTMGPDomainExtension(1, false, false, nothing),
    #         #     "1 Layer Ext. File No Update",
    #         # ),
    #         # (
    #         #     SimpleInitializer(temp_profile, FT(210), FT(0)),
    #         #     UpwardRRTMGPDomainExtension(1, false, true, nothing),
    #         #     "1 Layer Ext. T_top = 210 K",
    #         # ),
    #         # (
    #         #     SimpleInitializer(temp_profile, FT(230), FT(0)),
    #         #     UpwardRRTMGPDomainExtension(1, false, true, nothing),
    #         #     "1 Layer Ext. T_top = 230 K",
    #         # ),
    #         # (
    #         #     SimpleInitializer(temp_profile, FT(250), FT(0)),
    #         #     UpwardRRTMGPDomainExtension(ext_nlay, false, true, nothing),
    #         #     "$domain_height_km km; $ext_nlay ghost lay; 250 K top dynamic",
    #         # ),
    #         # (
    #         #     SimpleInitializer(temp_profile, FT(NaN), FT(0)),
    #         #     UpwardRRTMGPDomainExtension(1, false, true, FT(0.5)),
    #         #     "1 Layer Ext. ∂T/∂p = 0.5 K/Pa",
    #         # ),
    #         # (
    #         #     SimpleInitializer(temp_profile, FT(NaN), FT(0)),
    #         #     UpwardRRTMGPDomainExtension(1, false, true, FT(0.6)),
    #         #     "1 Layer Ext. ∂T/∂p = 0.6 K/Pa",
    #         # ),
    #     )
    #         radiation = RRTMGPModel(
    #             Source(),
    #             upper_bc,
    #             layer_interp,
    #             :Full,
    #             :TwoStream,
    #             pressure_profile,
    #             init_rrtmgp_ext,
    #             vmr_co2_override,
    #         )
    #         push!(
    #             driver_configs,
    #             driver_configuration(
    #                 FT,
    #                 compressibility,
    #                 radiation,
    #                 moisture,
    #                 temp_profile,
    #                 init_state_prognostic,
    #                 numerical_flux,
    #                 solver_type,
    #                 config_type,
    #                 domain_height_km * FT(1e3),
    #                 domain_width,
    #                 polyorder_vert,
    #                 polyorder_horz,
    #                 domain_height_km ÷ 5,
    #                 nelem_horz,
    #                 stretch,
    #                 label,
    #             )
    #         )
    #         push!(labels, label)
    #     end
    # end
    for nelem_vert in 5:5:25
        for _ in 1:3
            radiation = RRTMGPModel(
                Source(),
                UpwardRRTMGPDomainExtension(1, true, false, nothing),
                layer_interp,
                :Full,
                :TwoStream,
                pressure_profile,
                SimpleInitializer(temp_profile, ref_filename),
                vmr_co2_override,
            )
            label = "N = $nelem_vert"
            push!(
                driver_configs,
                driver_configuration(
                    FT,
                    compressibility,
                    radiation,
                    moisture,
                    temp_profile,
                    init_state_prognostic,
                    numerical_flux,
                    solver_type,
                    config_type,
                    domain_height,
                    domain_width,
                    polyorder_vert,
                    polyorder_horz,
                    nelem_vert,
                    nelem_horz,
                    stretch,
                    label,
                )
            )
            push!(labels, label)
        end
    end
    # for layer_interp_type in (
    #     GeometricCenter,
    #     # WeightedMean,
    #     CenterOfMass,
    #     CenterOfPres,
    # )
    #     for implied_var in (
    #         Dens,
    #         # Temp,
    #         Pres,
    #     )
    #         radiation = RRTMGPModel(
    #             Source(),
    #             UpwardRRTMGPDomainExtension(1, true, false, nothing),
    #             layer_interp_type{implied_var}(),
    #             :Full,
    #             :TwoStream,
    #             pressure_profile,
    #             SimpleInitializer(temp_profile, ref_filename),
    #             vmr_co2_override,
    #         )
    #         label = "$layer_interp_type{$implied_var}"
    #         push!(
    #             driver_configs,
    #             driver_configuration(
    #                 FT,
    #                 compressibility,
    #                 radiation,
    #                 moisture,
    #                 temp_profile,
    #                 init_state_prognostic,
    #                 numerical_flux,
    #                 solver_type,
    #                 config_type,
    #                 domain_height,
    #                 domain_width,
    #                 polyorder_vert,
    #                 polyorder_horz,
    #                 nelem_vert,
    #                 nelem_horz,
    #                 stretch,
    #                 label,
    #             )
    #         )
    #         push!(labels, label)
    #     end
    # end
    solve_and_plot_comparison(
        ref_driver_config, # ref_filename,
        driver_configs,
        labels,
        interp,
        timestart,
        timeend,
        substep_duration,
        substeps_per_step,
    )
end

main()