abstract type EnergyBC end

using ..TurbulenceClosures
using StaticArrays
using ..SurfaceFluxes

"""
    Insulating() :: EnergyBC

No energy flux across the boundary.
"""
struct Insulating <: EnergyBC end
function atmos_energy_boundary_state!(nf, bc_energy::Insulating, atmos, args...) end
function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::Insulating,
    atmos,
    args...,
) end


"""
    PrescribedTemperature(fn) :: EnergyBC

Prescribe the temperature at the boundary by `fn`, a function with signature
`fn(state, aux, t)` returning the temperature (in K).
"""
struct PrescribedTemperature{FN} <: EnergyBC
    fn::FN
end
function atmos_energy_boundary_state!(
    nf,
    bc_energy::PrescribedTemperature,
    atmos,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    bctype,
    t,
    args...,
)
    FT = eltype(aux⁻)
    _T_0::FT = T_0(atmos.param_set)
    _cv_d::FT = cv_d(atmos.param_set)

    T = bc_energy.fn(state⁻, aux⁻, t)
    E_int⁺ = state⁺.ρ * _cv_d * (T - _T_0)
    state⁺.ρe =
        E_int⁺ + state⁺.ρ * gravitational_potential(atmos.orientation, aux⁻)
end
function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::PrescribedTemperature,
    atmos,
    fluxᵀn,
    n⁻,
    state⁻,
    diff⁻,
    hyperdiff⁻,
    aux⁻,
    state⁺,
    diff⁺,
    hyperdiff⁺,
    aux⁺,
    bctype,
    t,
    args...,
)

    # TODO: figure out a better way...
    ν, D_t, _ = turbulence_tensors(atmos, state⁻, diff⁻, aux⁻, t)
    d_h_tot = -D_t .* diff⁻.∇h_tot
    nd_h_tot = dot(n⁻, d_h_tot)
    # both sides involve projections of normals, so signs are consistent
    fluxᵀn.ρe += nd_h_tot * state⁻.ρ
end


"""
    PrescribedEnergyFlux(fn) :: EnergyBC

Prescribe the net inward energy flux across the boundary by `fn`, a function
with signature `fn(state, aux, t)`, returning the flux (in W/m^2).
"""
struct PrescribedEnergyFlux{FN} <: EnergyBC
    fn::FN
end
function atmos_energy_boundary_state!(
    nf,
    bc_energy::PrescribedEnergyFlux,
    atmos,
    args...,
) end
function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::PrescribedEnergyFlux,
    atmos,
    fluxᵀn,
    n⁻,
    state⁻,
    diff⁻,
    hyperdiff⁻,
    aux⁻,
    state⁺,
    diff⁺,
    hyperdiff⁺,
    aux⁺,
    bctype,
    t,
    args...,
)

    # DG normal is defined in the outward direction
    # we want to prescribe the inward flux
    fluxᵀn.ρe -= bc_energy.fn(state⁻, aux⁻, t)
end


"""
    BulkFormulaEnergy(fn) :: EnergyBC

Prescribe the net inward energy flux across the boundary by `fn`, a function
with signature `fn(state, aux, t)`, returning the flux (in W/m^2).
"""
struct BulkFormulaEnergy{FN} <: EnergyBC
    fn::FN
end
function atmos_energy_boundary_state!(
    nf,
    bc_energy::BulkFormulaEnergy,
    atmos,
    args...,
) end

function compute_interior_properties(
    atmos,
    state1⁻,
    diff1⁻,
    aux1⁻,
)
    FT = eltype(state1⁻)
    TS = thermo_state(atmos, state1⁻, aux1⁻)
    u_1 = state1⁻.ρu / state1⁻.ρ
    thv_1 = virtual_pottemp(TS)
    qt_1 = state1⁻.moisture.ρq_tot
    return (u_1, thv_1, qt_1)
end

function compute_surface_properties(
    atmos,
    state⁻,
    diff⁻,
    aux⁻
)
    FT = eltype(state⁻)
    TS = thermo_state(atmos, state⁻, aux⁻)
    u_sfc = state⁻.ρu / state⁻.ρ
    thv_sfc = virtual_pottemp(TS)
    qt_sfc = FT(0)
    return (u_sfc, thv_sfc, qt_sfc)
end

function atmos_energy_normal_boundary_flux_second_order!(
    nf,
    bc_energy::BulkFormulaEnergy,
    atmos,
    fluxᵀn,
    n⁻,
    state⁻,
    diff⁻,
    hyperdiff⁻,
    aux⁻,
    state⁺,
    diff⁺,
    hyperdiff⁺,
    aux⁺,
    bctype,
    t,
    state1⁻,
    diff1⁻,
    aux1⁻
)
    # Get thermodynamic states and float-type
    # Establish thermodynamic state
    FT = eltype(state⁻)
    TS = thermo_state(atmos, state⁻, aux⁻)
    TS1 = thermo_state(atmos, state1⁻, aux1⁻)
    phase = thermo_state(atmos, state⁻, aux⁻)
    # DG normal is defined in the outward direction, want the inward flux
    # Establish initial guess for Monin-Obukhov fluxes
    
    # Establish interface for surface condition inputs
    Δz = aux1⁻.coord[3];
    a = FT(4.7);
    x_initial = MVector{4,FT}(FT(10), -FT(0.1), -FT(0.001), FT(0))
    z_0  = MVector{3,FT}(0.01, 0.001, 0.0001);
    (u_1⁻,thv_l,qt_1) = compute_interior_properties(atmos,state1⁻,diff1⁻,aux1⁻)
    x_1 = MVector(u_1⁻[1],thv_l,qt_1)
    (u_sfc,thv_sfc,qt_sfc) = compute_surface_properties(atmos,state⁻,diff⁻,aux⁻)
    x_sfc = MVector(u_sfc[1],thv_sfc,qt_sfc)
    h_int = total_specific_enthalpy(atmos,atmos.moisture,state1⁻,aux1⁻)
    Φ_int = gravitational_potential(atmos.orientation, aux1⁻)
    Φ_sfc = gravitational_potential(atmos.orientation, aux⁻)
    R_m = gas_constant_air(phase)
    # Allow struct field for surface temperature prescription
    T_sfc = FT(266)
    e_tot = state⁻.ρe * (1 / state⁻.ρ)
    h_sfc = e_tot + R_m * T_sfc
    dimensionless_number = MVector{3,FT}(1,1/3,1/3)
    θ_bar = virtual_pottemp(TS1);
    qt_bar = FT(0);
    F_exchange = MVector{3,FT}(-0.1,-100, 0)
    sfc = surface_conditions(atmos.param_set,
                            x_initial,
                            x_1,
                            x_sfc,
                            z_0,
                            F_exchange,
                            dimensionless_number,
                            θ_bar,
                            qt_bar,
                            Δz,
                            Δz / 20,
                            a,
                            nothing
                           )
    @show(sfc.K_exchange)
    C_h = 0.20 #sfc.K_exchange[2]; # Exchange coefficient for energy
    # int refers to first interior nodal value
    Δh = h_int - h_sfc + Φ_int - Φ_sfc  
    Pu1⁻ = u_1⁻ - dot(u_1⁻, n⁻) .* SVector(n⁻)
    normPu1⁻ = norm(Pu1⁻)
    fluxᵀn.ρe -= -C_h * state⁻.ρ * Δh * normPu1⁻
end
