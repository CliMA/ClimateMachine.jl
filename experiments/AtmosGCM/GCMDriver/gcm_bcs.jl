# GCM Boundary Conditions
# This file contains helpers and lists currely avaiable options

# Helper for parsing `--surface-flux` command line argument
function parse_surface_flux_arg(
    arg,
    ::Type{FT},
    param_set,
    orientation,
    moisture,
) where {FT}
    if arg === nothing || arg == "default"
        boundarycondition = (AtmosBC(), AtmosBC())
    elseif arg == "bulk"
        if !isa(moisture, EquilMoist)
            error("need a moisture model for surface-flux: bulk")
        end
        _C_drag = C_drag(param_set)::FT
        bulk_flux = Varying_SST_TJ16(param_set, orientation, moisture) # GCM-specific function for T_sfc, q_sfc = f(latitude, height)
        #bulk_flux = (T_sfc, q_sfc) # prescribed constant T_sfc, q_sfc
        boundarycondition = (
            AtmosBC(
                energy = BulkFormulaEnergy(
                    (state, aux, t, normPu_int) -> _C_drag,
                    (state, aux, t) -> bulk_flux(state, aux, t),
                ),
                moisture = BulkFormulaMoisture(
                    (state, aux, t, normPu_int) -> _C_drag,
                    (state, aux, t) -> begin
                        _, q_tot = bulk_flux(state, aux, t)
                        q_tot
                    end,
                ),
            ),
            AtmosBC(),
        )
    else
        error("unknown surface flux: " * arg)
    end

    return boundarycondition
end

# Current options for GCM boundary conditions:

"""
struct Varying_SST_TJ16{PS, O, MM}
    param_set::PS
    orientation::O
    moisture::MM

Defines analytical function for prescribed T_sfc and q_sfc, following
Thatcher and Jablonowski (2016), used to calculate bulk surface fluxes.
T_sfc_pole = SST at the poles (default: 271 K), specified above
"""
struct Varying_SST_TJ16{PS, O, MM}
    param_set::PS
    orientation::O
    moisture::MM
end
function (st::Varying_SST_TJ16)(state, aux, t)
    FT = eltype(state)
    φ = latitude(st.orientation, aux)

    T_sfc_pole = FT(271.0)              # surface polar temperature
    Δφ = FT(26) * FT(π) / FT(180)       # latitudinal width of Gaussian function
    ΔSST = FT(29)                       # Eq-pole SST difference in K
    T_sfc = ΔSST * exp(-φ^2 / (2 * Δφ^2)) + T_sfc_pole

    eps = FT(0.622)
    ρ = state.ρ

    q_tot = state.moisture.ρq_tot / ρ
    q = PhasePartition(q_tot)

    e_int = internal_energy(st.moisture, st.orientation, state, aux)
    T = air_temperature(st.param_set, e_int, q)
    p = air_pressure(st.param_set, T, ρ, q)

    _T_triple = T_triple(st.param_set)::FT          # triple point of water
    _press_triple = press_triple(st.param_set)::FT  # sat water pressure at T_triple
    _LH_v0 = LH_v0(st.param_set)::FT                # latent heat of vaporization at T_triple
    _R_v = R_v(st.param_set)::FT                    # gas constant for water vapor

    q_sfc =
        eps / p *
        _press_triple *
        exp(-_LH_v0 / _R_v * (FT(1) / T_sfc - FT(1) / _T_triple))

    return T_sfc, q_sfc
end
