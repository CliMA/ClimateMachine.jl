# Functions to compute saturation vapor pressures over liquid and ice

function sat_vapor_press_liquid(T)

    es = sat_vapor_press_generic(T, L_v0, cp_v - cp_l)

end

function sat_vapor_press_ice(T)

    es = sat_vapor_press_generic(T, L_s0, cp_v - cp_i)

end

function sat_vapor_press_generic(T, L_0, cp_diff)
# Computes vapor pressure by integration of Clausius-Clepeyron equation from T0,
# assuming constant isobaric specific heats of the phases and Kirchhoff's
# relation, which in this case gives a linear dependence of latent heat on
# temperature.
#
# That is, the saturation vapor pressure is a solution of
#
#       dlog(e_s)/dT = [L0 + cp_diff(T-T0)]/(R_v*T^2)

    es = sat_vapor_press_0 * exp.( (L_0 - cp_diff*T0)/R_v * (1.0/T0 .- 1.0./T)
        .+ cp_diff/R_v*log.(T/T0))

end
