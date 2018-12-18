function latent_heat_vapor(T)
# Computes latent heat of vaporization from Kirchhoff's relation, assuming
# constant isobaric specific heat capacities of liquid and vapor

L_v = L_v0 + (cp_v - cp_l) * (T - T0)

end
