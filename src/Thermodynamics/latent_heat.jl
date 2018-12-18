# Functions to compute latent heats of phase transitions

function latent_heat_vapor(T)

    L_v = latent_heat_generic(T, L_v0, cp_v - cp_l)

end

function latent_heat_sublim(T)

    L_s = latent_heat_generic(T, L_s0, cp_v - cp_i)

end

function latent_heat_fusion(T)

    L_f = latent_heat_generic(T, L_f0, cp_l - cp_i)

end

function latent_heat_generic(T, L_0, cp_diff)
# Computes latent heat for generic phase transition between
# two phases using Kirchhoff's relation.
# L0 is the latent heat of the phase transition at T0,
# and cp_diff is the difference between the isobaric specific heat
# capacities (cp in higher-temperature phase minus cp in
# lower-temperature phase)

    L = L_0 .+ cp_diff * (T .- T0)

end
