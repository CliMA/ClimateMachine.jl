function air_pressure(T, dens, q_t, q_v)
# Computes air pressure from equation of state (ideal gas law) given density,
# temperature, and specific humidities of water vapor and suspended total water

p = R_d * dens .* T .* (1 .+  R_v/R_d * q_v .- q_t)

end
