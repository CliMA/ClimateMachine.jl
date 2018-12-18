function air_pressure(T, dens, q_t, q_v)
# Computes air pressure from equation of state (ideal gas law) given density,
# temperature, and specific humidities of water vapor and suspended total water

p = dens * R_d * T * (1 - q_t +  R_v/R_d * q_v)

end
