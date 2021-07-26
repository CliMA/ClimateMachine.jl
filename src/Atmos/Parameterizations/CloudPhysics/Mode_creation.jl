"""
Text file format:
material, molar mass, osmotic coefficient, density, mass fraction, dissociation
"""

"""
functionality: gathers the data about the aerosol particle material stored in another 
          file
parameters: string of the name of the material, path to the data file
returns: list of the values associated with the aerosol particle material
"""
function get_data(material::String, path::String)
    line = []
    for r in eachline(path)
        value = ""
        for index in 1:length(r)
            if r[index] != '"' && r[index] != ','
                value *= r[index] 
            end
            if (r[index] == ',' || index == length(r))
                if (length(line) == 0)
                    if (value != material)
                        continue
                    end
                end
                push!(line, value)
                value = "" 
            end      
        end
        if (length(line) != 0)
            break
        end
    end
    for values in 2:length(line)
        line[values] = parse(Float64, line[values])
    end
    return line[2:length(line)]
end

# Indexing get_data returned list to value meanings
d_molar_mass = 1
d_osmotic_coeff = 2
d_density = 3
d_mass_frac = 4
d_dissoc = 5

# getting data
seasalt = get_data("seasalt", "/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/particle_data.txt")
dust = get_data("dust", "/home/skadakia/clones/ClimateMachine.jl/src/Atmos/Parameterizations/CloudPhysics/particle_data.txt")

# START OF CREATING MODES
# Accumulation mode
r_dry_accum = 0.243 * 1e-6 # μm
stdev_accum = 1.4          # -
N_accum = 100.0 * 1e6      # 1/m3

# Coarse Mode
r_dry_coarse = 1.5 * 1e-6  # μm
stdev_coarse = 2.1         # -
N_coarse = 1.0 * 1e6       # 1/m3

# Building the test structures:

# 1) create aerosol modes
accum_mode_seasalt = mode(
    r_dry_accum,
    stdev_accum,
    N_accum,
    (1.0,),
    (seasalt[d_mass_frac],),
    (seasalt[d_osmotic_coeff],),
    (seasalt[d_molar_mass],),
    (seasalt[d_dissoc],),
    (seasalt[d_density],),
    1,
)

half_accum_mode_seasalt = mode(
    r_dry_accum,
    stdev_accum,
    .5 * N_accum,
    (1.0,),
    (seasalt[d_mass_frac],),
    (seasalt[d_osmotic_coeff],),
    (seasalt[d_molar_mass],),
    (seasalt[d_dissoc],),
    (seasalt[d_density],),
    1,
)

coarse_mode_seasalt = mode(
    r_dry_coarse,
    stdev_coarse,
    N_coarse,
    (1.0,), # mass mix ratio TODO
    (seasalt[d_mass_frac],),
    (seasalt[d_osmotic_coeff],),
    (seasalt[d_molar_mass],),
    (seasalt[d_dissoc],),
    (seasalt[d_density],),
    1,
)

accum_mode_seasalt_dust = mode(
    r_dry_accum,
    stdev_accum,
    N_accum,
    (0.5, 0.5), # mass mix ratio TODO
    (seasalt[d_mass_frac], dust[d_mass_frac]),
    (seasalt[d_osmotic_coeff], dust[d_osmotic_coeff]),
    (seasalt[d_molar_mass], dust[d_molar_mass]),
    (seasalt[d_dissoc], dust[d_dissoc]),
    (seasalt[d_density], dust[d_density]),
    2,
)

coarse_mode_seasalt_dust = mode(
    r_dry_coarse,
    stdev_coarse,
    N_coarse,
    (0.25, 0.75), # mass mix ratio TODO
    (seasalt[d_mass_frac], dust[d_mass_frac]),
    (seasalt[d_osmotic_coeff], dust[d_osmotic_coeff]),
    (seasalt[d_molar_mass], dust[d_molar_mass]),
    (seasalt[d_dissoc], dust[d_dissoc]),
    (seasalt[d_density], dust[d_density]),
    2,
)

zeroinitialN_accum_mode_seasalt = mode(
    r_dry_accum,
    stdev_accum,
    0.0,
    (1.0,),
    (seasalt[d_mass_frac],),
    (seasalt[d_osmotic_coeff],),
    (seasalt[d_molar_mass],),
    (seasalt[d_dissoc],),
    (seasalt[d_density],),
    1,
)

coarse_mode_seasalt_seasalt = mode(
    r_dry_coarse,
    stdev_coarse,
    N_coarse,
    (0.50, 0.50), # mass mix ratio TODO
    (seasalt[d_mass_frac], seasalt[d_mass_frac]),
    (seasalt[d_osmotic_coeff], seasalt[d_osmotic_coeff]),
    (seasalt[d_molar_mass], seasalt[d_molar_mass]),
    (seasalt[d_dissoc], seasalt[d_dissoc]),
    (seasalt[d_density], seasalt[d_density]),
    2,
)

# 2) create aerosol models
AM_1 = aerosol_model((accum_mode_seasalt,))
AM_2 = aerosol_model((coarse_mode_seasalt,))
AM_3 = aerosol_model((accum_mode_seasalt, coarse_mode_seasalt))
AM_4 = aerosol_model((accum_mode_seasalt_dust,))
AM_5 = aerosol_model((accum_mode_seasalt_dust, coarse_mode_seasalt_dust))
AM_6 = aerosol_model((zeroinitialN_accum_mode_seasalt,))
AM_7 = aerosol_model((coarse_mode_seasalt_seasalt, ))
AM_8 = aerosol_model((half_accum_mode_seasalt, half_accum_mode_seasalt))
# 3) bundle them together
AM_test_cases = [AM_1, AM_2, AM_3, AM_4, AM_5, AM_6, AM_7, AM_8]