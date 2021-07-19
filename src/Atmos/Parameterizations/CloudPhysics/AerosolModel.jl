
struct MassFraction end
struct Dissociation end
struct AerosolMolarMass end
struct AerosolDensity end
struct MassMixRatio end
struct OsmoticCoeff end


struct ChemComposition
end

struct ChemMode{T}
    chem_comp::T
end



struct AerosolModes{T,PS}
    mode::T
    param_set::PS
end

get_param_set(am::AerosolModes) = am.param_set

am = AerosolModes((ChemMode((MolarMass(),))), ChemMode((MolarMass(), Density())), ParameterSet())

ps = get_param_set(am) # am.param_set
ams[1].tup[1] # MolarMass()

am[2].tup[2] # Density


