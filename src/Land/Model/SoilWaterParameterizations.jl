"""
    SoilWaterParameterizations

van Genuchten, Brooks and Corey, and Haverkamp parameters for and formulation of
  - hydraulic conductivity
  - matric potential

Hydraulic conductivity can be chosen to be dependent or independent of 
impedance, viscosity and moisture.

Functions for hydraulic head, effective saturation, pressure head, matric 
potential, and the relationship between augmented liquid fraction and liquid
fraction are also included.
"""
module SoilWaterParameterizations

using DocStringExtensions
using UnPack

export AbstractImpedanceFactor,
    NoImpedance,
    IceImpedance,
    impedance_factor,
    AbstractViscosityFactor,
    ConstantViscosity,
    TemperatureDependentViscosity,
    viscosity_factor,
    AbstractMoistureFactor,
    MoistureDependent,
    MoistureIndependent,
    moisture_factor,
    AbstractHydraulicsModel,
    vanGenuchten,
    BrooksCorey,
    Haverkamp,
    hydraulic_conductivity,
    effective_saturation,
    pressure_head,
    hydraulic_head,
    matric_potential,
    volumetric_liquid_fraction,
    inverse_matric_potential

"""
    AbstractImpedanceFactor{FT <: AbstractFloat}

"""
abstract type AbstractImpedanceFactor{FT <: AbstractFloat} end

"""
    AbstractViscosityFactor{FT <: AbstractFloat}
"""
abstract type AbstractViscosityFactor{FT <: AbstractFloat} end

"""
    AbstractMoistureFactor{FT <:AbstractFloat}
"""
abstract type AbstractMoistureFactor{FT <: AbstractFloat} end


"""
    AbstractsHydraulicsModel{FT <: AbstractFloat}

Hydraulics model is used in the moisture factor in hydraulic 
conductivity and in the matric potential. The single hydraulics model 
choice sets both of these.
"""
abstract type AbstractHydraulicsModel{FT <: AbstractFloat} end


"""
    vanGenuchten{FT} <: AbstractHydraulicsModel{FT}

The necessary parameters for the van Genuchten hydraulic model; 
defaults are for Yolo light clay.

 The user can supply either
floats or  functions of `aux` (`aux.x`, `aux.y`, `aux.z`), which return a scalar float.
Internally,
the parameters will be converted to type FT and the functions
altered to return type FT, so the parameters must be abstract floats
or functions that return abstract floats.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct vanGenuchten{FT, T1, T2, T3, T4} <: AbstractHydraulicsModel{FT}
    "Exponent parameter - used in matric potential"
    n::T1
    "used in matric potential. The inverse of this carries units in 
     the expression for matric potential (specify in inverse meters)."
    α::T2
    "Hydraulic conductivity exponent"
    L::T3
    "Exponent parameter - determined by n, used in hydraulic conductivity"
    m::T4
end


function vanGenuchten(
    ::Type{FT};
    n::Union{AbstractFloat, Function} = FT(1.43),
    α::Union{AbstractFloat, Function} = FT(2.6),
    L::Union{AbstractFloat, Function} = FT(0.5),
) where {FT}
    nt = n isa AbstractFloat ? FT(n) : (aux) -> FT(n(aux))
    mt =
        n isa AbstractFloat ? FT(1) - FT(1) / nt :
        (aux) -> FT(1) - FT(1) / nt(aux)
    αt = α isa AbstractFloat ? FT(α) : (aux) -> FT(α(aux))
    Lt = L isa AbstractFloat ? FT(L) : (aux) -> FT(L(aux))
    
    args = (nt, αt, Lt, mt)
    return vanGenuchten{FT, typeof.(args)...}(args...)
end


"""
    (model::vanGenuchten)(aux)

Evaluate the hydraulic model parameters at aux, and return
a struct of type `vanGenuchten` with those *constant* parameters,
which will be of float type FT.
"""
function (model::vanGenuchten{FT, T1, T2, T3, T4})(aux) where {FT, T1, T2, T3, T4}
    @unpack n, α ,L = model
    fn = typeof(n) == FT ? n : n(aux)
    fα = typeof(α) == FT ? α : α(aux)
    fL = typeof(L) == FT ? L : L(aux)
    return vanGenuchten(FT; n = fn, α = fα, L = fL)
end

"""
    BrooksCorey{FT,T1,T2} <: AbstractHydraulicsModel{FT}

The necessary parameters for the Brooks and Corey hydraulic model.

Defaults are chosen to somewhat mirror the Havercamp/vG Yolo light 
clay hydraulic conductivity/matric potential.  The user can supply either
floats or functions of `aux` (`aux.x`, `aux.y`, `aux.z`), which return a scalar float. 
Internally,
the parameters will be converted to type FT and the functions
altered to return type FT, so the parameters must be abstract floats
or functions that return abstract floats.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct BrooksCorey{FT, T1, T2} <: AbstractHydraulicsModel{FT}
    "ψ_b - used in matric potential. Units of meters."
    ψb::T1
    "Exponent used in matric potential and hydraulic conductivity."
    m::T2
end

function BrooksCorey(
    ::Type{FT};
    ψb::Union{AbstractFloat, Function} = FT(0.1656),
    m::Union{AbstractFloat, Function} = FT(0.5),
) where {FT}
    mt = m isa AbstractFloat ? FT(m) : (aux) -> FT(m(aux))
    ψt = ψb isa AbstractFloat ? FT(ψb) : (aux) -> FT(ψb(aux))
    args = (ψt, mt)
    return BrooksCorey{FT, typeof.(args)...}(args...)
end

"""
    (model::BrooksCorey)(aux)

Evaluate the hydraulic model parameters at aux, and return
a struct of type `BrooksCorey` with those parameters.
"""
function (model::BrooksCorey{FT, T1, T2})(aux) where {FT, T1, T2}
    @unpack ψb, m = model
    fψ = typeof(ψb) == FT ? ψb : ψb(aux)
    fm = typeof(m) == FT ? m : m(aux)
    return BrooksCorey(FT; ψb = fψ, m = fm)
end


"""
    Haverkamp{FT,T1,T2,T3,T4,T5} <: AbstractHydraulicsModel{FT}

The necessary parameters for the Haverkamp hydraulic model for Yolo light
 clay.

Note that this only is used in creating a hydraulic conductivity function,
 and another formulation for matric potential must be used.
The user can supply either
floats or functions of `aux` (`aux.x`, `aux.y`, `aux.z`), which return a scalar float. Internally,
the parameters will be converted to type FT and the functions
altered to return type FT, so the parameters must be abstract floats
or functions that return abstract floats.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct Haverkamp{FT, T1, T2, T3, T4, T5} <: AbstractHydraulicsModel{FT}
    "exponent in conductivity"
    k::T1
    "constant A (units of cm^k) using in conductivity. Our sim is in meters"
    A::T2
    "Exponent parameter - using in matric potential"
    n::T3
    "used in matric potential. The inverse of this carries units in the 
     expression for matric potential (specify in inverse meters)."
    α::T4
    "Exponent parameter - determined by n, used in hydraulic conductivity"
    m::T5
end

function Haverkamp(
    ::Type{FT};
    k::Union{AbstractFloat, Function} = FT(1.77),
    A::Union{AbstractFloat, Function} = FT(124.6 / 100.0^1.77),
    n::Union{AbstractFloat, Function} = FT(1.43),
    α::Union{AbstractFloat, Function} = FT(2.6),
) where {FT}
    nt = n isa AbstractFloat ? FT(n) : (aux) -> FT(n(aux))
    mt =
        n isa AbstractFloat ? FT(1) - FT(1) / nt :
        (aux) -> FT(1) - FT(1) / nt(aux)
    αt = α isa AbstractFloat ? FT(α) : (aux) -> FT(α(aux))
    kt = k isa AbstractFloat ? FT(k) : (aux) -> FT(k(aux))
    At = A isa AbstractFloat ? FT(A) : (aux) -> FT(A(aux))

    args = (kt, At, nt, αt, mt)
    return Haverkamp{FT, typeof.(args)...}(args...)
end

"""
    (model::Haverkcamp)(aux)

Evaluate the hydraulic model parameters at aux, and return
a struct of type `Haverkamp` with those parameters.
"""
function (model::Haverkamp{FT, T1, T2, T3, T4, T5})(
    aux,
) where {FT, T1, T2, T3, T4, T5}
    @unpack k, A, n, α = model
    fn = typeof(n) == FT ? n : n(aux)
    fα = typeof(α) == FT ? α : α(aux)
    fA = typeof(A) == FT ? A : A(aux)
    fk = typeof(k) == FT ? k : k(aux)
    return Haverkamp(FT; k = fk, A = fA, n = fn, α = fα)
end


"""
    MoistureIndependent{FT} <: AbstractMoistureFactor{FT} end

Moisture independent moisture factor.
"""
struct MoistureIndependent{FT} <: AbstractMoistureFactor{FT} end


"""
    MoistureDependent{FT} <: AbstractMoistureFactor{FT} end

Moisture dependent moisture factor.
"""
struct MoistureDependent{FT} <: AbstractMoistureFactor{FT} end


"""
    moisture_factor(
        mm::MoistureDependent,
        hm::vanGenuchten{FT},
        S_l::FT,
    ) where {FT}

Returns the moisture factor of the hydraulic conductivy assuming a 
MoistureDependent and van Genuchten hydraulic model.

This is intended to be used with an instance of `vanGenuchten`
that has float parameters. 
"""
function moisture_factor(
    mm::MoistureDependent,
    hm::vanGenuchten{FT},
    S_l::FT,
) where {FT}
    m = hm.m
    L = hm.L
    if S_l < FT(1)
        K = S_l^L * (FT(1) - (FT(1) - S_l^(FT(1) / m))^m)^FT(2)
    else
        K = FT(1)
    end
    return K
end

"""
    moisture_factor(
        mm::MoistureDependent,
        hm::BrooksCorey{FT},
        S_l::FT,
    ) where {FT}

Returns the moisture factor of the hydraulic conductivy assuming a 
MoistureDependent and Brooks/Corey hydraulic model.

This is intended to be used with an instance of `BrooksCorey`
that has float parameters.
"""
function moisture_factor(
    mm::MoistureDependent,
    hm::BrooksCorey{FT},
    S_l::FT,
) where {FT}
    m = hm.m
    if S_l < FT(1)
        K = S_l^(FT(2) * m + FT(3))
    else
        K = FT(1)
    end
    return K
end

"""
    moisture_factor(
        mm::MoistureDependent,
        hm::Haverkamp{FT},
        S_l::FT,
    ) where {FT}

Returns the moisture factor of the hydraulic conductivy assuming a 
MoistureDependent and Haverkamp hydraulic model.

This is intended to be used with an instance of `Haverkamp`
that has float parameters.
"""
function moisture_factor(
    mm::MoistureDependent,
    hm::Haverkamp{FT},
    S_l::FT,
) where {FT}
    @unpack k, A, n, m, α = hm
    if S_l < FT(1)
        ψ = -((S_l^(-FT(1) / m) - FT(1)) * α^(-n))^(FT(1) / n)
        K = A / (A + abs(ψ)^k)
    else
        K = FT(1)
    end
    return K
end


"""
    moisture_factor(mm::MoistureIndependent,
                    hm::AbstractHydraulicsModel{FT},
                    S_l::FT,
    ) where {FT}

Returns the moisture factor in hydraulic conductivity when a 
MoistureIndependent model is chosen. Returns 1.

Note that the hydraulics model and S_l are not used, but are included 
as arguments to unify the function call.
"""
function moisture_factor(
    mm::MoistureIndependent,
    hm::AbstractHydraulicsModel{FT},
    S_l::FT,
) where {FT}
    Factor = FT(1.0)
    return Factor
end

"""
    ConstantViscosity{FT} <: AbstractViscosityFactor{FT}

A model to indicate a constant viscosity - independent of temperature - 
factor in hydraulic conductivity.
"""
struct ConstantViscosity{FT} <: AbstractViscosityFactor{FT} end


"""
    TemperatureDependentViscosity{FT} <: AbstractViscosityFactor{FT}

The necessary parameters for the temperature dependent portion of hydraulic 
conductivity.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct TemperatureDependentViscosity{FT} <:
                   AbstractViscosityFactor{FT}
    "Empirical coefficient"
    γ::FT = FT(2.64e-2)
    "Reference temperature"
    T_ref::FT = FT(288.0)
end


"""
    viscosity_factor(
        vm::ConstantViscosity{FT},
        T::FT,
    ) where {FT}

Returns the viscosity factor when we choose no temperature dependence, i.e. 
a constant viscosity. Returns 1.

T is included as an argument to unify the function call.
"""
function viscosity_factor(vm::ConstantViscosity{FT}, T::FT) where {FT}
    Theta = FT(1.0)
    return Theta
end

"""
    viscosity_factor(
        vm::TemperatureDependentViscosity{FT},
        T::FT,
    ) where {FT}

Returns the viscosity factor when we choose a TemperatureDependentViscosity.
"""
function viscosity_factor(
    vm::TemperatureDependentViscosity{FT},
    T::FT,
) where {FT}
    γ = vm.γ
    T_ref = vm.T_ref
    factor = FT(γ * (T - T_ref))
    Theta = FT(exp(factor))
    return Theta
end


"""
    NoImpedance{FT} <: AbstractImpedanceFactor{FT}

A model to indicate to dependence on ice for the hydraulic conductivity.
"""
struct NoImpedance{FT} <: AbstractImpedanceFactor{FT} end



"""
    IceImpedance{FT} <: AbstractImpedanceFactor{FT}

The necessary parameters for the empirical impedance factor due to ice.

# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct IceImpedance{FT} <: AbstractImpedanceFactor{FT}
    "Empirical coefficient from Hansson 2014. "
    Ω::FT = FT(7)
end

"""
    impedance_factor(
        imp::NoImpedance{FT},
        f_i::FT,
    ) where {FT}

Returns the impedance factor when no effect due to ice is desired. 
Returns 1.

The other arguments are included to unify the function call.
"""
function impedance_factor(imp::NoImpedance{FT}, f_i::FT) where {FT}
    gamma = FT(1.0)
    return gamma
end

"""
    impedance_factor(
        imp::IceImpedance{FT},
        f_i::FT,
    ) where {FT}

Returns the impedance factor when an effect due to the fraction of 
ice is desired. 
"""
function impedance_factor(imp::IceImpedance{FT}, f_i::FT) where {FT}
    Ω = imp.Ω
    gamma = FT(10.0^(-Ω * f_i))
    return gamma
end

"""
    hydraulic_conductivity(
        Ksat::FT,
        impedance::FT,
        viscosity::FT,
        moisture::FT,
    ) where {FT}

Returns the hydraulic conductivity.
"""
function hydraulic_conductivity(
    Ksat::FT,
    impedance::FT,
    viscosity::FT,
    moisture::FT,
) where {FT}
    K = Ksat * impedance * viscosity * moisture
    return K
end

"""
    hydraulic_head(z,ψ)

Return the hydraulic head.

The hydraulic head is defined as the sum of vertical height z and 
pressure head ψ; meters.
"""
hydraulic_head(z, ψ) = z + ψ

"""
    volumetric_liquid_fraction(
        ϑ_l::FT,
        eff_porosity::FT,
    ) where {FT}

Compute the volumetric liquid fraction from the effective porosity and the augmented liquid
fraction.
"""
function volumetric_liquid_fraction(ϑ_l::FT, eff_porosity::FT) where {FT}
    if ϑ_l < eff_porosity
        θ_l = ϑ_l
    else
        θ_l = eff_porosity
    end
    return θ_l
end


"""
    effective_saturation(
        porosity::FT,
        ϑ_l::FT,
        θ_r::FT,
    ) where {FT}

Compute the effective saturation of soil.

`ϑ_l` is defined to be larger than `θ_r`. If `ϑ_l-θ_r` is negative, 
hydraulic functions that take it as an argument will return 
imaginary numbers, resulting in domain errors. Exit in this 
case with an error.
"""
function effective_saturation(porosity::FT, ϑ_l::FT, θ_r::FT) where {FT}
    ϑ_l < θ_r && error("Effective saturation is negative")
    S_l = (ϑ_l - θ_r) / (porosity - θ_r)
    return S_l
end

"""
    pressure_head(
        model::AbstractHydraulicsModel{FT},
        ν::FT,
        S_s::FT,
        θ_r::FT,
        ϑ_l::FT,
        θ_i::FT,
    ) where {FT,PS}

Determine the pressure head in both saturated and unsaturated soil. 

If ice is present, it reduces the volume available for liquid water. 
The augmented liquid fraction changes behavior depending on if this 
volume is full of liquid water vs not. Therefore, the region of saturated
vs unsaturated soil depends on porosity - θ_i, not just on porosity.  
If the liquid water is unsaturated, the usual matric potential expression
is treated as unaffected by the presence of ice.
"""
function pressure_head(
    model::AbstractHydraulicsModel{FT},
    ν::FT,
    S_s::FT,
    θ_r::FT,
    ϑ_l::FT,
    θ_i::FT,
) where {FT}
    eff_porosity = ν - θ_i
    if ϑ_l < eff_porosity
        S_l = effective_saturation(ν, ϑ_l, θ_r)
        ψ = matric_potential(model, S_l)
    else
        ψ = (ϑ_l - eff_porosity) / S_s
    end
    return ψ
end


"""
    matric_potential(
            model::vanGenuchten{FT},
            S_l::FT
    ) where {FT}

Wrapper function which computes the van Genuchten function for matric potential.
"""
function matric_potential(model::vanGenuchten{FT}, S_l::FT) where {FT}
    @unpack n, m, α = model
    ψ_m = -((S_l^(-FT(1) / m) - FT(1)) * α^(-n))^(FT(1) / n)
    return ψ_m
end

"""
    matric_potential(
            model::Haverkamp{FT},
            S_l::FT
    ) where {FT}

Compute the van Genuchten function as a proxy for the Haverkamp model 
matric potential (for testing purposes).
"""
function matric_potential(model::Haverkamp{FT}, S_l::FT) where {FT}
    @unpack n, m, α = model
    ψ_m = -((S_l^(-FT(1) / m) - FT(1)) * α^(-n))^(FT(1) / n)
    return ψ_m
end

"""
    matric_potential(
            model::BrooksCorey{FT},
            S_l::FT
    ) where {FT}

Compute the Brooks and Corey function for matric potential.
"""
function matric_potential(model::BrooksCorey{FT}, S_l::FT) where {FT}
    @unpack ψb, m = model
    ψ_m = -ψb * S_l^(-m)
    return ψ_m
end



"""
    inverse_matric_potential(
        model::vanGenuchten{FT},
        ψ::FT
    ) where {FT}

Compute the effective saturation given the matric potential, using
the van Genuchten formulation.
"""
function inverse_matric_potential(model::vanGenuchten{FT}, ψ::FT) where {FT}
    ψ > 0 && error("Matric potential is positive")
    @unpack n, m, α = model
    S = (FT(1) + (α * abs(ψ))^n)^(-m)
    return S
end


"""
    inverse_matric_potential(
        model::Haverkamp{FT}
        ψ::FT
    ) where {FT}

Compute the effective saturation given the matric potential using the 
Haverkamp hydraulics model. This model uses the van Genuchten 
formulation for matric potential.
"""
function inverse_matric_potential(model::Haverkamp{FT}, ψ::FT) where {FT}
    ψ > 0 && error("Matric potential is positive")
    @unpack n, m, α = model
    S = (FT(1) + (α * abs(ψ))^n)^(-m)
    return S
end


"""
    inverse_matric_potential(
        model::BrooksCorey{FT}
        ψ::FT
    ) where {FT}

Compute the effective saturation given the matric potential using the 
Brooks and Corey formulation.
"""
function inverse_matric_potential(model::BrooksCorey{FT}, ψ::FT) where {FT}
    ψ > 0 && error("Matric potential is positive")
    @unpack ψb, m = model
    S = (-ψ / ψb)^(-FT(1) / m)
    return S
end

end #Module
