# # Hydraulic functions

# This tutorial shows how to specify the hydraulic functions
# used in Richard's equation. In particular,
# we show how to choose the formalism for matric potential and hydraulic
# conductivity, and how to make the hydraulic conductivity account for
# the presence of ice as well as the temperature dependence of the
# viscosity of liquid water.

# # Multiple dispatch
# Before diving into the hydraulics, let's go over a feature of Julia called
# multiple dispatch. This will make things simpler moving forwards!

# In the Climate Machine code, we make use of multiple dispatch.
# With multiple dispatch, a function can have many
# ways of executing (called methods), depending on the *type* of the
# variables passed in. A simple example of multiple dispatch is the division operation.
# Integer division takes two numbers as input, and returns an integer - ignoring the decimal.
# Float division takes two numbers as input, and returns a floating point number, including the decimal.
# In Julia, we might write these as:
# ```
#    function division(a::Int, b::Int)
#         return floor(Int, a/b)
#     end
#
#    function division(a::Float64, b::Float64)
#         return a/b
#     end
#  ```
# We can see that `division` is now a function with two methods.
# ```
#     julia> division
#     division (generic function with 2 methods)
# ```
# Now, using the same function signature, we can carry out integer
# division or floating point division, depending on the types of the
# arguments:
# ```
#     julia> division(1,2)
#     0
#
#     julia> division(1.0,2.0)
#     0.5
# ```


# One benefit of this is that a function like `matric_potential`, which
# should return different answers depending on what type of hydraulics
# model (e.g. van Genchten, or Brooks and Corey) the user chooses,
# is always called in the same way. If not, we would need to have a
# `matric_potential_van_genuchten` function, and a `matric_potential_brooks_and_corey`
# function, for example, and we would need to change the source code
# whenever the user wished to changed hydraulics model. Or, we could
# have a single function with a branch in it, (the choice made based on a flag passed in),
# but this slows things down on the GPU.
# Multiple dispatch provides a nice solution.

# # Preliminary setup

# - load external packages

using Plots

# - load ClimateMachine modules

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations

FT = Float32

# # Choose general soil parameters for your soil type.
# For the water equation, the user needs to choose a porosity `ν`,
# saturated hydraulic conductivity `Ksat` (and a specific storage `S_s`,
# though that is not needed here).
# Note that the two hydraulic models included in ClimateMachine -
# that of Brooks and Corey (1964, 1977) and van Genuchten (1980) - use the same
# value of `Ksat`. All values are given in SI/mks units. We
# neglect the residual pore space in the ClimateMachine model. Below
# we choose parameters for sandy loam (Bonan, 2019, Chapter 8, page 120).
ν = FT(0.41)
Ksat = FT(4.42 / (3600 * 100))

# # Matric Potential
# The matric potential represents how much water clings to soil. Drier soil
# holds onto water more tightly, making diffusion more difficult. As soil becomes
# wetter, the matric potential decreases in magnitude, making diffusion easier.
# ClimateMachine's Land model allows for two `hydraulics` models for matric potential
# (and hydraulic conductivity), that of van Genuchten (1980), and Brooks and Corey (1967, 1970).

# The van Genucthen model requires two free parameters, `α` and `n`.
# The third parameter is computed from `n`. Of these, only `α` carries
# units, of inverse meters. The Brooks and Corey model also uses
# two free parameters, `ψ_b`, the matric potential at saturation,
#  and a constant `m`. `ψ_b` carries units of meters.

# Below we show how to create two concrete examples of these hydraulics models,
# for sandy loam. Importantly - the parameters chosen are a function of soil type!
vg_α = FT(7.5)
vg_n = FT(1.89)
hydraulics = vanGenuchten{FT}(α = vg_α, n = vg_n)

ψ_sat = -0.09
mval = 0.228
hydraulics_bc = BrooksCorey{FT}(ψb = ψ_sat, m = mval)

# As alluded to above, we use multiple dispatch.
# As a concrete example:
# `hydraulics` is of type vanGenuchten{Float32} based on our choice of FT:
# ```
#     julia> typeof(hydraulics)
#     vanGenuchten{Float32}
# ```
# but meanwhile,
# ```
#     julia> typeof(hydraulics_bc)
#     BrooksCorey{Float32}
# ```
# The function `matric_potential` will execute different methods
# depending on if we pass a hydraulics model of type vanGenuchten or
# e.g. BrooksCorey! In both cases, it will return the correct value
# for `ψ`.

# Let's plot the matric potential as a function of the effective saturation `S_l`.

S_l = FT.(0.01:0.01:0.99)
ψ = matric_potential.(Ref(hydraulics), S_l)
ψ_bc = matric_potential.(Ref(hydraulics_bc), S_l)
plot(
    S_l,
    log10.(-ψ),
    xlabel = "effective saturation",
    ylabel = "Log10(|ψ|)",
    label = "van Genuchten",
)
plot!(S_l, log10.(-ψ_bc), label = "Brooks and Corey")
# The huge range in `ψ` as `S_l` varies, as well as the steep slope in
# `ψ` near saturation and completely dry soil, are part of the reason
# why Richard's equation is such a challenging numerical problem.


# # Hydraulic conductivity
# The hydraulic conductivity is a more complex function than the matric potential,
# as it depends on the temperature of the water, the volumetric ice fraction, and
# the volumetric liquid water fraction. It also depends on the `hydraulics` model
# chosen.

# We represent the hydraulic conductivity `K` as the product of four factors:
# `Ksat`, an `impedance_factor` (which accounts for the effect of ice)
# a `viscosity_factor` (which accounts for the effect of temperature)
# and a `moisture_factor` (which accounts for the effect of liquid water),
#  Let's start with ice and temperature independence, but moisture dependence.
# The `moisture_factor` will make use of the `hydraulics` model chosen above when we compute `K`. 

moisture_choice = MoistureDependent{FT}()
viscosity_choice = ConstantViscosity{FT}()
impedance_choice = NoImpedance{FT}()

# We are going to calculate `K = Ksat × 1 × 1 × moisture_factor`, but as alluded
# to above, our functions will use multiple dispatch. Just like we defined new type
# classes for vanGenuchten{FT} and BrooksCorey{FT}, we also created new type classes
# for the `impedance_factor`, `viscosity_factor`, and `moisture_factor`. 
# Based on the type choices the user makes for these, the correct conductivity 
# value will be returned. For example, in the source code we have defined a function
# called `viscosity_factor` with a method that, when passed `ConstantViscosity{FT}`, always
# returns 1. The same is true for a method of `impedance_factor`, using the type
# `NoImpedance{FT}`, and for `moisture_factor`, using the type `MoistureIndependent{FT}`.


# One byproduct of this flexibility in functions is that `hydraulic_conductivity`
# requires all the arguments it could possibly need passed to it, which is
# why here we must supply a value `T` and `θ_ice`, even though they are not used.
T = FT(0.0)
θ_ice = FT(0.0)

K =
    Ksat .*
    hydraulic_conductivity.(
        Ref(impedance_choice),
        Ref(viscosity_choice),
        Ref(moisture_choice),
        Ref(hydraulics),
        Ref(θ_ice),
        Ref(ν),
        Ref(T),
        S_l,
    )
plot(
    S_l,
    log10.(K),
    xlabel = "effective saturation",
    ylabel = "Log10(K)",
    label = "K",
)

# Let's see how the curves change when we include the effects of temperature
# and ice on the hydraulic conductivity.
viscosity_choice_T = TemperatureDependentViscosity{FT}()
T = FT(300.0)
K_T =
    Ksat .*
    hydraulic_conductivity.(
        Ref(impedance_choice),
        Ref(viscosity_choice_T),
        Ref(moisture_choice),
        Ref(hydraulics),
        Ref(θ_ice),
        Ref(ν),
        Ref(T),
        S_l,
    )
ice_impedance_I = IceImpedance{FT}()
θ_ice = FT(0.3)
S_l_accounting_for_ice = FT.(0.01:0.01:0.7) # note that the total volumetric water fraction cannot exceed unity.
K_ice =
    Ksat .*
    hydraulic_conductivity.(
        Ref(ice_impedance_I),
        Ref(viscosity_choice),
        Ref(moisture_choice),
        Ref(hydraulics),
        Ref(θ_ice),
        Ref(ν),
        Ref(T),
        S_l_accounting_for_ice,
    )
plot(
    S_l,
    log10.(K),
    xlabel = "effective saturation",
    ylabel = "Log10(K)",
    label = "Base case",
)
plot!(S_l, log10.(K_T), label = "Temperature Dependent Viscosity")
plot!(S_l_accounting_for_ice, log10.(K_ice), label = "Ice Impedance")

# We can also look and see how the Brooks and Corey moisture factor differs from the
# van Genuchten moisture factor by changing the `hydraulics` model passed:
T = FT(0.0)
θ_ice = FT(0.0)

K_bc =
    Ksat .*
    hydraulic_conductivity.(
        Ref(impedance_choice),
        Ref(viscosity_choice),
        Ref(moisture_choice),
        Ref(hydraulics_bc),
        Ref(θ_ice),
        Ref(ν),
        Ref(T),
        S_l,
    )
plot(
    S_l,
    log10.(K),
    xlabel = "effective saturation",
    ylabel = "Log10(K)",
    label = "van Genuchten",
)
plot!(
    S_l,
    log10.(K_bc),
    xlabel = "effective saturation",
    ylabel = "Log10(K)",
    label = "Brooks and Corey",
)

# # Other features
# The user also has the choice of making the conductivity constant by choosing
# `MoistureIndependent{FT}()`. This is useful for debugging!
no_moisture_dependence = MoistureIndependent{FT}()
K_constant =
    Ksat .*
    hydraulic_conductivity.(
        Ref(impedance_choice),
        Ref(viscosity_choice),
        Ref(no_moisture_dependence),
        Ref(hydraulics),
        Ref(θ_ice),
        Ref(ν),
        Ref(T),
        S_l,
    )
# ```
#     julia> unique(K_constant)
#     1-element Array{Float32,1}:
#      1.2277777f-5
# ```
# Note that choosing this option does not meant the matric potential
# is constant, as a hydraulics model is still required and employed.


# And, lastly, you might be wondering why we left `Ksat` out of the function
# for `hydraulic_conductivity`. It turns out it is also useful for debugging
# to be able to turn off the diffusion of water, by setting `K_sat = 0`.
