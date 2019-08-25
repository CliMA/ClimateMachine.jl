abstract type Geometry end
struct Box2D <: Geometry end
struct Box3D <: Geometry end
struct Sphere <: Geometry end

abstract type Moisture end
struct Dry <: Moisture end
struct EquilMoist <: Moisture end
struct NonEquilMoist <: Moisture end

abstract type TurbConv end
struct NoTurbConv <: TurbConv end
struct TurbConvModel <: TurbConv end

abstract type Microphysics end
struct NoMicrophysics <: Microphysics end
struct MicrophysicsModel <: Microphysics end

abstract type Radiation end
struct NoRadiation <: Radiation end
struct TwoStream <: Radiation end

"""
    AtmosCase{Geometry, Moisture, Radiation, TurbConv, Microphysics}

Atmospheric model types for specifying custom and
pre-set benchmark configurations.
"""
abstract type AtmosCase{Geometry, Moisture, Radiation, TurbConv, Microphysics} end

struct IsentropicVortex{G<:Union{Box2D,Box3D}, M, R, TC, MP}                   <: AtmosCase{G, M, R, TC, MP} end
struct MethodOfManufacturedSolutions{G<:Union{Box2D,Box3D}, M<:Dry, R, TC, MP} <: AtmosCase{G, M, R, TC, MP} end
struct RisingThermalBubble{G, M, R, TC, MP}                                    <: AtmosCase{G, M, R, TC, MP} end
struct Dycoms{G<:Union{Box2D,Box3D}, M, R, TC, MP}                             <: AtmosCase{G, M, R, TC, MP} end
struct RayleighBernard{G<:Union{Box2D,Box3D}, M, R, TC, MP}                    <: AtmosCase{G, M, R, TC, MP} end
struct Advection{G<:Sphere, M, R, TC, MP}                                      <: AtmosCase{G, M, R, TC, MP} end
struct HeldsSuarez{G<:Sphere, M, R, TC, MP}                                    <: AtmosCase{G, M, R, TC, MP} end
struct AquaPlanet{G<:Sphere, M, R, TC, MP}                                     <: AtmosCase{G, M, R, TC, MP} end
