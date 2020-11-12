##### Multi-physics types

export Subsidence
struct Subsidence{PV, FT} <: TendencyDef{Source, PV}
    D::FT
end

# Subsidence includes tendencies in Mass, Energy and TotalMoisture equations:
Subsidence(D::FT) where {FT} = (
    Subsidence{Mass, FT}(D),
    Subsidence{Energy, FT}(D),
    Subsidence{TotalMoisture, FT}(D),
)

subsidence_velocity(subsidence::Subsidence{PV, FT}, z::FT) where {PV, FT} =
    -subsidence.D * z

struct PressureGradient{PV <: Momentum} <: TendencyDef{Flux{FirstOrder}, PV} end
struct Pressure{PV <: Energy} <: TendencyDef{Flux{FirstOrder}, PV} end

struct Advect{PV} <: TendencyDef{Flux{FirstOrder}, PV} end
