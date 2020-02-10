#### Subsidence model
export AbstractSubsidence,
       NoSubsidence,
       ConstantSubsidence,
       subsidence_velocity

abstract type AbstractSubsidence{FT<:AbstractFloat} end

struct NoSubsidence{FT} <: AbstractSubsidence{FT} end

struct ConstantSubsidence{FT} <: AbstractSubsidence{FT}
  D::U(FT,:frequency)
end

subsidence_velocity(m::NoSubsidence{FT}, z::U(FT,:space)) where {FT} = FT(0) * get_unit(m, :velocity)
subsidence_velocity(subsidence::ConstantSubsidence{FT}, z::U(FT,:space)) where {FT} = subsidence.D*z
