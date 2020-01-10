#### Subsidence model
export AbstractSubsidence,
       NoSubsidence,
       ConstantSubsidence,
       subsidence_velocity

abstract type AbstractSubsidence{FT<:AbstractFloat} end

struct NoSubsidence{FT} <: AbstractSubsidence{FT} end

struct ConstantSubsidence{FT} <: AbstractSubsidence{FT}
  D::FT
end

subsidence_velocity(::NoSubsidence{FT}, z::FT) where {FT} = FT(0)
subsidence_velocity(subsidence::ConstantSubsidence{FT}, z::FT) where {FT} = subsidence.D*z
