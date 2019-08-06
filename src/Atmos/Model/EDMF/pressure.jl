#### Pressure Model
abstract type AbstractPressureModel{T} end

export PressureModel

struct PressureModel{T} <: AbstractPressureModel{T} end
PressureModel(::Type{T}) where T = PressureModel{T}()

