##### Prognostic variable

export Mass, Momentum, Energy, ρθ_liq_ice
export Moisture, TotalMoisture, LiquidMoisture, IceMoisture
export Precipitation, Rain, Snow
export Tracers

struct Mass <: PrognosticVariable end
struct Momentum <: PrognosticVariable end

abstract type AbstractEnergy <: PrognosticVariable end
struct Energy <: AbstractEnergy end
struct ρθ_liq_ice <: AbstractEnergy end

abstract type Moisture <: PrognosticVariable end
struct TotalMoisture <: Moisture end
struct LiquidMoisture <: Moisture end
struct IceMoisture <: Moisture end

abstract type Precipitation <: PrognosticVariable end
struct Rain <: Precipitation end
struct Snow <: Precipitation end

struct Tracers{N} <: PrognosticVariable end
