##### Prognostic variable

export Mass, Momentum, Energy, ρθ_liq_ice
export AbstractMoisture, TotalMoisture, LiquidMoisture, IceMoisture
export AbstractPrecipitation, Rain, Snow
export Tracers

struct Mass <: PrognosticVariable end
struct Momentum <: AbstractMomentum end

struct Energy <: AbstractEnergy end
struct ρθ_liq_ice <: AbstractEnergy end

struct TotalMoisture <: AbstractMoisture end
struct LiquidMoisture <: AbstractMoisture end
struct IceMoisture <: AbstractMoisture end

struct Rain <: AbstractPrecipitation end
struct Snow <: AbstractPrecipitation end

struct Tracers{N} <: AbstractTracers{N} end
