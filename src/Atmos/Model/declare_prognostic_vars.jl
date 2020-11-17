##### Prognostic variable

export Mass, Momentum, Energy
export Moisture, TotalMoisture, LiquidMoisture, IceMoisture
export Tracers

struct Mass <: PrognosticVariable end
struct Momentum <: PrognosticVariable end
struct Energy <: PrognosticVariable end

abstract type Moisture <: PrognosticVariable end
struct TotalMoisture <: Moisture end
struct LiquidMoisture <: Moisture end
struct IceMoisture <: Moisture end

struct Tracers <: PrognosticVariable end
