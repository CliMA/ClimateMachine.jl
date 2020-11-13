##### Prognostic variable

export Mass, Momentum, Energy
export TotalMoisture, LiquidMoisture, IceMoisture
export Tracers

struct Mass <: PrognosticVariable end
struct Momentum <: PrognosticVariable end
struct Energy <: PrognosticVariable end
struct TotalMoisture <: PrognosticVariable end
struct LiquidMoisture <: PrognosticVariable end
struct IceMoisture <: PrognosticVariable end
struct Tracers <: PrognosticVariable end
