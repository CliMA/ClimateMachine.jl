using ClimateMachine
using CLIMAParameters
using ClimateMachine.SurfaceFluxes

using NCDatasets

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

data = NCDataset("/home/asridhar/CLIMA/bomex-f64/BOMEX_AtmosLESDefault_2020-05-31T11.52.05.244_num0055.nc")
FT = Float64;

# Data at first interior node (x_ave)
qt_ave = data["qt"][:][2];
z_ave = data["z"][:][2];
thv_ave = data["thv"][:][2];
u_ave = data["u"][:][2];
x_ave = [u_ave, thv_ave, qt_ave];

# Initial guesses for MO parameters
LMO_init = FT(100);
u_star_init = FT(0.3);
th_star_init = FT(290);
qt_star_init = FT(1e-5);
x_init = [LMO_init, u_star_init, th_star_init, qt_star_init];

# Surface values for variables
u_sfc = FT(0)
thv_sfc = data["thv"][:][1];
qt_sfc = data["qt"][:][1];
z_sfc = data["z"][:][1];
x_s = [u_sfc, thv_sfc, qt_sfc];

# Dimensionless numbers 
dimless_num = [FT(1), FT(1/3), FT(1/3)];

# Roughness
z0 = [FT(0.001), FT(0.0001), FT(0.0001)]; 

# Constants 
a  = FT(4.7)
Δz = data["z"][2];

# F_exchange
F_exchange = [FT(0), FT(0), FT(0)];

result = surface_conditions(
            param_set,
            x_init,
            x_ave,
            x_s, 
            z0,
            F_exchange,
            dimless_num,
            thv_ave,
            qt_ave,
            Δz,
            z_ave / 2, 
            a,
            nothing
         );

