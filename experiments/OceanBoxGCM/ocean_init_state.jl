#
# 1. Custom initial condition setup
#    - ClimateMachine.HydrostaticBoussinesq.ocean_init_state!
#

# Setup types and globals we need
using StaticArrays
og=ClimateMachine.HydrostaticBoussinesq.OceanGyre

"""
    ocean_init_state!(::og)

initialize u,v,η with 0 and θ linearly distributed between 9 at z=0 and 1 at z=H

# Arguments
- `p`: OceanGyre problem object, used to dispatch on and obtain ocean height H
- `Q`: state vector
- `A`: auxiliary state vector, not used
- `coords`: the coordidinates
- `t`: time to evaluate at, not used
"""
function ClimateMachine.HydrostaticBoussinesq.ocean_init_state!(p::OceanGyre, Q, A, coords, t)
    @inbounds y = coords[2]
    @inbounds z = coords[3]
    @inbounds H = p.H

    Q.u = @SVector [0, 0]
    Q.η = 0

    # Wavy
    # Q.θ = (5 + 4 * cos(y * π / p.Lʸ)) * (1 + z / H)

    # Constant everywhere
    # Q.θ = 9

    # Big number
    # Q.θ = 50

    # Unstable column
    #
    # Define function with inversion near surface and stable stratification below
    #
    # $ \theta(z) = Ae^{\frac{-(z+L)}{L}} - Ce^{\frac{-D(z+L)}{L}} + B + Ez$
    #
    H=4000.;L=H/10.;A=20.;C=50.;D=2.5;B=8.;E=5.e-4;
    ft(xx,L)=exp(-xx/L);
    th1(zz)=A*ft.(-zz .+ L, L);
    th2(zz)=C*ft(D*(-zz .+ L), L);
    phi1=th1.(z)
    phi2=th2.(z)
    Q.θ = phi1 .- phi2 .+ B .+ E .* z;


    return nothing
end
