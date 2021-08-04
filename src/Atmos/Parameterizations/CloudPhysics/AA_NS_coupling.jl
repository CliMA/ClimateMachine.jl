# Packages:
using Random
using SpecialFunctions

using Thermodynamics

using ClimateMachine.AerosolModel: mode, aerosol_model
using ClimateMachine.AerosolActivation

using CLIMAParameters
using CLIMAParameters: gas_constant
using CLIMAParameters.Planet: molmass_water, ρ_cloud_liq, grav, cp_d
using CLIMAParameters.Atmos.Microphysics

struct EarthParameterSet <: AbstractEarthParameterSet end
const EPS = EarthParameterSet
const param_set = EarthParameterSet()

# 2D mesh and time stepping parameters
nx = 41
ny = nx
dx = 1/(nx-1) # m 
dy = dx # m 
dz = dx
vol = dx^3
xmax = 1.0 # m 
ymax = xmax # m 
nt = 5000 # number of time steps
dt = 0.6 * dx # s 
nit = 200 
x = 0:dx:xmax
y = 0:dy:ymax

# Generate Grid Mesh 
MESH_particle_density = zeros(Float64, nx, ny)

# System parameters
T = 273.15 # K 
ν_air = 1.338*10^-5 # m^2/s
ρ_air = 1.292 # kg/m^3 

# Sea Salt parameters 
r_dry_accum = 0.243 * 1e-6 # μm
stdev_accum = 1.4          # -
mass_mix_ratio = 1.0
soluble_mass_frac = 1.0
osmotic_coeff = 0.9
molar_mass = 0.058443 
dissoc = 2.0
aerosol_density = 2170.
n_components = 1


# Particle Density distribution
μ_particle_density = 100. * 10^5
σ_particle_density =μ_particle_density/20 

# Sample distribution and populate grid space
for i in 1:nx
    for j in 1:ny
        MESH_particle_density[i, j] = randn()*σ_particle_density +μ_particle_density
    end
end


# Iterate with NS and determine 'clouds' formed 
u = zeros(Float64, ny, nx)
v = zeros(Float64, ny, nx)
b = zeros(Float64, ny, nx)
p = zeros(Float64, ny, nx)

for it in 1:nt+1
    for i in 2:nx-1
        for j in 2:ny-1
            b[i,j] .= ρ_air * (  (1/dt)*(((u[j, i+1]-u[j,i-1])/(2*dx))+((v[j+1,i]-v[j-1,i])/(2*dy)))  +  
                     ((u[j,i+1]-u[j,i-1])/(2*dx))^2  -2*((u[j+1,i]-u[j-1,i])/(2*dy))*
                     ((v[j, i+1]-v[j,i-1])/(2*dx))- ((v[j+1,i]-v[j-1,i])/(2*dy))^2 )
        end
    end

    for iit in 1:nit+1
        pd = p;
        for i in 2:nx-1
            for j in 2:ny-1
                p[j,i] .= (((pd[j,i+1]+pd[j,i-1])*(dy^2)) +  ((pd[j+1,i]+pd[j-1,i])*(dx^2))  - (b[j,i]*dx^2*dy^2)  )/(2*(dx^2+dy^2))
            end
        end
   
    end

    p[1,:]=p[2, :]
    p[:,nx]=p[:,nx-1]
    p[:, 1]=p[:, 2]
    p[ny,:]=p[ny-1,:]
    p[1,:] = 0.
    un = u
    vn = v

    for i in 2:nx-1
        for j in 2:ny-1
            # vn[j,i]*(dt/dy)*(un[j,i]-un[j-1,i])
            # vn[j,i]*(dt/dy)*(vn[j,i]-vn[j-1,i]) 
           u[j,i] .= un[j,i] - un[j,i]*((dt/dx)*(un[j,i]-un[j,i-1])) - vn[j,i]*(dt/dy)*(un[j,i]-un[j-1,i]) - (dt/(2*ρ_air*dx))*(p[j,i+1]-p[j,i-1]) + ν_air*(dt/(dx^2))*(un[j,i+1]-2*un[j,i]+un[j,i-1]) + ν_air * (dt/(dy^2))*(un[j+1,i]-2*un[j,i]+un[j-1,i])   
           v[j,i] .= vn[j,i] - un[j,i]*((dt/dx)*(vn[j,i]-vn[j,i-1])) - vn[j,i]*(dt/dy)*(vn[j,i]-vn[j-1,i]) - (dt/(2*ρ_air*dy))*(p[j+1,i]-p[j-1,i]) + ν_air*(dt/(dx^2))*(vn[j,i+1]-2*vn[j,i]+vn[j,i-1]) + ν_air * (dt/(dy^2))*(vn[j+1,i]-2*vn[j,i]+vn[j-1,i])   
        end
    end
    u[1,:] = 0.
    u[:,nx] = 0.
    u[:,1] = 0.
    u[ny,:] = 1.
    v[1,:] = 0.
    v[:,nx] = 0.
    v[:,1] = 0.
    v[ny,:] = 0.    
end

# Determine aerosols activated in final configuration
num_activated_particles = zeros(Float64, ny, nx)
for i in 1:nx
    for j in 1:ny
        updft_velo = v[j,i]
        PRESS = p[j,i]
        N = MESH_particle_density[j,i]
        modeij = mode(r_dry_accum, stdev_accum, N, (mass_mix_ratio,),
                      (soluble_mass_frac, ), (osmotic_coeff, ), 
                      (molar_mass, ), (dissoc, ), 1)
        modelij = aerosol_model((modeij, ))
        parts_act = total_N_activated(param_set, modelij, T, PRESS, updft_velo)  
        num_activated_particles[j,i] = parts_act*vol 
    end
end

println(num_activated_particles)
println(typeof(num_activated_particles))
println(size(num_activated_particles))