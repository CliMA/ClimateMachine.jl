import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np

def add_plots(axes, j, datadir, esdg):
    if esdg:
        mesh = pv.read('/'.join([datadir, 'baroclinicwave_mpirank0000_step0011.vtu']))
    else:
        mesh = pv.read('/'.join([datadir, 'BaroclinicWave_mpirank0000_num0012.vtu']))
        
    rho = mesh['ρ']
    rhoE = mesh['ρe']
    if esdg:
        phi = mesh['Φ']
    else:
        phi = mesh['orientation.Φ']

    rhoU = mesh['ρu[1]']
    rhoV = mesh['ρu[2]']
    rhoW = mesh['ρu[3]']


    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    z = mesh.points[:, 2]

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    rlat = np.arcsin(z / r)
    rlon = np.arctan2(y, x)
    lat = 180 / np.pi * np.arcsin(z / r)
    lon = 180 / np.pi * np.arctan2(y, x)
    lonshift = 60
    lon = np.mod(lon + 180 - lonshift, 360) - 180
    a = np.min(r)
    r = r - a

    print(np.min(r))
    print(np.max(r))

#print(r[r < 1000])
#mask = np.abs(r - 560) < 10
    mask = np.abs(r) < 20

    rho_s = rho[mask]
    rhoE_s = rhoE[mask]
    rhoU_s = rhoU[mask]
    rhoV_s = rhoV[mask]
    rhoW_s = rhoW[mask]
    phi_s = phi[mask]

    lat_s = lat[mask]
    lon_s = lon[mask] 

    rlat_s = rlat[mask]
    rlon_s = rlon[mask] 

    g = 7.0 / 5.0
    cv_d = 717.5060234725578
    T_0 = 273.16

    if esdg:
        p_s = (g - 1) * (rhoE_s -\
              (rhoU_s ** 2 + rhoV_s ** 2 + rhoW_s ** 2) / (2 * rho_s) - rho_s * phi_s)
    else:
        p_s = (g - 1) * (rhoE_s -\
              (rhoU_s ** 2 + rhoV_s ** 2 + rhoW_s ** 2) / (2 * rho_s) - rho_s * (phi_s - cv_d * T_0))

    p_s = p_s / 100

    u_s = -np.sin(rlon_s) * rhoU_s + np.cos(rlon_s) * rhoV_s
    u_s = u_s / rho_s

    v_s = -np.sin(rlat_s) * np.cos(rlon_s) * rhoU_s -\
          np.sin(rlat_s) * np.sin(rlon_s) * rhoV_s +\
          np.cos(rlat_s) * rhoW_s
    v_s = v_s / rho_s

    w_s = np.cos(rlat_s) * np.cos(rlon_s) * rhoU_s +\
          np.cos(rlat_s) * np.sin(rlon_s) * rhoV_s +\
          np.sin(rlat_s) * rhoW_s
    w_s = w_s / rho_s

    print("P")
    print(np.min(p_s))
    print(np.max(p_s))

    print("U")
    print(np.min(u_s))
    print(np.max(u_s))

    print("V")
    print(np.min(v_s))
    print(np.max(v_s))

    print("W")
    print(np.min(w_s))
    print(np.max(w_s))

    print("Rho")
    print(np.min(rho_s))
    print(np.max(rho_s))

    print(np.min(lon))
    print(np.max(lon))

    mask = (lat_s > 0) & (lon_s > -lonshift)

    titlefs = 22

# pressure
    if esdg:
        title = "Entropy Stable"
    else:
        title = "Master" 
        
    levels = 16
    axes[0, j].set_title("{}\n\nP, min = {:.2f}, max = {:.2f}".format(title, np.min(p_s), np.max(p_s)), fontsize=titlefs)
    axes[0, j].tricontour(lon_s[mask] + lonshift, lat_s[mask], p_s[mask], levels=levels, colors='k', linewidths=1)
    axes[0, j].tricontourf(lon_s[mask] + lonshift, lat_s[mask], p_s[mask], levels=levels)

# u
    levels = np.arange(-15, 22, 4)
    axes[1, j].set_title("U, min = {:.2f}, max = {:.2f}".format(np.min(u_s), np.max(u_s)), fontsize=titlefs)
    axes[1, j].tricontour(lon_s[mask] + lonshift, lat_s[mask], u_s[mask], levels=levels, colors='k', linewidths=1)
    axes[1, j].tricontourf(lon_s[mask] + lonshift, lat_s[mask], u_s[mask], levels=levels)

# v
    levels = np.arange(-28, 36, 6)
    axes[2, j].set_title("V, min = {:.2f}, max = {:.2f}".format(np.min(v_s), np.max(v_s)), fontsize=titlefs)
    axes[2, j].tricontour(lon_s[mask] + lonshift, lat_s[mask], v_s[mask], levels=levels, colors='k', linewidths=1)
    axes[2, j].tricontourf(lon_s[mask] + lonshift, lat_s[mask], v_s[mask], levels=levels)

# w
#levels = [-0.14, -0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
    levels = np.arange(-0.26, 0.26, 0.04)
    axes[3, j].set_title("W, min = {:.2f}, max = {:.2f}".format(np.min(w_s), np.max(w_s)), fontsize=titlefs)
    axes[3, j].tricontour(lon_s[mask] + lonshift, lat_s[mask], w_s[mask], levels=levels, colors='k', linewidths=1)
    axes[3, j].tricontourf(lon_s[mask] + lonshift, lat_s[mask], w_s[mask], levels=levels)


fig, axes = plt.subplots(4, 2)

#datadir = '/home/mwarusz/data/esdg/vtk_master_baroclinic_poly3_horz12_vert6_CuArray_Float64/'
#datadir = '/home/mwarusz/data/esdg/vtk_master_baroclinic_poly4_horz12_vert6_CuArray_Float64/'

#datadir = '/home/mwarusz/data/esdg/vtk_esdg_baroclinic_poly3_horz12_vert6_CuArray_Float64/'
#datadir = '/home/mwarusz/data/esdg/vtk_esdg_baroclinic_poly7_horz8_vert3_CuArray_Float64/'
#datadir = '/home/mwarusz/data/esdg/vtk_esdg_baroclinic_poly7_horz10_vert5_CuArray_Float64/'
#datadir = '/home/mwarusz/data/esdg/vtk_esdg_baroclinic_poly3_horz30_vert8_CuArray_Float64/'

datadir = '/home/mwarusz/data/esdg/vtk_esdg_baroclinic_poly4_horz12_vert6_CuArray_Float64_Matrix/'
add_plots(axes, 0, datadir, True)

#datadir = '/home/mwarusz/data/esdg/vtk_esdg_baroclinic_poly4_horz12_vert8_CuArray_Float64/'
#add_plots(axes, 1, datadir, True)

datadir = '/home/mwarusz/data/esdg/vtk_master_baroclinic_poly4_horz12_vert6_CuArray_Float64_Roe/'
add_plots(axes, 1, datadir, False)


for ax in axes:
    for a in ax:
        a.set_xticks([0, 60, 120, 180, 240])
        a.set_yticks([0, 30, 60, 90])

#fig = plt.gcf()
#fig.set_size_inches(13.3, 20)
fig.set_size_inches(27, 20)
#plt.savefig('bw_esdg_day7_poly4_12h_6v_matrix.pdf')
plt.savefig('bw_comp_day7_poly4.pdf', bbox_inches='tight')
