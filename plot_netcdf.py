import netCDF4 as nc
import pylab as plt
import argparse
import numpy as np
# command line:
# python clima_edmf_plots.py
def main():
    # p_ncfile = '/Users/yaircohen/Documents/codes/scampy/tests/les_data/Bomex.nc'
    s_ncfile = 'output/SBL_EDMF_ANELASTIC_1D_AtmosLESDefault.nc'
    w_levels = [-1, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]
    mf_levels = [0.001, 0.002, 0.005, 0.007, 0.01, 0.025, 0.05, 0.1]
    ql_levels = [0.0001, 0.001, 0.002, 0.005, 0.007, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]
    cf_levels = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]
    entr_levels = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    ml_levels = [0,40,80,120,160,200,240,280,320]
    with nc.Dataset(s_ncfile, 'r') as s_data:
        print(s_data.variables)
        # p_data = nc.Dataset(p_ncfile, 'r')
        # z_s = np.divide(s_data.variables['z'],1000.0)
        # t_s = np.divide(s_data.variables['time'],3600.0)
        # ρ = np.divide(s_data.variables['ρ'],3600.0)
        # ρu = np.divide(s_data.variables['ρu[1]'],3600.0)
        # ρv = np.divide(s_data.variables['ρu[2]'],3600.0)
        # ρw = np.divide(s_data.variables['ρu[3]'],3600.0)
        # upd_ρaw = s_data.variables['turbconv.updraft[1].ρaw'][:].data
        # upd_ρa = s_data.variables['turbconv.updraft[1].ρa'][:].data
        # env_ρatke = s_data.variables['turbconv.environment.ρatke'][:].data
        # ρe = s_data.variables['energy.ρe'][:].data
        # ρq_tot = s_data.variables['moisture.ρq_tot'][:].data
        # env_ρaθ_liq_cv = s_data.variables['turbconv.environment.ρaθ_liq_cv'][:].data
        # env_ρaq_tot_cv = s_data.variables['turbconv.environment.ρaq_tot_cv'][:].data
        # env_ρaθ_liq_q_tot_cv = s_data.variables['turbconv.environment.ρaθ_liq_q_tot_cv'][:].data
        # upd_ρaθ_liq = s_data.variables['turbconv.updraft[1].ρaθ_liq'][:].data
        # upd_ρaq_tot = s_data.variables['turbconv.updraft[1].ρaq_tot'][:].data
        
        z_s = s_data.variables['z']
        t_s = s_data.variables['time']
        rho = s_data.variables['rho']
        u = s_data.variables['u']
        v = s_data.variables['v']
        w = s_data.variables['w']
        th = s_data.variables['thd']
        T = s_data.variables['temp']
        et = s_data.variables['et']
        ei = s_data.variables['ei']
        cov_w_u = s_data.variables['cov_w_u']
        cov_w_thd = s_data.variables['cov_w_thd']
        # upd_ρaw = s_data.variables['turbconv.updraft[1].ρaw'][:].data
        # upd_ρa = s_data.variables['turbconv.updraft[1].ρa'][:].data
        # env_ρatke = s_data.variables['turbconv.environment.ρatke'][:].data
        # ρe = s_data.variables['energy.ρe'][:].data
        # ρq_tot = s_data.variables['moisture.ρq_tot'][:].data
        # env_ρaθ_liq_cv = s_data.variables['turbconv.environment.ρaθ_liq_cv'][:].data
        # env_ρaq_tot_cv = s_data.variables['turbconv.environment.ρaq_tot_cv'][:].data
        # env_ρaθ_liq_q_tot_cv = s_data.variables['turbconv.environment.ρaθ_liq_q_tot_cv'][:].data
        # upd_ρaθ_liq = s_data.variables['turbconv.updraft[1].ρaθ_liq'][:].data
        # upd_ρaq_tot = s_data.variables['turbconv.updraft[1].ρaq_tot'][:].data
        
        print(np.shape(z_s))
        print(np.shape(t_s))
        print(np.shape(th))
        snapshots = np.shape(t_s)[0]
        snaps_ = np.linspace(0, snapshots-1, 5).astype(int)
        for timestamp in snaps_:
            # fig = plt.figure('fluxes')
            plt.figure('theta')
            # plt.plot(th[0, :],z_s, label='init')
            plt.plot(th[timestamp,:],z_s, label=str(timestamp)+' min')
            plt.xlabel('theta (K)')
            plt.ylabel('z [m]')
            plt.legend(frameon=False)
            plt.show()
            
            plt.figure('temperature')
            # plt.plot(T[0, :],z_s, label='init')
            plt.plot(T[timestamp,:],z_s, label=str(timestamp)+' min')
            plt.xlabel('T (K)')
            plt.ylabel('z [m]')
            plt.legend(frameon=False)
            plt.show()
            
            plt.figure('horizontal velocity 1')
            # plt.plot(u[0, :],z_s, label='init')
            plt.plot(u[timestamp,:],z_s, label=str(timestamp)+' min')
            plt.xlabel('u (m/s)')
            plt.ylabel('z [m]')
            plt.legend(frameon=False)
            plt.show()
            
            plt.figure('horizontal velocity 2')
            # plt.plot(v[0, :],z_s, label='init')
            plt.plot(v[timestamp,:],z_s, label=str(timestamp)+' min')
            plt.xlabel('v (m/s)')
            plt.ylabel('z [m]')
            plt.legend(frameon=False)
            plt.show()
            
            plt.figure('vertical velocity')
            # plt.plot(w[0, :],z_s, label='init')
            plt.plot(w[timestamp,:],z_s, label=str(timestamp)+' min')
            plt.xlabel('w (m/s)')
            plt.ylabel('z [m]')
            plt.legend(frameon=False)
            plt.show()
            
            plt.figure('total specific energy')
            # plt.plot(et[0, :],z_s, label='init')
            plt.plot(et[timestamp,:],z_s, label=str(timestamp)+' min')
            plt.xlabel('et (J kg^{-1})')
            plt.ylabel('z [m]')
            plt.legend(frameon=False)
            plt.show()
            
            plt.figure('internal specific energy')
            # plt.plot(ei[0, :],z_s, label='init')
            plt.plot(ei[timestamp,:],z_s, label=str(timestamp)+' min')
            plt.xlabel('ei (J kg^{-1})')
            plt.ylabel('z [m]')
            plt.legend(frameon=False)
            plt.show()
        
        # plt.figure('heat flux')
        # plt.plot(cov_w_thd[0, :],z_s, label='init')
        # plt.plot(cov_w_thd[timestamp,:],z_s, label='final')
        # plt.xlabel('cov_w_thd')
        # plt.ylabel('z [m]')
        # plt.legend(frameon=False)
        # plt.show()
        
        # plt.figure('horizontal momentum flux')
        # plt.plot(cov_w_u[0, :],z_s, label='init')
        # plt.plot(cov_w_u[timestamp,:],z_s, label='final')
        # plt.xlabel('cov_w_u')
        # plt.ylabel('z [m]')
        # plt.legend(frameon=False)
        # plt.show()
    
main()