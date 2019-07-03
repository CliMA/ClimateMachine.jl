clear all
close all

icase = 'dycoms'

ug    =  7.0;
vg    = -5.0;
ztop  = 1500.0
nzmax =    100;
dataz = linspace(0, ztop, nzmax);
sounding = zeros(nzmax, 6);

gas_constant   = 8.3144598;
molmass_dryair = 28.97e-3;
%Coriolis_parameter = 
kappa_d = 2/7;
R_d     = gas_constant/molmass_dryair
cp_d    = R_d/kappa_d
cv_d    = cp_d - R_d
cp      = 1015
Lv      = 2.5008e6

gravity = 9.81
c       = cv_d/R_d
rvapor  = 461
levap   = 2.5e6
es0     = 6.1e2
c2      = R_d/cp_d
c1      = 1/c2
g       = gravity
pi0     = 1

theta0  = 292.5
p0      = 1.0178e5 %1.01325e5

zi      = 840.0
for k = 1:nzmax
	
	if ( dataz(k) <= zi)
		thetal(k) = 289.0;
		dataq(k)  = 9.0e-3;                  %kg/kg  specific humidity --> approx. to mixing ratio is ok
		rt(k)     = dataq(k)/(1.0 - dataq(k));  %total water mixing ratio
	else
		thetal(k) = 297.5 + (dataz(k) - zi)^(1/3);
		dataq(k)  = 1.5e-3;                  %kg/kg  specific humidity --> approx. to mixing ratio is ok
		rt(k)     = dataq(k)/(1.0 - dataq(k));  %total water mixing ratio
	end
	ql(k) = dataq(k);%  rt(k);%dataq(k);
end


%Calcuate theta from theta_l using the approximation formula:
%
%                    Lv
% thetal = theta - ------- rl
%                   cp_d
%
% From http://glossary.ametsoc.org/wiki/Liquid_water_potential_temperature
%
%      Betts, A. K., 1973: Non-precipitating cumulus convection and its parameterization. Quart. J. Roy. Met. Soc., 99, 178?196
%      Emanuel, K. A., 1994: Atmospheric Convection. Oxford Univ. Press, 580 pp.
%
theta = thetal; %+ ql.*Lv/cp_d;


% calculate the hydrostatically balanced exner potential and pressure
datapi(1) = 1.0;
datap(1)  = p0;
thetav(1) = theta(1)*(1.0 + 0.61*dataq(1));
for k = 2:nzmax
	thetav(k) = theta(k)*(1.0 + 0.61*dataq(k));
	
	datapi(k) = datapi(k-1) - gravity/(cp_d * 0.5*(thetav(k) + thetav(k-1))) * (dataz(k) - dataz(k-1));
 	
	%Pressure is computed only if it is NOT passed in the sounding file
	datap(k) = p0 * datapi(k)^(cp_d/R_d);
	
	%T = 289.0 * (datap(k)/p0)^(287.0/1015.0);
end



%Geostrophic wind:
%f = Coriolis_parameter;
%u_geostrophic = (-1.0/(f * rho) * dp/dy;
%v_geostrophic = 1/(f * rho) * dp/dx);


% 
% %Read sounding and calculate P if missing:
% 
sounding(:, 1) = dataz;
sounding(:, 2) = thetal;
sounding(:, 3) = dataq*1e+3;
sounding(:, 4) = ug;
sounding(:, 5) = vg;
sounding(:, 6) = datap;
[nrows, ncols] = size(sounding);

figure(1)
plot(thetal, dataz, '*r', theta, dataz, '--g')
xlabel('theta')
ylabel('z [m]')
title(' Thetal (red), theta (blue) [K]')
ylim([100 1200])

figure(2)
plot(datap, dataz)
xlabel('P [Pa]')
ylabel('z [m]')
ylim([100 1200])

figure(3)
plot(dataq, dataz)
xlabel('qt [g/kg]')
ylabel('z [m]')
ylim([100 1200])

%write to file
fileID = fopen('sounding_DYCOMS_TEST1.dat','w');
fprintf('Writing SOUNDING to file\n');
%fprintf(fileID, 'z  qt  theta_l  theta  P\n');
for i = 1:nzmax
	for j = 1:ncols
		fprintf(fileID,'%12.6f ', sounding(i, j));
		%fprintf(        '%12.6f ', sounding(i, j));
	end
	fprintf(fileID, '\n');
	%fprintf('\n');
end
fclose(fileID);



