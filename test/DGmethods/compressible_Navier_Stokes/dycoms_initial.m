%
%
% Initialize Dycoms: 
% see details in CLIMA-doc
%
%
clear all
close all
addpath /Users/simone/Work/Matlab-functions/export_fig/
ifig=1;
write2file = 10;
%

gas_constant   = 8.3144598;
molmass_dryair = 28.97e-3;

R_d     = 287.0
R_v     = 461.5
cp_d    = 1015.0
cv_d    =  717.0
cp_v    = 1859.0
cp_l    = 4181.0
Lv0     = 2.47e6
epsdv   = 1.61
g       = 9.81

p_sfc     = 1.0178e5
rho_sfc   = 1.22;
pi0       =    1;
q_tot_sfc = 9.0e-3;
Rm_sfc    = R_d*(1 + (epsdv - 1)*q_tot_sfc);
T_sfc     = p_sfc/(rho_sfc*Rm_sfc);
T_0       = 285.0; %average b.l. temperature

u    =  7.0;
v    = -5.0;
ztop  = 1500.0

nop = 4;
nzmax    = 2*(150*nop+1);
zh       = linspace(0, ztop, nzmax);
zz       = linspace(0, 300, nzmax);
r_tot    = 0.0*zh;
q_tot    = 0.0*r_tot;
sounding = zeros(nzmax, 6);

%Define q_liq:
zb = 600.0 %cloud bottom
zi = 840.0 %cloud top
q_liq_peak = 0.00045;
dz_cloud = zi - zb;
for k = 1:length(zh)
	
	q_liq(k) =0;
	if zh(k) > zb && zh(k) <= zi	
		q_liq(k) = (zh(k) - zb)*q_liq_peak/dz_cloud;
	end
end

for k = 1:length(zh)

	if ( zh(k) <= zi)
		theta_liq(k)  = 289.0;
		r_tot(k)      = 9.0e-3;                  %kg/kg  specific humidity --> approx. to mixing ratio is ok
		q_tot(k)      = r_tot(k)/(1.0 - r_tot(k));  %total water mixing ratio
	else
		theta_liq(k) = 297.5 + (zh(k) - zi)^(1/3);
		r_tot(k)     = 1.5e-3;                  %kg/kg  specific humidity --> approx. to mixing ratio is ok
		q_tot(k)     = r_tot(k)/(1.0 - r_tot(k));  %total water mixing ratio
	end
end

%Derive T from theta_liq (find roots of non-linear equation p.546 of Stull):
for k = 1:length(zh)
	
	z   = zh(k);
	thl = theta_liq(k);
	rl  = q_liq(k);
	
	%Pressure (what follows is only valid in absence of ice):
	%R_m, c_p
	Rm  = R_d*(1 + (epsdv - 1)*q_tot(k) - epsdv*q_liq(k));
	cpm = cp_d + (cp_v - cp_d)*q_tot(k) + (cp_l - cp_v)*q_liq(k);
	 
	
	
%	[pd, rhod]   = calculate_dry_pressure(z, theta);
	
%	fun = @(x) theta_liq_to_T(x, thl, rl, z); % x is zero that I am looking for: T
%	T = fzero(fun, 300.0);
%	
%	verify = thl - (T + g*z/cp_d) + rl*(Lv0 * (T + g*z/cp_d) /(cp_d*T));
%	
%	if(abs(verify) <= 1e-12)
%		Tz(k) = T;
%	end	
%	
%	%Tv:
%	r_vap(k) = q_tot(k) - q_liq(k);
%	Tv(k) = T*(1 + (epsdv - 1)*q_tot(k) - epsdv*q_liq(k)) ;
%	
%	%theta, thetav:
%	theta(k)  = Tz(k) + g*zh(k)/cp_d;
%	thetav(k) = theta(k)*(1 + (epsdv - 1)*q_tot(k) - epsdv*q_liq(k));
%
%	pd(k)   = p_sfc*(1 - g*zh(k)/(cp_d * theta_liq(1)));
%	p(k)   = pd(k) * (1 + (q_tot(k) - q_liq(k))*R_v/R_d);
%	p1(k)   = (1 + (epsdv - 1)*q_tot(k) - epsdv*q_liq(k))*pd(k);
%	rhod(k) = (p_sfc/(Rm * theta_liq(k)))*(pd(k)/p_sfc)^((cp(k)/Rm - 1));

    %From Tapio in clima-doc
	%Pressure
	H = Rm_sfc * T_0 / g;
	p(k) = p_sfc * exp(-zh(k)/H);
	
	%Exner
	exner = (p(k)/p_sfc)^(R_d/cp_d);
	
	%T, Tv 
	T(k)     = exner*theta_liq(k) + Lv0*q_liq(k)/(cpm*exner);
	Tv(k)    = T(k)*(1 + (epsdv - 1)*q_tot(k) - epsdv*q_liq(k));
	
	%Density
	rho(k)   = p(k)/(Rm*T(k));
	
	%Theta, Thetav
	theta(k)  = T(k)/exner;
	thetav(k) = theta(k)*(1 + (epsdv - 1)*q_tot(k) - epsdv*q_liq(k));
	
end

%
% Plot:
%

%q_tot, q_liq
figure(ifig);
plot(q_tot, zh, 'LineWidth',2)
hold on
plot(q_liq, zh, 'LineWidth',2)
legend('q_t', 'q_l');
xlabel('q (kg/kg)','Fontsize',21,'FontName','Times');
ylabel('z (m)','Fontsize',21,'FontName','Times');
set(gca,'FontName','Times', 'Fontsize',20);
set(gcf, 'Color', 'w');
if(write2file==1)
	outfilename = strcat('dy_mixing_ratios.eps');
	export_fig(outfilename, '-native');
end
ifig = ifig+1;


%T, Tv
figure(ifig);
plot(T, zh, 'LineWidth',2)
hold on
plot(Tv, zh, 'LineWidth',2)
legend('T', 'T_v');
%axis([0.5 4 0 6]);
xlabel('T, T_v (K)','Fontsize',21,'FontName','Times');
ylabel('z (m)','Fontsize',21,'FontName','Times');
set(gca,'FontName','Times', 'Fontsize',20);
set(gcf, 'Color', 'w');
if(write2file==1)
	outfilename = strcat('dy_tempe.eps');
	export_fig(outfilename, '-native');
end
ifig = ifig+1;


%theta, thetav
figure(ifig);
plot(theta, zh, 'LineWidth',2)
hold on
plot(thetav, zh, 'LineWidth',2)
hold on
plot(theta_liq, zh, 'LineWidth',2)
legend('\theta', '\theta_v', '\theta_l');
xlabel('\theta, \theta_v, \theta_l (K)','Fontsize',21,'FontName','Times');
ylabel('z (m)','Fontsize',21,'FontName','Times');
set(gca,'FontName','Times', 'Fontsize',20);
set(gcf, 'Color', 'w');
if(write2file==1)
	outfilename = strcat('dy_pot_temp.eps');
	export_fig(outfilename, '-native');
end
ifig = ifig+1;


%P
figure(ifig);
plot(p, zh, 'LineWidth',2)
%legend('p_d', 'p_m');
xlabel('p (Pa)','Fontsize',21,'FontName','Times');
ylabel('z (m)','Fontsize',21,'FontName','Times');
set(gca,'FontName','Times', 'Fontsize',20);
set(gcf, 'Color', 'w');
axis([80000 102000 0 1500])
%Write pd(z=0)
hold on
scatter([p(1) p(1)], [zh(1) zh(1)], 'o')
str = (['(',num2str(p(1)), ',', num2str(zh(1)),')'])
t = text([100000],[70], str,'FontSize',16)
if(write2file==1)
	outfilename = strcat('dy_press.eps');
	export_fig(outfilename, '-native');
end
ifig = ifig+1;

%rho
figure(ifig);
plot(rho, zh, 'LineWidth',2)
xlabel('\rho (kg/m3)','Fontsize',21,'FontName','Times');
ylabel('z (m)','Fontsize',21,'FontName','Times');
set(gca,'FontName','Times', 'Fontsize',20);
set(gcf, 'Color', 'w');
hold on
%Circle around rho=1.13 and rho=1.22
%rho(z=0)
scatter([rho(1) rho(1)], [zh(1) zh(1)], 'o')
%rho(z=zi)
Irho_zi = min(find(abs(zh - zi) <= 1));
scatter([rho(Irho_zi) rho(Irho_zi)], [zh(Irho_zi) zh(Irho_zi)], 'o')
%rho(z=zb)
Irho_zb = min(find(abs(zh - zb) <= 1));
scatter([rho(Irho_zb) rho(Irho_zb)], [zh(Irho_zb) zh(Irho_zb)], 'o')

clear t
str = (['(',num2str(rho(Irho_zi)), ',', num2str(zh(Irho_zi)),')'])
t = text([1.14],[860], str,'FontSize',16)
str = (['(',num2str(rho(Irho_zb)), ',', num2str(zh(Irho_zb)),')'])
t = text([1.16],[650], str,'FontSize',16)
str = (['(',num2str(rho(1)), ',', num2str(zh(1)),')'])
t = text([1.16],[70], str,'FontSize',16)
if(write2file==1)
	outfilename = strcat('dy_densi.eps');
	export_fig(outfilename, '-native');
end
ifig = ifig+1;
