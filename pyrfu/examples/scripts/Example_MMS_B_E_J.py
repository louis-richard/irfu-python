% Plots of B, J, E, JxB electric field, and J.E. Calculates J using
% Curlometer method. 
% Written by D. B. Graham

from pyrfu import pyrf
from pyrfu import mms
from pyrfu import plot as pltrf
import numpy as np

Tint = ["2016-06-07T01:53:44.000","2016-06-07T19:30:34.000"]


ic = np.arange(1,5)

for i in ic:
	exec("Bxyz? = mms.get_data('B_dmpa_dfg_srvy_l2pre',Tint,?)".replace("?",str(i)))
	exec("Bxyz? = pyrf.resample(Bxyz?,Bxyz1)".replace("?",str(i)))

Bxyzav = pyrf.ts_vec_xyz(Bxyz1.time.data,(Bxyz1.data+Bxyz2.data+Bxyz3.data+Bxyz4.data)/4);

# Load electric field
for i in ic:
	exec("Exyz? = mms.get_data('E2d_dsl_edp_fast_l2pre',Tint,?)".replace("?",str(i)))
	exec("Exyz? = pyrf.resample(Exyz?,Exyz1)".replace("?",str(i)))
	#exec("[Exyz?,d?] = pyrf.edb(Exyz?,Bxyz?,15,'E.B=0')".replace("?",str(i))) 			# Removes some wake fields

Exyzav = pyrf.ts_vec_xyz(Exyz1.time.data,(Exyz1.data+Exyz2.data+Exyz3.data+Exyz4.data)/4)


for i in ic:
	exec("ni? = mms.get_data('Ni_fpi_fast_l2',Tint,?)".replace("?",str(i))) 			# For MP phases
	exec("ni? = mms.get_data('Nhplus_hpca_srvy_l2',Tint,?)".replace("?",str(i))) 		# For 1x phases
	exec("ni? = pyrf.resample(ni?,ni1)".replace("?",str(i)))

ni = pyrf.ts_scalar(ni1.time.data,(ni1.data+ni2.data+ni3.data+ni4.data)/4)
ni = pyrf.resample(ni,Bxyz1)

for i in ic:
	exec("Rxyz? = mms.get_data('R_gse',Tint,?)".replace("?",str(i)))
	exec("Rxyz? = pyrf.resample(Rxyz?,Bxyz1)".replace("?",str(i)))


% Assuming GSE and DMPA are the same coordinate system.
[j,divB,B,jxB,divTshear,divPb] = c_4_j('Rxyz?','Bxyz?');

divovercurl = divB;
divovercurl.data = abs(divovercurl.data)./j.abs.data;

% Transform current density into field-aligned coordinates
jfac = irf_convert_fac(j,Bxyzav,[1 0 0]);
jfac.coordinateSystem = 'FAC';




%%
h = irf_plot(8,'newfigure');

hca = irf_panel('BMMS');
irf_plot(hca,Bxyzav);
ylabel(hca,{'B_{DMPA}','(nT)'},'Interpreter','tex');
irf_legend(hca,{'B_{x}','B_{y}','B_{z}'},[0.88 0.10])
irf_zoom(hca,'y',[-70 70]);
irf_legend(hca,'(a)',[0.99 0.98],'color','k')

mmsColors=[0 0 0; 1 0 0 ; 0 0.5 0 ; 0 0 1];
hca = irf_panel('niMMS4'); set(hca,'ColorOrder',mmsColors)
irf_pl_tx(hca,'ni?',1);
ylabel(hca,{'n_i','(cm^{-3})'},'Interpreter','tex');
set(hca,'yscale','log');
irf_zoom(hca,'y',[1e-4 10]);
irf_legend(hca,'(b)',[0.99 0.98],'color','k')
irf_legend(hca,{'MMS1','MMS2','MMS3','MMS4'},[0.99 0.1],'color','cluster')

hca = irf_panel('J');
j.data = j.data*1e9;
irf_plot(hca,j);
ylabel(hca,{'J_{DMPA}','(nA m^{-2})'},'Interpreter','tex');
irf_legend(hca,{'J_{x}','J_{y}','J_{z}'},[0.88 0.10])
irf_legend(hca,'(c)',[0.99 0.98],'color','k')

hca = irf_panel('Jfac');
jfac.data = jfac.data*1e9;
irf_plot(hca,jfac);
ylabel(hca,{'J_{FAC}','(nA m^{-2})'},'Interpreter','tex');
irf_legend(hca,{'J_{\perp 1}','J_{\perp 2}','J_{||}'},[0.88 0.10])
irf_legend(hca,'(d)',[0.99 0.98],'color','k')

hca = irf_panel('divovercurl');
irf_plot(hca,divovercurl);
ylabel(hca,{'|\nabla . B|','|\nabla \times B|'},'Interpreter','tex');
irf_legend(hca,'(e)',[0.99 0.98],'color','k')

hca = irf_panel('EMMS1');
irf_plot(hca,Exyzav);
ylabel(hca,{'E_{DSL}','(mV m^{-1})'},'Interpreter','tex');
irf_legend(hca,{'E_{x}','E_{y}','E_{z}'},[0.88 0.10])
irf_legend(hca,'(b)',[0.99 0.98],'color','k')

hca = irf_panel('jxB');
jxB.data = jxB.data./[ni.data ni.data ni.data]; 
jxB.data = jxB.data/1.6e-19/1000; %Convert to (mV/m)
jxB.data(abs(jxB.data) > 100) = NaN; % Remove some questionable fields
irf_plot(hca,jxB);
ylabel(hca,{'J \times B/n_{e} q_{e}','(mV m^{-1})'},'Interpreter','tex');
irf_legend(hca,'(f)',[0.99 0.98],'color','k')

j = j.resample(Exyzav);
EdotJ = dot(Exyzav.data,j.data,2)/1000; %J (nA/m^2), E (mV/m), E.J (nW/m^3)
EdotJ = TSeries(Exyzav.time,EdotJ);

hca = irf_panel('jdotE');
irf_plot(hca,EdotJ);
ylabel(hca,{'E . J','(nW m^{-3})'},'Interpreter','tex');
irf_legend(hca,'(g)',[0.99 0.98],'color','k')

title(h(1),'MMS - Current density and fields');

irf_plot_axis_align(1,h(1:8))
irf_zoom(h(1:8),'x',Tint);
