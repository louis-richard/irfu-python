import numpy as np
import xarray as xr
from astropy import constants

from ..pyrf.calc_dt import calc_dt
from ..pyrf.resample import resample
from ..pyrf.convert_fac import convert_fac
from ..pyrf.ts_scalar import ts_scalar
from ..pyrf.extend_tint import extend_tint
from ..pyrf.tlim import tlim


def lhwaveanalysis(tints=None, e_xyz=None, b_scm=None, Bxyz=None, ne=None,**kwargs):
    """
    Calculates lower-hybrid wave properties from MMS data

    Parameters :
        tints : list of str
            Time interval

        Exyz : DataArray
            Time series pf the electric field
        
        Bscm : DataArray
            Time series of the fluctuations of the magnetic field

        Bxyz : DataArray
            Time series of the background magnetic field

        ne : DataArray
            Time series of the number density

    Options :
        lhfilt : float/int/list of float/list of int
            Filter for LH fluctuations. For one element it is the minimum frequency in the highpass filter. For two elements the fields are bandpassed between the frequencies.
        
        blpass : float/int
            Set maximum frequency for low-pass filter of background magnetic field (FGM)

    Example :
        >>> # Large time interval
        >>> Tintl   = ["2015-12-14T01:17:39.000","2015-12-14T01:17:43.000"]
        >>> # Load fields and density
        >>> Bxyz    = mms.get_data("B_gse_fgm_brst_l2",Tintl,2)
        >>> Exyz    = mms.get_data("E_gse_edp_brst_l2",Tintl,2)
        >>> Bscm    = mms.get_data("B_gse_scm_brst_l2",Tintl,2)
        >>> ne      = mms.get_data("Ne_fpi_brst_l2",Tintl,2)
        >>> 
        >>> # Time interval of focus
        >>> Tint    = ["2015-12-14T01:17:40.200","2015-12-14T01:17:41.500"]
        >>> 
        >>> phiEB, vbest, dirbest, thetas, corrs = mms.lhwaveanalysis(Tint,Exyz,Bscm,Bxyz,ne,lhfilt=[5,100],blpass=5)

    """

    # Default bandpasses
    minfreq     = 10
    maxfreq     = 0  # No low-pass filter
    lowpassBxyz = 2
    frange      = 0


    if "lhfilt" in kwargs:
        if isinstance(kwarg["lhfilt"],(float,int)):
            minfreq = kwargs["lhfilt"]
        elif isinstance(kwargs["lhfilt"],(list,np.ndarray)) and len(kwargs["lhfilt"]):
            minfreq = kwargs["lhfilt"][0]
            maxfreq = kwargs["lhfilt"][1]
        else :
            raise ValueError("lhfilt option not recognized")

    if "blpass" in kwargs:
        if isinstance(kwargs["blpass"],(float,int)):
            lowpassBxyz = kwargs["blpass"]
        else :
            raise ValueError("blpass option not recognized")


    # Bandpass filter data
    Bxyz    = filt(Bxyz,0,lowpassBxyz,5)
    Exyz    = resample(Exyz,Bscm)
    ne      = resample(ne,Bscm)
    Bxyz    = resample(Bxyz,Bscm)
    Bscmfac = convert_fac(Bscm,Bxyz,[1,0,0])
    Bscmfac = filt(Bscmfac,minfreq,maxfreq,5)
    Exyz    = filt(Exyz,minfreq,maxfreq,5)

    qe  = constants.e.value
    mu  = constants.mu0.value

    Bmag    = np.linalg.norm(Bxyz,axis=1)
    phiB    = (Bscmfac.data[:,2])*Bmag*1e-18/(ne.data*qe*mu*1e6)
    phiB    = ts_scalar(Bscmfac.time.data,phiB)

    # short buffer so phi_E does not begin at zero.
    tint    = extend_tint(tints,[-.2,.2])
    Exyz    = tlim(Exyz,tint)
    phiBs   = tlim(phiB,tints)

    # Rotate Exyz into field-aligned coordinates
    Bxyzs   = tlim(Bxyz,tints)
    Bmean   = np.mean(Bxyzs.data,axis=0)
    Bvec    = Bmean/np.linalg.norm(Bmean)
    Rtemp   = [1,0,0]
    R2      = np.cross(Bvec,Rtemp)
    R2      = R2/np.linalg.norm(R2)
    R1      = np.cross(R2,Bvec)
    ER1     = Exyz.data[:,0]*R1[0]+Exyz.data[:,1]*R1[1]+Exyz.data[:,2]*R1[2]
    ER2     = Exyz.data[:,0]*R2[0]+Exyz.data[:,1]*R2[1]+Exyz.data[:,2]*R2[2]
    ER3     = Exyz.data[:,0]*Bvec[0]+Exyz.data[:,1]*Bvec[1]+Exyz.data[:,2]*v[2]
    Efac    = ts_vec_xyz(Exyz.time.data,np.vstack([ER1,ER2,ER3]).T)


    # Find best direction
    dtEfac  = calc_dt(Efac)
    thetas  = np.linspace(0,2*np.pi,361)
    corrs   = np.zeros(len(thetas))

    for theta in thetas:
        Etemp       = np.cos(theta)*Efac.data[:,0]+np.sin(theta)*Efac.data[:,1]
        phitemp     = ts_scalar(Exyz.time.data,np.cumsum(Etemp)*dt)
        phitemp     = tlim(phitemp,tints)
        phitemp     -= np.mean(phitemp)
        corrs[ii]   = np.corrcoef(phiBs.data,phitemp.data)


    corrpos         = np.argmax(corrs)
    Ebest           = np.cos(thetas[corrpos])*Efac.data[:,0]+np.sin(thetas[corrpos])*Efac.data[:,1]
    Ebest           = ts_scalar(Exyz.time.data,Ebest)
    phibest         = ts_scalar(Exyz.time.data,np.cumsum(Ebest)*dt)
    phibest         = tlim(phibest,tints)
    phibest         -= np.mean(phibest)
    thetabest       = thetas[corrpos]
    dirbest         = R1*np.cos(thetabest)+R2*np.sin(thetabest)
    dirbestround    = np.round(dirbest,2)

    #Find best speed
    vphvec  = np.linspace(1e1,5e2,491)      # Maximum velocity may need to be increased in rare cases
    corrv   = np.zeros(len(vphvec))

    for ii, vph in enumerate(vphvec):
        phiEtemp    = phibest.data*vph
        corrv[ii]   = np.sum(np.abs(phiEtemp-phiBs.data)**2);


    corrvpos    = np.argmin(corrv)
    phiEbest    = phibest.data*vphvec[corrvpos]
    phiEbest    = ts_scalar(phiBs.time.data,phiEbest)
    vbest       = vphvec[corrvpos]

    phiEB = xr,DataArray(np.vstack([phiEbest.data,phiBs.data]).T,coords==[phiBs.time,["Ebest","Bs"]],dims=["time","comp"])

    """
    phiEmax = max(abs(phiEB.data(:,1)));
    phiBmax = max(abs(phiEB.data(:,2)));

    if plotfigure
    fn=figure;
    set(fn,'Position',[10 10 600 600])
        h(1)=axes('position',[0.1 0.58 0.8 0.4]); 
        h(2)=axes('position',[0.1 0.07 0.8 0.4]);
        ud=get(fn,'userdata');
        ud.subplot_handles=h;
        set(fn,'userdata',ud);
        set(fn,'defaultLineLineWidth',2); 
        
    h(1)=irf_panel('phi');
    irf_plot(h(1),phiEB);
    ylabel(h(1),'\phi (V)','Interpreter','tex','fontsize',14);
    irf_legend(h(1),{'\phi_{E}','\phi_{B}'},[0.1 0.12],'fontsize',14)
    if frange
        freqlab = [num2str(minfreq) ' Hz < f < ' num2str(maxfreq) ' Hz'];
    else
        freqlab = ['f > ' num2str(minfreq) ' Hz'];
    end
    vlab = ['v = ' num2str(vbest) ' km s^{-1}'];
    xdir = num2str(dirbestround(1));
    ydir = num2str(dirbestround(2));
    zdir = num2str(dirbestround(3));
    dirlab = ['dir: [' xdir ',' ydir ',' zdir ']'];
    irf_legend(h(1),'(a)',[0.99 0.98],'color','k','fontsize',14)
    c_eval('irf_legend(h(1),''MMS ?'',[0.01 0.98],''color'',''k'',''fontsize'',14)',ic);
    irf_legend(h(1),freqlab,[0.95 0.2],'color','k','fontsize',14)
    irf_legend(h(1),vlab,[0.95 0.13],'color','k','fontsize',14)
    irf_legend(h(1),dirlab,[0.95 0.06],'color','k','fontsize',14)
    irf_legend(h(1),['|\phi_E|_{max} = ' num2str(round(phiEmax,1))],[0.90 0.98],'color','k','fontsize',14)
    irf_legend(h(1),['|\phi_B|_{max} = ' num2str(round(phiBmax,1))],[0.90 0.90],'color','k','fontsize',14)
    irf_zoom(h(1),'x',tints);

    plot(h(2),thetas,corrs);
    [maxcorr,ind] = max(corrs);
    hold(h(2),'on')
    plot(h(2),thetas(ind),maxcorr,'ro');
    hold(h(2),'off')
    xlabel(h(2),'\theta (deg)','Interpreter','tex','fontsize',14);
    ylabel(h(2),'C_{\phi}','Interpreter','tex','fontsize',14);
    axis(h(2),[0 360 -1 1]);
    irf_legend(h(2),['C_{\phi max} = ' num2str(round(maxcorr,2))],[0.95 0.06],'color','k','fontsize',14)
    irf_legend(h(2),'(b)',[0.99 0.98],'color','k','fontsize',14)

    set(h(1:2),'fontsize',14);
    set(gcf,'color','w');
    end

    end
    """

    return (phiEB,vbest,dirbest,thetas,corrs)
