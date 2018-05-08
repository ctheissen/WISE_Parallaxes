import numpy as np
import sys, os, os.path, time, gc
from astropy.table import Table, vstack, join
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.optimize import curve_fit, minimize
from astropy.time import Time
import astropy.coordinates as coords
from astropy.stats import sigma_clip
from matplotlib.offsetbox import AnchoredText
from astroquery.irsa import Irsa
import emcee
from scipy.stats import norm
import matplotlib.gridspec as gridspec
import matplotlib
import Neighbor_Offsets as ne
import Register_Frames as reg

# Set a few defaults
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
Irsa.ROW_LIMIT = -1
Irsa.TIMEOUT = 60*10 # 10 minutes

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('text', usetex=True)
plt.rc('legend', fontsize=7)
plt.rc('axes', labelsize=10)
plt.rc('axes', titlesize=10)

# Set some constants
d2a  = 3600.
d2ma = 3600000.
d2y  = 1/365.25 


###########################################################################################################
clickpoints  = []
clickpoints2 = []

def onclick(event):

  global clickpoints
  clickpoints.append([event.xdata, event.ydata])
  plt.axvline(event.xdata, c='r', ls=':') 
  plt.axhline(event.ydata, c='r', ls=':')  
  plt.draw()
  if len(clickpoints) == 2:
    print('Closing figure')
    RAmin,  RAmax  = np.min(list(zip(*clickpoints))[0]), np.max(list(zip(*clickpoints))[0])
    DECmin, DECmax = np.min(list(zip(*clickpoints))[1]), np.max(list(zip(*clickpoints))[1])
    plt.fill_between([RAmin, RAmax], [DECmin, DECmin], [DECmax, DECmax], color='0.5', alpha=0.5, zorder=-100)
    plt.draw()
    plt.pause(2)
    plt.close('all')

def onclick2(event):

  global clickpoints2
  clickpoints2.append([event.xdata, event.ydata])
  plt.axvline(event.xdata, c='r', ls=':') 
  plt.draw()
  if len(clickpoints2) == 2:
    print('Closing figure')
    plt.axvspan(clickpoints2[0][0], clickpoints2[1][0], color='0.5', alpha=0.5, zorder=-100)
    plt.draw()
    plt.pause(2)
    plt.close('all')

def onclickclose(event):
  if event.button: plt.close('all')


###########################################################################################################


def AstrometryFunc(x, Delta1, Delta2, PMra, PMdec, pi, JPL=True, RA=True, DEC=True):

  ras, decs, mjds = x
  years = (mjds - mjds[0])*d2y

  bary0 = coords.get_body_barycentric('earth', Time(mjds, format='mjd'))
  
  if JPL: # Use JPL DE430 ephemeris
      bary = bary0 / 1.496e8
  else:
      bary = bary0
  
  # Parallax factors 
  Fac1 = (bary.x * np.sin(ras/d2a*np.pi/180.) - bary.y * np.cos(ras/d2a *np.pi/180.) ) 
  Fac2 = bary.x * np.cos(ras/d2a *np.pi/180.) * np.sin(decs/d2a *np.pi/180.) + \
         bary.y * np.sin(ras/d2a *np.pi/180.) * np.sin(decs/d2a *np.pi/180.) - \
         bary.z * np.cos(decs/d2a *np.pi/180.)

  RAsend  = Delta1 + PMra  * years + pi * Fac1.value
  DECsend = Delta2 + PMdec * years + pi * Fac2.value

  if RA == True and DEC == False:
    return RAsend
  elif RA == False and DEC == True:
    return DECsend
  else:
    return np.concatenate( [RAsend, DECsend]).flatten()


###########################################################################################################


def MeasureParallax(Name='JohnDoe', radecstr=None, ra0=None, dec0=None, radius=10, 
                    PLOT=True, method='mcmc', savechain=True, JPL=True, Register=False, Calibrate=True, 
                    AllowUpperLimits=False, sigma=3, **kwargs):

  # Make directories for the plots and results
  name = Name.replace('$','').replace(' ','').replace('.','')
  if not os.path.exists('%s/Plots'%name):
    os.makedirs('%s/Plots'%name)
  if not os.path.exists('%s/Results'%name):
    os.makedirs('%s/Results'%name)

  # Get the object
  if radecstr != None:

    t1 = Irsa.query_region(coords.SkyCoord(radecstr, unit=(u.deg,u.deg), frame='icrs'), 
                           catalog="allsky_4band_p1bs_psd", spatial="Cone", radius=radius * u.arcsec)
    t2 = Irsa.query_region(coords.SkyCoord(radecstr, unit=(u.deg,u.deg), frame='icrs'), 
                           catalog="allsky_3band_p1bs_psd", spatial="Cone", radius=radius * u.arcsec)
    if len(t2) == 0:
      t2 = Irsa.query_region(coords.SkyCoord(radecstr, unit=(u.deg,u.deg), frame='icrs'), 
                             catalog="allsky_2band_p1bs_psd", spatial="Cone", radius=radius * u.arcsec)
    t3 = Irsa.query_region(coords.SkyCoord(radecstr, unit=(u.deg,u.deg), frame='icrs'), 
                          catalog="neowiser_p1bs_psd", spatial="Cone", radius=radius * u.arcsec)

  elif ra0 != None and dec0 != None:

    t1 = Irsa.query_region(coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs'), 
                           catalog="allsky_4band_p1bs_psd", spatial="Cone", radius=radius * u.arcsec)
    t2 = Irsa.query_region(coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs'), 
                           catalog="allsky_3band_p1bs_psd", spatial="Cone", radius=radius * u.arcsec)
    if len(t2) == 0:
      t2 = Irsa.query_region(coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs'), 
                             catalog="allsky_2band_p1bs_psd", spatial="Cone", radius=radius * u.arcsec)
    t3 = Irsa.query_region(coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs'), 
                           catalog="neowiser_p1bs_psd", spatial="Cone", radius=radius * u.arcsec)
  else:
    raise ValueError("Need to supply either radecstr or ra0 and dec0") # Need some coords

  t00  = vstack([t1, t2], join_type='inner')
  t0   = vstack([t00, t3], join_type='inner')
  
  index00   = np.argsort(t0['mjd'])

  t = t0[index00]

  if JPL: # Use the JPL DE430 ephemeris
      from astropy.coordinates import solar_system_ephemeris
      solar_system_ephemeris.set('jpl') 


  ######################################################################################################


  def AstrometryFunc0(x, Delta1, Delta2, PMra, PMdec, pi):

    ras, decs, mjds = x
    years = (mjds - mjds[0])*d2y

    bary0 = coords.get_body_barycentric('earth', Time(mjds, format='mjd'))
    
    if JPL:
        bary = bary0 / 1.496e8
    else:
        bary = bary0

    # Parallax factors   
    Fac1 = (bary.x * np.sin(ras/d2a*np.pi/180.) - bary.y * np.cos(ras/d2a *np.pi/180.) ) 
    Fac2 = bary.x * np.cos(ras/d2a *np.pi/180.) * np.sin(decs/d2a *np.pi/180.) + \
           bary.y * np.sin(ras/d2a *np.pi/180.) * np.sin(decs/d2a *np.pi/180.) - \
           bary.z * np.cos(decs/d2a *np.pi/180.)

    RAsend  = Delta1 + PMra  * years + pi * Fac1.value
    DECsend = Delta2 + PMdec * years + pi * Fac2.value

    if RA == True and DEC == False:
      return RAsend
    elif RA == False and DEC == True:
      return DECsend
    else:
      return np.concatenate( [RAsend, DECsend]).flatten()
  

  ######################################################################################################

  # Plot the RA, DEC, and MJDs
  fig0 = plt.figure(1, figsize=(7,6))
  ax0  = fig0.add_subplot(211)
  ax00 = fig0.add_subplot(212)
  ax0.scatter(t1['mjd'], t1['ra']*d2a, c='r', alpha=0.5)
  ax0.scatter(t2['mjd'], t2['ra']*d2a, c='g', alpha=0.5)
  ax0.scatter(t3['mjd'], t3['ra']*d2a, c='b', alpha=0.5)

  ax00.scatter(t1['mjd'], t1['dec']*d2a, c='r', alpha=0.5)
  ax00.scatter(t2['mjd'], t2['dec']*d2a, c='g', alpha=0.5)
  ax00.scatter(t3['mjd'], t3['dec']*d2a, c='b', alpha=0.5)

  ax0.set_xlabel('MJD')
  ax0.set_ylabel('R.A. (arcsec)')
  ax00.set_xlabel('MJD')
  ax00.set_ylabel('Dec. (arcsec)')

  if PLOT:
    fig = plt.figure(2, figsize=(7,6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.scatter(t['mjd'], t['ra']*d2a, c='r', alpha=0.5)
    for j in np.arange(-1, 30, 2): 
      ax1.axvline(np.min(t['mjd']) + 365.25/4*j , c='k', ls='--')
    ax2.scatter(t['mjd'], t['dec']*d2a, c='r', alpha=0.5)
    for j in np.arange(-1, 30, 2): 
      ax2.axvline(np.min(t['mjd']) + 365.25/4*j , c='k', ls='--')

    fig3, ax3 = plt.subplots()
    cid = fig3.canvas.mpl_connect('button_press_event', onclick)
    ax3.scatter(t['ra']*d2a, t['dec']*d2a, alpha=0.3)
    ax3.set_xlabel('R.A. (arcsec)')
    xmin,xmax = ax3.get_xlim()
    ax3.set_xlim(xmax, xmin)
    ax3.set_ylabel('Dec. (arcsec)')
    ax3.set_title('Select two points to form a bounding box around target\n(plot will close automatically when finished)')
    plt.show()
    plt.close('all')

    # Get the RADEC limits from the click points
    RAmin,  RAmax  = np.min(list(zip(*clickpoints))[0]), np.max(list(zip(*clickpoints))[0])
    DECmin, DECmax = np.min(list(zip(*clickpoints))[1]), np.max(list(zip(*clickpoints))[1])
    slice1 = np.where( (t['ra']*d2a  >= RAmin)  & (t['ra']*d2a  <= RAmax) & 
                       (t['dec']*d2a >= DECmin) & (t['dec']*d2a <= DECmax) )
    # Get the magnitude limits from the click points
    fig4 = plt.figure(104)
    cid = fig4.canvas.mpl_connect('button_press_event', onclick2)
    x = np.linspace(np.min(t['w2mpro'][slice1])-2*np.max(t['w2sigmpro'][slice1]), np.max(t['w2mpro'][slice1])+2*np.max(t['w2sigmpro'][slice1]), 2000)
    W2pdf = np.zeros(len(x))
    for w2, w2err in list(zip(t['w2mpro'][slice1].filled(-9999), t['w2sigmpro'][slice1].filled(-9999))):
      if w2err == -9999: 
        if w2 == -9999: 
          continue # Skip only upper limits
        else:
          plt.axvline(w2, ls='--', lw=0.75, c='r', alpha=0.5, zorder=-100)
          continue # Skip only upper limits
      if w2 == -9999: 
        continue # Skip only upper limits
      plt.axvline(w2, ls=':', lw=0.75, c='0.5', alpha=0.5, zorder=-100)
      W2pdf += norm.pdf(x, loc=w2, scale=w2err)

    plt.plot(x, W2pdf / np.trapz(W2pdf, x=x), zorder=100)
    plt.plot([x[1000],x[1000]],[0,0], 'r--', lw=0.75, alpha=0.5, label='Upper limits')
    plt.plot([x[1000],x[1000]],[0,0], c='0.5', ls=':', lw=0.75, alpha=0.5, label='Individual measurements')
    plt.xlabel(r'$W2$')
    plt.ylabel('PDF')
    plt.legend(frameon=False)
    plt.title('Select the lower and upper bound magnitudes\n(plot will close automatically when finished)')
    plt.show()
    plt.close('all')

    W2min,  W2max  = np.min(list(zip(*clickpoints2))[0]), np.max(list(zip(*clickpoints2))[0])
    if AllowUpperLimits: # Sometimes useful for very faint sources (Y dwarfs)
      slice2 = np.where( (t['w2mpro'][slice1] >= W2min)  & (t['w2mpro'][slice1] <= W2max) )
    else:
      slice2 = np.where( (t['w2mpro'][slice1] >= W2min)  & (t['w2mpro'][slice1] <= W2max) & (t['w2sigmpro'][slice1].filled(-9999) != -9999) )
      

  #### Find the date clusters
  Groups = []
  Epochs = []
  DateGrps = np.arange(-1, 30, 2)
  for i in range(len(DateGrps)-1): 
    bottom = np.min(t['mjd'][slice1][slice2]) + 365.25/4*DateGrps[i]
    top    = np.min(t['mjd'][slice1][slice2]) + 365.25/4*DateGrps[i+1]
    group  = np.where( (t['mjd'][slice1][slice2] > bottom) & (t['mjd'][slice1][slice2] < top) )
    if len(group[0]) != 0:
      Groups.append(group[0])
      Epochs.append([bottom, top])

  MJDs  = []
  Ys1   = []
  Ys2   = []
  unYs1 = []
  unYs2 = []

  XsALL  = []
  Ys1ALL = []
  Ys2ALL = []

  Colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
  i = 0
  groupcount = 0

  if PLOT:

    fig2 = plt.figure(3)
    cid  = fig2.canvas.mpl_connect('button_press_event', onclickclose)
    ax3  = fig2.add_subplot(111)
  
  for group in Groups:

    groupcount += 1

    if Register != False: ## Register the epoch

      # Get the first position of the epoch
      ra00     = t['ra'][slice1][slice2][group][0]
      dec00    = t['dec'][slice1][slice2][group][0]
      epochs00 = t['mjd'][slice1][slice2][group]

      # Get the shifts (only need to find the correct search radius for the 1st epoch)
      if groupcount == 1:
        rashifts0, decshifts0, RegisterRadius = reg.GetRegistrators(name, epochs00, subepoch=groupcount, ra0=ra00, dec0=dec00)
      else: 
        rashifts0, decshifts0, RegisterRadius = reg.GetRegistrators(name, epochs00, subepoch=groupcount, ra0=ra00, dec0=dec00, radius=RegisterRadius)

      # Shift the epoch
      shiftedRAs  = t['ra'][slice1][slice2][group]  + rashifts0
      shiftedDECs = t['dec'][slice1][slice2][group] + decshifts0
      #sys.exit()

      filteredRA  = sigma_clip(shiftedRAs,  sigma=sigma, iters=None)
      filteredDEC = sigma_clip(shiftedDECs, sigma=sigma, iters=None)

    else: # Don't register each subepoch

      filteredRA  = sigma_clip(t['ra'][slice1][slice2][group],  sigma=sigma, iters=None)
      filteredDEC = sigma_clip(t['dec'][slice1][slice2][group], sigma=sigma, iters=None)

    index = np.where( (~filteredRA.mask) & (~filteredDEC.mask) )[0]
    print('Epoch %s - Group / Filtered Group: %s / %s'%(groupcount, len(t['ra'][slice1][slice2][group]), len(t['ra'][slice1][slice2][group][index])))

    if PLOT:
      ax1.scatter(t['mjd'][slice1][slice2][group][index], t['ra'][slice1][slice2][group][index]*d2a, c='b', marker='x', alpha=0.5)
      ax2.scatter(t['mjd'][slice1][slice2][group][index], t['dec'][slice1][slice2][group][index]*d2a, c='b', marker='x', alpha=0.5)

      ax3.scatter(t['ra'][slice1][slice2][group]*d2a,        t['dec'][slice1][slice2][group]*d2a, alpha=0.3, color=Colors[i], label='%s'%np.average(t['mjd'][slice1][slice2][group][index]))
      ax3.scatter(t['ra'][slice1][slice2][group][index]*d2a, t['dec'][slice1][slice2][group][index]*d2a, s=2, color=Colors[i], label='%s'%np.average(t['mjd'][slice1][slice2][group][index]))
      ax3.errorbar(np.average(t['ra'][slice1][slice2][group][index],  weights = 1./(t['sigra'][slice1][slice2][group][index]/d2a)**2)*d2a,
                   np.average(t['dec'][slice1][slice2][group][index], weights = 1./(t['sigdec'][slice1][slice2][group][index]/d2a)**2)*d2a,
                   xerr = np.std(t['ra'][slice1][slice2][group][index])  / np.sqrt(len(t[slice1][slice2][group][index][0])), 
                   yerr = np.std(t['dec'][slice1][slice2][group][index]) / np.sqrt(len(t[slice1][slice2][group][index][0])), c=Colors[i], marker='x', ms=20)

    i += 1

    MJDs.append(np.average(t['mjd'][slice1][slice2][group][index]))

    Ys1.append(np.average(t['ra'][slice1][slice2][group][index],  weights = 1./(t['sigra'][slice1][slice2][group][index]/d2a)**2))
    Ys2.append(np.average(t['dec'][slice1][slice2][group][index], weights = 1./(t['sigdec'][slice1][slice2][group][index]/d2a)**2))

    # Uncertainty weighted position
    unYs1.append( (1. / np.sqrt(np.sum(1./t['sigra'][slice1][slice2][group][index]**2)) ) / d2a)
    unYs2.append( (1. / np.sqrt(np.sum(1./t['sigdec'][slice1][slice2][group][index]**2)) ) / d2a)

    # This is just for plotting
    XsALL.append(t['mjd'][slice1][slice2][group][index].data.compressed())
    Ys1ALL.append(t['ra'][slice1][slice2][group][index].data.compressed())
    Ys2ALL.append(t['dec'][slice1][slice2][group][index].data.compressed())
    
  raP, decP = Ys1[0], Ys2[0]

  if PLOT: 

    ax1.set_xlabel('MJD')
    ax1.set_ylabel('R.A. (arcsec)')
    ax2.set_xlabel('MJD')
    ax2.set_ylabel('Dec. (arcsec)')

    ax3.legend(frameon=False)
    ax3.set_title('Here are the measurements we will fit to\n(click anywhere to close)')    
    ax3.set_xlabel('R.A. (arcsec)')
    xmin,xmax = ax3.get_xlim()
    ax3.set_xlim(xmax, xmin)
    ax3.set_ylabel('Dec. (arcsec)')
    
    fig0.savefig('%s/Plots/MJDs0.png'%name, dpi=600, bbox_inches='tight')
    fig.savefig('%s/Plots/MJDs1.png'%name, dpi=600, bbox_inches='tight')
    fig2.savefig('%s/Plots/MJDs2.png'%name, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close('all')

  if groupcount < 4: 
    # Only do objects that have multiple observations
    raise Exception('Not enough epochs available. Only %s epochs available'%groupcount) 

  # Get the shifts using calibrators (Only use 10 arcsec/arcminutes)
  if Calibrate == True:

    if radecstr != None:
      if Register == True:
        rashifts, decshifts = ne.GetCalibrators(name, Epochs, radecstr=radecstr, radius=RegisterRadius)
      else:
        rashifts, decshifts = ne.GetCalibrators(name, Epochs, radecstr=radecstr)

    elif ra0 != None and dec0 != None:
      if Register == True:
        rashifts, decshifts = ne.GetCalibrators(name, Epochs, ra0=ra0, dec0=dec0, radius=RegisterRadius)
      else: 
        rashifts, decshifts = ne.GetCalibrators(name, Epochs, ra0=ra0, dec0=dec0)

    print('Shifts (mas):')
    print('RA:', rashifts*d2ma)
    print('DEC:', decshifts*d2ma)

    Ys1   = np.array(Ys1).flatten() - rashifts
    Ys2   = np.array(Ys2).flatten() - decshifts
    Ys1ALL0 = np.array(Ys1ALL) - rashifts
    Ys2ALL0 = np.array(Ys2ALL) - decshifts
  else:
    Ys1   = np.array(Ys1).flatten()
    Ys2   = np.array(Ys2).flatten()
    Ys1ALL0 = np.array(Ys1ALL)
    Ys2ALL0 = np.array(Ys2ALL)

  MJDs  = np.array(MJDs).flatten()
  unYs1 = np.array(unYs1).flatten()
  unYs2 = np.array(unYs2).flatten()

  XsALL0  = np.array(XsALL)

  # Need to reshape the arrays. Not the most efficient thing.
  XsALL  = np.empty(0)
  Ys1ALL = np.empty(0)
  Ys2ALL = np.empty(0)
  for i in range(len(Ys1ALL0)):
    XsALL    = np.append(XsALL, XsALL0[i])
    Ys1ALL   = np.append(Ys1ALL, Ys1ALL0[i])
    Ys2ALL   = np.append(Ys2ALL, Ys2ALL0[i])
  XsALL  = np.array(XsALL).flatten()
  Ys1ALL = np.array(Ys1ALL).flatten()
  Ys2ALL = np.array(Ys2ALL).flatten() 

  Twrite1 = Table([Ys1, unYs1, Ys2, unYs2, MJDs], names=['RA','SIGRA','DEC','SIGDEC','MJD'])
  Twrite1.write('%s/Results/Weighted_Epochs.csv'%name, overwrite=True)
  Twrite2 = Table([Ys1ALL, Ys2ALL, XsALL], names=['RA','DEC','MJD'])
  Twrite2.write('%s/Results/All_Epochs.csv'%name, overwrite=True)

  print('Epochs:', groupcount)
  print('Average Pos Uncert (RA; mas):', np.mean(unYs1 * d2ma), np.median(unYs1 * d2ma))
  print('Average Pos Uncert (Decl; mas):', np.mean(unYs2 * d2ma), np.median(unYs2 * d2ma)) 
  print('Average Pos Uncert (mas):', (np.mean(unYs1 * d2ma) + np.mean(unYs2 * d2ma)) / 2.)
  print('Average Pos Uncert (Combined; mas):', (np.mean(np.sqrt(unYs1**2 + unYs1[0]**2)) * d2ma + np.mean(np.sqrt(unYs2**2 + unYs2[0]**2)) * d2ma) / 2.)
  print('Time Baseline (yr):', (np.max(MJDs) - np.min(MJDs)) * d2y)

  # Uncertainty arrays in arcsec
  RA_Uncert  = np.sqrt(unYs1**2 + unYs1[0]**2)*d2a
  #RA_Uncert  = np.sqrt( np.cos(Ys2[0]*np.pi/180.)**2 * (unYs1**2 + unYs1[0]**2) + \
  #                      np.sin(Ys2[0]*np.pi/180.)**2 * (Ys1 - Ys1[0])**2 * unYs2**2 ) * d2a
  DEC_Uncert = np.sqrt(unYs2**2 + unYs2[0]**2)*d2a

  print('INITIAL FIT')
  RA, DEC = False, False
  #poptD, popc = curve_fit(func1radec, [Ys1*d2a, Ys2*d2a, MJDs] , np.concatenate( [np.cos(Ys2[0]*np.pi/180.)*(Ys1 - Ys1[0])*d2a, (Ys2 - Ys2[0])*d2a] ).flatten(), sigma=np.concatenate( [np.sqrt( (unYs1**2 + unYs1[0]**2)*np.cos(Ys2[0]*np.pi/180.)**2 + unYs2[0]**2*(Ys1-Ys1[0])**2*np.sin(Ys2[0]*np.pi/180)**2)*d2a, np.sqrt(unYs2**2 + unYs2[0]**2)*d2a ] ).flatten())
  #bounds = [[-10,-10,-10,-10,0],[10,10,10,10,1]]
  poptD, popc = curve_fit(AstrometryFunc0, [Ys1*d2a, Ys2*d2a, MJDs] , 
                          np.concatenate( [(Ys1 - Ys1[0])*np.cos(Ys2[0]*np.pi/180.)*d2a, (Ys2 - Ys2[0])*d2a] ).flatten(), 
                          sigma=np.concatenate( [ RA_Uncert, DEC_Uncert ] ).flatten() )
  #print(poptD)
  #print(popc)
  #print(np.diag(popc))
  print('DELTARA\tDELTADE\tPM_RA \tPM_DEC\tPLX\n{:.7}\t{:.7}\t{:.6}\t{:.6}\t{:.5}'.format(str(poptD[0]), str(poptD[1]), str(poptD[2]), str(poptD[3]), str(poptD[4])))
  print('%s pc'%(1/poptD[-1]))

  if method == 'leastsq':
    return poptD, np.diag(popc)

  if method == 'amoeba': # still a work in progress
    RA, DEC = False, False
    #poptD, popc = curve_fit(func1radec, [Ys1*d2a, Ys2*d2a, MJDs] , np.concatenate( [np.cos(Ys2[0]*np.pi/180.)*(Ys1 - Ys1[0])*d2a, (Ys2 - Ys2[0])*d2a] ).flatten(), sigma=np.concatenate( [np.sqrt( (unYs1**2 + unYs1[0]**2)*np.cos(Ys2[0]*np.pi/180.)**2 + unYs2[0]**2*(Ys1-Ys1[0])**2*np.sin(Ys2[0]*np.pi/180)**2)*d2a, np.sqrt(unYs2**2 + unYs2[0]**2)*d2a ] ).flatten())
    #bounds = [[-10,-10,-10,-10,0],[10,10,10,10,1]]
    poptD, popc = minimize(AstrometryFunc0, [Ys1*d2a, Ys2*d2a, MJDs] , 
                           np.concatenate( [(Ys1 - Ys1[0])*np.cos(Ys2[0]*np.pi/180.)*d2a, (Ys2 - Ys2[0])*d2a] ).flatten(), 
                           method='Nelder-Mead', 
                           sigma=np.concatenate( [ RA_Uncert, DEC_Uncert ] ).flatten() )

    return poptD, np.diag(popc)
  
  ##########################################################################
  print('Starting MCMC')

  def ln_likelihood(parameters, x, yerr):

      delta1, delta2, pmra, pmdec, pi = parameters

      y    = np.concatenate( [ np.cos(Ys2[0]*np.pi/180.)*(Ys1 - Ys1[0])*d2a, (Ys2 - Ys2[0])*d2a ] ).flatten()

      modelval = AstrometryFunc0(x, delta1, delta2, pmra, pmdec, pi)
      invsig2  = 1 / np.array(yerr).flatten()**2
      LogChiSq = -0.5*np.sum( ( (np.array(y) - np.array(modelval).flatten() )**2 * invsig2 - np.log(invsig2) ) ) 

      return LogChiSq

  def ln_prior(parameters):
      # Define the priors. 
   
      pmra0, pmra00   = -100, 100
      pmdec0, pmdec00 = -100, 100
      pi0, pi00       = 0, 1
      
      delta1, delta2, pmra, pmdec, pi = parameters

      if  pmra0 <= pmra <= pmra00 and pmdec0 <= pmdec <= pmdec00 and pi0 <= pi <= pi00:
          return 0
      else:
          return -np.inf

  def ln_probability(parameters, x, yerrs):

      delta1, delta2, pmra, pmdec, pi = parameters

      priors = ln_prior(parameters)
      
      if not np.isfinite(priors):
        return -np.inf
      else:
        return priors + ln_likelihood(parameters, x, yerrs)

  # Set up the MCMC
  n_dim, n_walkers, n_steps = 5, 200, 200
  x       = [Ys1*d2a, Ys2*d2a, MJDs]
  yerr    = np.concatenate( [ RA_Uncert, DEC_Uncert ] ).flatten()
  sampler = emcee.EnsembleSampler(n_walkers, n_dim, ln_probability, args=(x, yerr))

  pos = np.array([poptD + 0.2*np.random.randn(n_dim) for i in range(n_walkers)]) # Take initial walkers by best fit values
  pos[:,-1] = abs(pos[:,-1]) # Fix for negative parallax values

  RA, DEC = True, True

  sampler.run_mcmc(pos, n_steps)
  samples = sampler.chain

  # Now plot them
  nwalkers, nsamples, ndim = samples.shape
  labels  = ['Delta1', 'Delta2', 'PMra', 'PMdec', 'pi']
  truths  = None
  extents = None
  xs      = samples
  fig     = plt.figure()
  cid     = fig.canvas.mpl_connect('button_press_event', onclickclose)
  gs      = gridspec.GridSpec(ndim, 5)
  # For each parameter, I want to plot each walker on one panel, and a histogram
  # of all links from all walkers
  for ii in range(ndim):
      walkers = xs[:,:,ii]
      flatchain = np.hstack(walkers)
      ax1 = plt.subplot(gs[ii, :5])
      
      steps = np.arange(nsamples)
      for walker in walkers:
          ax1.plot(steps, walker,
                   drawstyle="steps", color="0.5", alpha=0.4)
      
      if labels:
          ax1.set_ylabel(labels[ii])
      
      # Don't show ticks on the y-axis
      #ax1.yaxis.set_ticks([])
      
      # For the plot on the bottom, add an x-axis label. Hide all others
      if ii == ndim-1:
          ax1.set_xlabel("step number")
      else:
          ax1.xaxis.set_visible(False)


  samples = sampler.chain[:, 100:, :].reshape((-1, n_dim))

  # Save the chain?
  if savechain:
    np.save('%s/Results/MCMCresults.npy'%name, samples)

  a_mcmc, b_mcmc, c_mcmc, d_mcmc, e_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                               list(zip(*np.percentile(samples, [16, 50, 84],
                                                  axis=0))))

  print('Printing Quantiles (median, 16th, 84th)')
  print('DELTARA:', a_mcmc)
  print('DELTADEC:', b_mcmc)
  print('PMRA:', c_mcmc)
  print('PMDEC:', d_mcmc)
  print('PLX:', e_mcmc)
  print('dist (pc), [high, low]: %0.2f [%0.2f, %0.2f]'%(1./e_mcmc[0], 1./(e_mcmc[0] - e_mcmc[1]), 1./(e_mcmc[0] + e_mcmc[2])))

  if PLOT: 
    plt.show()
    plt.close('all')



  ##### EMCEE Values
  poptEMCEE    = np.array([ a_mcmc[0], b_mcmc[0], c_mcmc[0], d_mcmc[0], e_mcmc[0] ])
  poptEMCEEcov = np.array([ np.max([a_mcmc[1], a_mcmc[2]]), np.max([b_mcmc[1], b_mcmc[2]]), np.max([c_mcmc[1], c_mcmc[2]]),
                            np.max([d_mcmc[1], d_mcmc[2]]), np.max([e_mcmc[1], e_mcmc[2]]) ])
  ##################


  if PLOT == False: # Don't need to go any further
    return poptEMCEE, poptEMCEEcov

  # Get a random sample of walkers for plotting
  Xs   = np.linspace(np.min(MJDs), np.max(MJDs), 2000)
  RAs  = np.zeros(len(Xs)) + Ys1[0]*d2a
  DECs = np.zeros(len(Xs)) + Ys2[0]*d2a

  ########################################################################### Plot EMCEE values

  fig = plt.figure(1, figsize=(5, 4*3/4.))
  cid = fig.canvas.mpl_connect('button_press_event', onclickclose)
  ax = fig.add_subplot(111)
  fontsize = 7
  ms=2
  offset = 1

  if poptEMCEE[0] > poptEMCEE[1]:

    ### PLOT THE RA

    ax.errorbar(MJDs, offset+(Ys1-Ys1[0])*d2a*np.cos(Ys2[0]*np.pi/180.), 
                yerr = np.sqrt(unYs1**2 + unYs1[0]**2)*d2a, marker='o',linestyle='None', color='b', ms=ms)

    ax.scatter(XsALL, offset+(Ys1ALL - Ys1[0])*d2a*np.cos(Ys2[0]*np.pi/180.), c='0.5', alpha=0.3, zorder=-10, s=4)

    Xs   = np.linspace(np.min(MJDs), np.max(MJDs), 2000)
    RAs  = np.zeros(len(Xs)) + Ys1[0]*d2a
    DECs = np.zeros(len(Xs)) + Ys2[0]*d2a
    RA, DEC = True, False
    RAplot = AstrometryFunc0([RAs, DECs, Xs], *poptEMCEE)
    ax.plot(Xs, offset+RAplot, 'k-', lw=0.5)

    ax.text(np.min(MJDs)+50, 1.5*offset, r'$\Delta \alpha \cos \delta$', fontsize=fontsize)

    ### PLOT THE DEC

    ax.errorbar(MJDs, (Ys2-Ys2[0])*d2a, yerr = np.sqrt(unYs2**2 + unYs2[0]**2)*d2a, marker='^',linestyle='None', ms=ms)

    ax.scatter(XsALL, (Ys2ALL - Ys2[0])*d2a, c='0.5', alpha=0.3, zorder=-10, s=4)

    RA, DEC = False, True
    DECplot = AstrometryFunc0([RAs, DECs, Xs], *poptEMCEE)
    ax.plot(Xs, DECplot, 'k-', lw=0.5)

    ax.text(np.min(MJDs)+50, -1*offset, r'$\Delta \delta$', fontsize=fontsize)

  else: 

    ### PLOT THE RA

    ax.errorbar(MJDs, (Ys1-Ys1[0])*d2a*np.cos(Ys2[0]*np.pi/180.), yerr = np.sqrt(unYs1**2 + unYs1[0]**2)*d2a, marker='o',linestyle='None', color='b', ms=ms)
  
    ax.scatter(XsALL, (Ys1ALL - Ys1[0])*d2a*np.cos(Ys2[0]*np.pi/180.), c='0.5', alpha=0.3, zorder=-10, s=4)

    Xs   = np.linspace(np.min(MJDs), np.max(MJDs), 2000)
    RAs  = np.zeros(len(Xs)) + Ys1[0]*d2a
    DECs = np.zeros(len(Xs)) + Ys2[0]*d2a
    RA, DEC = True, False
    RAplot = AstrometryFunc0([RAs, DECs, Xs], *poptEMCEE)
    ax.plot(Xs, RAplot, 'k-', lw=0.5)

    ax.text(np.min(MJDs)+50, -1*offset, r'$\Delta \alpha \cos \delta$', fontsize=fontsize)


    ### PLOT THE DEC

    ax.errorbar(MJDs, offset+(Ys2-Ys2[0])*d2a, yerr = np.sqrt(unYs2**2 + unYs2[0]**2)*d2a, marker='^',linestyle='None', ms=ms)

    ax.scatter(XsALL, offset+(Ys2ALL - Ys2[0])*d2a, c='0.5', alpha=0.3, zorder=-10, s=4)

    RA, DEC = False, True
    DECplot = AstrometryFunc0([RAs, DECs, Xs], *poptEMCEE)
    ax.plot(Xs, offset+DECplot, 'k-', lw=0.5)

    ax.text(np.min(MJDs)+50, 1.5*offset, r'$\Delta \delta$', fontsize=fontsize)


  at = AnchoredText('MCMC Fit' + '\n' + r'$\mu_\alpha \cos \delta = %0.0f \pm %0.0f$ mas yr$^{-1}$'%(poptEMCEE[-3]*1e3, poptEMCEEcov[-3]*1e3) + '\n' + r'$\mu_\delta = %0.0f \pm %0.0f$ mas yr$^{-1}$'%(poptEMCEE[-2]*1e3, poptEMCEEcov[-2]*1e3)  + '\n' + r'$\pi = %0.0f \pm %0.0f$ mas'%(poptEMCEE[-1]*1e3, poptEMCEEcov[-1]*1e3),
                    prop=dict(size=8), frameon=False,
                    loc=2,
                    )
  ax.add_artist(at)

  ax.set_ylabel(r'Motion + offset (arcsec)')
  ax.set_xlabel(r'MJD (day)')
  plt.minorticks_on()
  plt.savefig('%s/Plots/Pi_radec_solution.png'%name, dpi=600, bbox_inches='tight')
  plt.show()
  plt.close('all')


  ###############################


  fig = plt.figure(2, figsize=(3.4, 3.4*3/4.))
  cid = fig.canvas.mpl_connect('button_press_event', onclickclose)

  plt.errorbar( (Ys1-Ys1[0])*d2a*np.cos(Ys2[0]*np.pi/180.), 
                (Ys2-Ys2[0])*d2a, 
                xerr = RA_Uncert, 
                yerr = DEC_Uncert, marker='o',linestyle='None', color='b', ms=ms)
  plt.plot(RAplot, DECplot, 'k-', lw=0.5)

  plt.xlabel(r'$\Delta \alpha \cos \delta$ (arcsec)')
  plt.ylabel(r'$\Delta \delta$ (arcsec)')
  plt.minorticks_on()
  plt.savefig('%s/Plots/Pi_all_solution.png'%name, dpi=600, bbox_inches='tight')
  plt.show()
  plt.close('all')

  ################################################## Plot residuals without proper motion

  fig = plt.figure(3, figsize=(8,6))
  cid = fig.canvas.mpl_connect('button_press_event', onclickclose)
  ax1 = fig.add_subplot(211)
  ax2 = fig.add_subplot(212)

  ax1.errorbar(MJDs, (Ys1-Ys1[0])*d2a*np.cos(Ys2[0]*np.pi/180.) - poptEMCEE[-3]*(MJDs-MJDs[0])*d2y - poptEMCEE[0], 
               yerr = RA_Uncert, marker='o',linestyle='None', color='b', ms=ms)
  ax1.scatter(XsALL, (Ys1ALL - Ys1[0])*d2a*np.cos(Ys2[0]*np.pi/180.) - poptEMCEE[-3]*(XsALL-XsALL[0])*d2y - poptEMCEE[0], 
              c='0.5', alpha=0.3, zorder=-10, s=4, marker='o')

  Xs2  = np.linspace(np.min(MJDs), np.max(MJDs), 2000)
  RAs  = np.zeros(len(Xs2)) + Ys1[0]*d2a
  DECs = np.zeros(len(Xs2)) + Ys2[0]*d2a

  RA, DEC = True, False
  RAplot = AstrometryFunc0([RAs, DECs, Xs2], *poptEMCEE)
  ax1.plot(Xs2, RAplot - poptEMCEE[-3]*(Xs2-Xs2[0])*d2y - poptEMCEE[0], 'k-', lw=0.5)

  ####### Dec part

  ax2.errorbar(MJDs, (Ys2-Ys2[0])*d2a - poptEMCEE[-2]*(MJDs-MJDs[0])*d2y - poptEMCEE[1], 
               yerr = DEC_Uncert, marker='^',linestyle='None', ms=ms)
  ax2.scatter(XsALL, (Ys2ALL - Ys2[0])*d2a - poptEMCEE[-2]*(XsALL-XsALL[0])*d2y - poptEMCEE[1], 
              c='0.5', alpha=0.3, zorder=-10, s=4, marker='^')
    
  RA, DEC = False, True
  DECplot = AstrometryFunc0([RAs, DECs, Xs2], *poptEMCEE)
  ax2.plot(Xs2, DECplot - poptEMCEE[-2]*(Xs2-Xs2[0])*d2y - poptEMCEE[1], 'k-', lw=0.5)

  ax1.set_ylabel(r'$\Delta$RA (arcsec)')
  ax2.set_ylabel(r'$\Delta$Dec (arcsec)')
  ax2.set_xlabel(r'MJD (day)')

  plt.savefig('%s/Plots/Pi_RA_DEC_solution.png'%name, dpi=600, bbox_inches='tight')

  plt.show()
  plt.close('all')

  ################################################## Plot the parallax circle

  fig = plt.figure(4, figsize=(6,6))
  cid = fig.canvas.mpl_connect('button_press_event', onclickclose)
  ax1 = fig.add_subplot(111)
  ax1.errorbar( (Ys1-Ys1[0])*d2a*np.cos(Ys2[0]*np.pi/180.) - poptEMCEE[-3]*(MJDs-MJDs[0])*d2y - poptEMCEE[0], 
                (Ys2-Ys2[0])*d2a - poptEMCEE[-2]*(MJDs-MJDs[0])*d2y - poptEMCEE[1],
                xerr = RA_Uncert, 
                yerr = DEC_Uncert,
                marker='o',linestyle='None', color='b', ms=ms)
  ax1.plot(RAplot  - poptEMCEE[-3]*(Xs2-Xs2[0])*d2y - poptEMCEE[0], 
           DECplot - poptEMCEE[-2]*(Xs2-Xs2[0])*d2y - poptEMCEE[1], 'k-', lw=0.5)

  ax1.set_xlabel(r'$\Delta$RA (arcsec)')
  ax1.set_ylabel(r'$\Delta$Dec (arcsec)')

  plt.savefig('%s/Plots/Pi_circle_solution.png'%name, dpi=600, bbox_inches='tight')

  plt.show()
  plt.close('all')

  ##################################################

  return 0



###########################################################################################################


def PlotParallax(Name, Place, offset, offset1, offset2, PDFsave=False):

  # Get the name for the filepaths
  name = Name.replace('$','').replace(' ','').replace('.','')

  ##### EMCEE Values
  samples = np.load('%s/Results/MCMCresults.npy'%name)
  a_mcmc, b_mcmc, c_mcmc, d_mcmc, e_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                 list(zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0))))
  poptEMCEE    = np.array([ a_mcmc[0], b_mcmc[0], c_mcmc[0], d_mcmc[0], e_mcmc[0] ])
  poptEMCEEcov = np.array([ np.max([a_mcmc[1], a_mcmc[2]]), np.max([b_mcmc[1], b_mcmc[2]]), np.max([c_mcmc[1], c_mcmc[2]]),
                            np.max([d_mcmc[1], d_mcmc[2]]), np.max([e_mcmc[1], e_mcmc[2]]) ])
  ##################

  JPL = True
  if JPL:
      from astropy.coordinates import solar_system_ephemeris
      solar_system_ephemeris.set('jpl') 

  T1 = Table.read('%s/Results/Weighted_Epochs.csv'%name)
  T2 = Table.read('%s/Results/All_Epochs.csv'%name)

  MJDs   = T1['MJD'].data
  Ys1    = T1['RA'].data
  Ys2    = T1['DEC'].data
  unYs1  = T1['SIGRA'].data
  unYs2  = T1['SIGDEC'].data

  XsALL  = T2['MJD'].data
  Ys1ALL = T2['RA'].data
  Ys2ALL = T2['DEC'].data
  

  # Uncertainty arrays in arcsec
  RA_Uncert  = np.sqrt(unYs1**2 + unYs1[0]**2) * d2a
  DEC_Uncert = np.sqrt(unYs2**2 + unYs2[0]**2) * d2a

  # Get a random sample of walkers for plotting
  Xs   = np.linspace(np.min(MJDs), np.max(MJDs), 2000)
  RAs  = np.zeros(len(Xs)) + Ys1[0]*d2a
  DECs = np.zeros(len(Xs)) + Ys2[0]*d2a

  print('Getting 300 random solutions')
  RAplots  = []
  DECplots = []
  for a, b, c, d, e in samples[np.random.randint(len(samples), size=300)]:
    poptEMCEEtest = np.array([ a, b, c, d, e ])
    RA, DEC = True, False
    RAplots.append(AstrometryFunc([RAs, DECs, Xs], *poptEMCEEtest, RA=RA, DEC=DEC))
    RA, DEC = False, True
    DECplots.append(AstrometryFunc([RAs, DECs, Xs], *poptEMCEEtest, RA=RA, DEC=DEC))

  ################################# Plot the emcee


  print('Plotting MCMC')
  fig = plt.figure(1, figsize=(7.1, 3.4*3/4.))
  cid = fig.canvas.mpl_connect('button_press_event', onclickclose)

  ax   = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
  ax11 = plt.subplot2grid((2, 2), (0, 1))
  ax22 = plt.subplot2grid((2, 2), (1, 1))


  fontsize = 7
  ms=2

  if poptEMCEE[-3] > poptEMCEE[-2]: # The PMRA > PMDEC

    ### PLOT THE RA

    ax.errorbar(MJDs, offset+(Ys1-Ys1[0])*d2a*np.cos(Ys2[0]*np.pi/180.), 
                yerr = RA_Uncert, marker='o',linestyle='None', color='b', ms=ms)
  
    ax.scatter(XsALL, offset+(Ys1ALL - Ys1[0])*d2a*np.cos(Ys2[0]*np.pi/180.), c='0.5', alpha=0.3, zorder=-10, s=4)

    Xs   = np.linspace(np.min(MJDs), np.max(MJDs), 2000)
    RAs  = np.zeros(len(Xs)) + Ys1[0]*d2a
    DECs = np.zeros(len(Xs)) + Ys2[0]*d2a
    RA, DEC = True, False
    RAplot = AstrometryFunc([RAs, DECs, Xs], *poptEMCEE, RA=RA, DEC=DEC)
    ax.plot(Xs, offset+RAplot, 'k-', lw=0.5)

    for RAplot2 in RAplots:
      ax.plot(Xs, offset+RAplot2, color="0.5", alpha=0.01)

    ax.text(np.min(MJDs)+50, offset1, r'$\Delta \alpha \cos \delta$', fontsize=fontsize)

    ### PLOT THE DEC

    ax.errorbar(MJDs, (Ys2-Ys2[0])*d2a, yerr = DEC_Uncert, marker='^',linestyle='None', ms=ms)

    ax.scatter(XsALL, (Ys2ALL - Ys2[0])*d2a, c='0.5', alpha=0.3, zorder=-10, s=4)

    RA, DEC = False, True
    DECplot = AstrometryFunc([RAs, DECs, Xs], *poptEMCEE, RA=RA, DEC=DEC)
    ax.plot(Xs, DECplot, 'k-', lw=0.5)

    for DECplot2 in DECplots:
      ax.plot(Xs, DECplot2, color="0.5", alpha=0.01)

    ax.text(np.min(MJDs)+50, offset2, r'$\Delta \delta$', fontsize=fontsize)


  else: # PMDEC > PMRA

    ### PLOT THE RA

    ax.errorbar(MJDs, (Ys1-Ys1[0])*d2a*np.cos(Ys2[0]*np.pi/180.), 
                yerr = RA_Uncert, marker='o',linestyle='None', color='b', ms=ms)
 
    ax.scatter(XsALL, (Ys1ALL - Ys1[0])*d2a*np.cos(Ys2[0]*np.pi/180.), c='0.5', alpha=0.3, zorder=-10, s=4)

    Xs   = np.linspace(np.min(MJDs), np.max(MJDs), 2000)
    RAs  = np.zeros(len(Xs)) + Ys1[0]*d2a
    DECs = np.zeros(len(Xs)) + Ys2[0]*d2a
    RA, DEC = True, False
    RAplot = AstrometryFunc([RAs, DECs, Xs], *poptEMCEE, RA=RA, DEC=DEC)
    ax.plot(Xs, RAplot, 'k-', lw=0.5)

    for RAplot2 in RAplots:
      ax.plot(Xs, RAplot2, color="0.5", alpha=0.01)

    ax.text(np.min(MJDs)+50, offset2, r'$\Delta \alpha \cos \delta$', fontsize=fontsize)


    ### PLOT THE DEC

    ax.errorbar(MJDs, offset+(Ys2-Ys2[0])*d2a, yerr = DEC_Uncert, marker='^',linestyle='None', ms=ms)

    ax.scatter(XsALL, offset+(Ys2ALL - Ys2[0])*d2a, c='0.5', alpha=0.3, zorder=-10, s=4)

    RA, DEC = False, True
    DECplot = AstrometryFunc([RAs, DECs, Xs], *poptEMCEE, RA=RA, DEC=DEC)
    ax.plot(Xs, offset+DECplot, 'k-', lw=0.5)

    for DECplot2 in DECplots:
      ax.plot(Xs, offset+DECplot2, color="0.5", alpha=0.01)

    ax.text(np.min(MJDs)+50, offset1, r'$\Delta \delta$', fontsize=fontsize)

  at = AnchoredText('MCMC Fit' + '\n' + r'$\mu_\alpha \cos \delta = %0.0f \pm %0.0f$ mas yr$^{-1}$'%(poptEMCEE[-3]*1e3, poptEMCEEcov[-3]*1e3) + '\n' + r'$\mu_\delta = %0.0f \pm %0.0f$ mas yr$^{-1}$'%(poptEMCEE[-2]*1e3, poptEMCEEcov[-2]*1e3)  + '\n' + r'$\pi = %0.0f \pm %0.0f$ mas'%(poptEMCEE[-1]*1e3, poptEMCEEcov[-1]*1e3),
                    prop=dict(size=8), frameon=False,
                    loc=Place,
                    )

  ax.add_artist(at)

  ax.set_ylabel(r'Motion + offset (arcsec)')
  ax.set_xlabel(r'MJD (day)')
  ax.minorticks_on()


  ################################################## Plot residuals without proper motion


  ax1 = ax11.twinx()
  ax2 = ax22.twinx()

  ax11.set_yticks([])
  ax11.set_xticks([])
  ax22.set_yticks([])

  ax1.errorbar(MJDs, (Ys1-Ys1[0])*d2a*np.cos(Ys2[0]*np.pi/180.) - poptEMCEE[-3]*(MJDs-MJDs[0])*d2y - poptEMCEE[0], 
                 yerr = RA_Uncert, marker='o',linestyle='None', color='b', ms=ms)
  ax1.scatter(XsALL, (Ys1ALL - Ys1[0])*d2a*np.cos(Ys2[0]*np.pi/180.) - poptEMCEE[-3]*(XsALL-XsALL[0])*d2y - poptEMCEE[0], 
                c='0.5', alpha=0.3, zorder=-10, s=4, marker='o')

  Xs2  = np.linspace(np.min(MJDs), np.max(MJDs), 2000)
  RAs  = np.zeros(len(Xs2)) + Ys1[0]*d2a
  DECs = np.zeros(len(Xs2)) + Ys2[0]*d2a

  RA, DEC = True, False
  RAplot = AstrometryFunc([RAs, DECs, Xs2], *poptEMCEE, RA=RA, DEC=DEC)
  ax1.plot(Xs2, RAplot - poptEMCEE[-3]*(Xs2-Xs2[0])*d2y - poptEMCEE[0], 'k-', lw=0.5)

  # Errorbar
  for RAplot2 in RAplots:
    ax1.plot(Xs2, RAplot2 - poptEMCEE[-3]*(Xs2-Xs2[0])*d2y - poptEMCEE[0], color="0.5", alpha=0.01)

  ####### Dec part

  ax2.errorbar(MJDs, (Ys2-Ys2[0])*d2a - poptEMCEE[-2]*(MJDs-MJDs[0])*d2y - poptEMCEE[1], 
               yerr = DEC_Uncert, marker='^',linestyle='None', ms=ms)
  ax2.scatter(XsALL, (Ys2ALL - Ys2[0])*d2a - poptEMCEE[-2]*(XsALL-XsALL[0])*d2y - poptEMCEE[1], 
              c='0.5', alpha=0.3, zorder=-10, s=4, marker='^')

 
  RA, DEC = False, True
  DECplot = AstrometryFunc([RAs, DECs, Xs2], *poptEMCEE, RA=RA, DEC=DEC)
  ax2.plot(Xs2, DECplot - poptEMCEE[-2]*(Xs2-Xs2[0])*d2y - poptEMCEE[1], 'k-', lw=0.5)

  # Errorbar
  for DECplot2 in DECplots:
    ax2.plot(Xs2, DECplot2 - poptEMCEE[-2]*(Xs2-Xs2[0])*d2y - poptEMCEE[1], color="0.5", alpha=0.01)

  ax1.set_ylabel(r'$\Delta\alpha$ (arcsec)')
  ax1.minorticks_on()
  ax2.set_ylabel(r'$\Delta\delta$ (arcsec)')
  ax22.set_xlabel(r'MJD (day)')
  ax2.minorticks_on()

  plt.suptitle('%s'%Name)

  fig.subplots_adjust(wspace=0.05, hspace=0.1)

  if PDFsave:
    plt.savefig('%s/Plots/FullSolution.pdf'%name, dpi=600, bbox_inches='tight')
  else:
    plt.savefig('%s/Plots/FullSolution.png'%name, dpi=600, bbox_inches='tight')

  plt.show()
  ##################################################
