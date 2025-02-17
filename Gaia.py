import numpy as np
import sys, os, os.path, time, gc
from astropy.table import Table, vstack, hstack, join
import matplotlib.pyplot as plt
from astropy import units as u
import astropy.coordinates as coords
from astropy.stats import sigma_clip
from astroquery.ipac.irsa import Irsa
import warnings
warnings.simplefilter('ignore')
Irsa.ROW_LIMIT = -1
Irsa.TIMEOUT = 60*10 # 10 minutes
#from astroquery.vizier import Vizier
from astroquery.gaia import Gaia as aGaia
from astropy.coordinates import SkyCoord
from skimage import transform as tf
#Vizier.ROW_LIMIT = -1
aGaia.ROW_LIMIT = -1

d2a  = 3600.
d2ma = 3600000.
d2y  = 1/365.25 

##################

def calibrate(MJDs, RAs, DECs, radecstr=None, ra0=None, dec0=None, radius=10, cache=True, plot=False):

  #### Gaia calibration

  print('Gaia Registration Starting (Experimental mode, this might take a while)')
  #v    = Vizier(timeout=1000000, columns=['**', '+_r'], vizier_server='vizier.cfa.harvard.edu', catalog='I/355/gaiadr3')
  #v.ROW_LIMIT = -1
  #print(ra0, dec0)
  #print(radius)
  c        = SkyCoord(ra0, dec0, unit=('deg','deg'), frame='icrs')
  #results  = v.query_region(c, radius = radius*u.arcmin)

  EnoughGaia = True
  while EnoughGaia:

    results  = aGaia.cone_search_async(c, radius=radius*u.arcmin).get_results()
    #print('1', results)
    #print('2', results[0])
    #print('3', results[0].colnames)
    GaiaT = results[np.where( (results['phot_g_mean_mag']<20) & (results['parallax']<1) )]# & (results['parallax_over_error']>5) )]
    print('Number of Gaia Sources within %s arcmin: %s'%(radius, len(GaiaT)) )

    if len(GaiaT) < 20:
      radius += 1
      print("Not enough registration sources. Increasing search radius to %s arcmin."%radius)
    else: 
      EnoughGaia = False

  print("Querying WISE for the Gaia sources.")

  selcols = "ra,dec,sigra,sigdec,w1mpro,w2mpro,w1sigmpro,w2sigmpro,mjd,qual_frame"

  t1w = Irsa.query_region(coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs'), 
                     catalog="allsky_4band_p1bs_psd", spatial="Cone", radius=radius * u.arcmin, cache=cache,
                     columns=selcols)
  t2w = Irsa.query_region(coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs'), 
                         catalog="allsky_3band_p1bs_psd", spatial="Cone", radius=radius * u.arcmin, cache=cache,
                         columns=selcols)
  if len(t2w) == 0:
    t2w = Irsa.query_region(coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs'), 
                           catalog="allsky_2band_p1bs_psd", spatial="Cone", radius=radius * u.arcmin, cache=cache,
                           columns=selcols)
  t3w = Irsa.query_region(coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs'), 
                         catalog="neowiser_p1bs_psd", spatial="Cone", radius=radius * u.arcmin, cache=cache,
                         columns=selcols)

  t00w  = vstack([t1w[np.where(t1w['qual_frame']!=0)], t2w[np.where(t2w['qual_frame']!=0)]], join_type='inner')
  t0w   = vstack([t00w, t3w[np.where(t3w['qual_frame']!=0)]], join_type='inner')
  tWG  = t0w
  #print(len(tWG))

  #print(t1)
  #print(np.unique(t1['mjd']))
  #plt.scatter(Gaia['RA_ICRS'], Gaia['DE_ICRS'], label='Gaia', alpha=0.5)
  newRAarray  = []
  newDECarray = []
  for mjd, raOld, decOld in zip(MJDs, RAs, DECs):

    j               = np.where(tWG['mjd'] == mjd)[0]
    if len(j) < 10: continue # This mjd has few or no observations

    max_sep         = 1.0 * u.arcsec # Cross match the catalogs within a small radius
    
    c               = SkyCoord(ra=tWG['ra'][j], dec=tWG['dec'][j])
    catalog         = SkyCoord(ra=GaiaT['ra'], dec=GaiaT['dec'])
    idx, d2d, d3d   = c.match_to_catalog_3d(catalog)
    sep_constraint  = d2d < max_sep
    c_matches       = tWG[j][sep_constraint]
    catalog_matches = GaiaT[idx[sep_constraint]]
    print('Number of matched sources: %s'%len(c_matches))
    # Warp to new coordinates
    src = np.array([[a,b] for a,b in zip(c_matches['ra'], c_matches['dec'])])
    dst = np.array([[a,b] for a,b in zip(catalog_matches['ra'], catalog_matches['dec'])])
    
    tform0 = tf.estimate_transform('euclidean', src, dst)
    tform1 = tf.estimate_transform('polynomial', src, dst, 1)
    tform2 = tf.estimate_transform('polynomial', src, dst, 2)
    tform3 = tf.estimate_transform('polynomial', src, dst, 3)
    #print(tform)
    #print(tform(src))
    #print(tform(src)[:,0])
    newRAs0  = tform0(src)[:,0]
    newDECs0 = tform0(src)[:,1]
    newRAs1  = tform1(src)[:,0]
    newDECs1 = tform1(src)[:,1]
    newRAs2  = tform2(src)[:,0]
    newDECs2 = tform2(src)[:,1]
    newRAs3  = tform3(src)[:,0]
    newDECs3 = tform3(src)[:,1]
      
    print('%0.3f'%(np.std(c_matches['ra'] - catalog_matches['ra'])*d2ma),   '%0.3f'%(np.std(newRAs0 - catalog_matches['ra'])*d2ma),   '%0.3f'%(np.std(newRAs0 - catalog_matches['ra'])/np.std(c_matches['ra'] - catalog_matches['ra'])),     'RA euc')
    print('%0.3f'%(np.std(c_matches['dec'] - catalog_matches['dec'])*d2ma), '%0.3f'%(np.std(newDECs0 - catalog_matches['dec'])*d2ma), '%0.3f'%(np.std(newDECs0 - catalog_matches['dec'])/np.std(c_matches['dec'] - catalog_matches['dec'])), 'DEC euc')
    print('%0.3f'%(np.std(c_matches['ra'] - catalog_matches['ra'])*d2ma),   '%0.3f'%(np.std(newRAs1 - catalog_matches['ra'])*d2ma),   '%0.3f'%(np.std(newRAs1 - catalog_matches['ra'])/np.std(c_matches['ra'] - catalog_matches['ra'])),     'RA poly 1')
    print('%0.3f'%(np.std(c_matches['dec'] - catalog_matches['dec'])*d2ma), '%0.3f'%(np.std(newDECs1 - catalog_matches['dec'])*d2ma), '%0.3f'%(np.std(newDECs1 - catalog_matches['dec'])/np.std(c_matches['dec'] - catalog_matches['dec'])), 'DEC poly 1')
    print('%0.3f'%(np.std(c_matches['ra'] - catalog_matches['ra'])*d2ma),   '%0.3f'%(np.std(newRAs2 - catalog_matches['ra'])*d2ma),   '%0.3f'%(np.std(newRAs2 - catalog_matches['ra'])/np.std(c_matches['ra'] - catalog_matches['ra'])),     'RA poly 2')
    print('%0.3f'%(np.std(c_matches['dec'] - catalog_matches['dec'])*d2ma), '%0.3f'%(np.std(newDECs2 - catalog_matches['dec'])*d2ma), '%0.3f'%(np.std(newDECs2 - catalog_matches['dec'])/np.std(c_matches['dec'] - catalog_matches['dec'])), 'DEC poly 2')
    print('%0.3f'%(np.std(c_matches['ra'] - catalog_matches['ra'])*d2ma),   '%0.3f'%(np.std(newRAs3 - catalog_matches['ra'])*d2ma),   '%0.3f'%(np.std(newRAs3 - catalog_matches['ra'])/np.std(c_matches['ra'] - catalog_matches['ra'])),     'RA poly 3')
    print('%0.3f'%(np.std(c_matches['dec'] - catalog_matches['dec'])*d2ma), '%0.3f'%(np.std(newDECs3 - catalog_matches['dec'])*d2ma), '%0.3f'%(np.std(newDECs3 - catalog_matches['dec'])/np.std(c_matches['dec'] - catalog_matches['dec'])), 'DEC poly 3')
    print()

    if plot:
      fig3 = plt.figure(figsize=(12,4))
      ax1  = fig3.add_subplot(131)
      ax2  = fig3.add_subplot(132)
      ax3  = fig3.add_subplot(133)
    
      ax1.scatter(GaiaT['ra'], GaiaT['dec'], label='Gaia', alpha=0.5)
      ax1.scatter(tWG['ra'][j], tWG['dec'][j], label='WISE', alpha=0.5)
      #plt.scatter(c_matches['ra'], c_matches['dec'], label='matched', marker='x', alpha=0.5)
      ax1.legend()
      #plt.show()
    
      bins = np.linspace(-1000, 1000, 20)
      ax2.hist((c_matches['ra'] - catalog_matches['ra'])*d2ma, bins=bins, histtype='step', label='raw', alpha=0.5)
      ax2.hist((newRAs0 - catalog_matches['ra'])*d2ma, bins=bins, histtype='step', label='calibrated (euc)', alpha=0.5)
      ax2.hist((newRAs1 - catalog_matches['ra'])*d2ma, bins=bins, histtype='step', label='calibrated (poly)', alpha=0.5)
      #ax2.hist((newRAs2 - catalog_matches['ra'])*d2ma, bins=bins, histtype='step', label='calibrated (poly2)', alpha=0.5)
      ax3.hist((c_matches['dec'] - catalog_matches['dec'])*d2ma, bins=bins, histtype='step', label='raw', alpha=0.5)
      ax3.hist((newDECs0 - catalog_matches['dec'])*d2ma, bins=bins, histtype='step', label='calibrated (euc)', alpha=0.5)
      ax3.hist((newDECs1 - catalog_matches['dec'])*d2ma, bins=bins, histtype='step', label='calibrated (poly1)', alpha=0.5)
      #ax3.hist((newDECs2 - catalog_matches['dec'])*d2ma, bins=bins, histtype='step', label='calibrated (poly2)', alpha=0.5)
      ax2.set_xlabel('R.A. separation [WISE-Gaia] (mas)')
      ax3.set_xlabel('Dec. separation [WISE-Gaia] (mas)')
      ax2.axvline(0, color='k', ls=':', lw=0.5)
      ax3.axvline(0, color='k', ls=':', lw=0.5)
      ax2.legend()
      plt.show()

    #print(raOld, decOld)
    #print(src)
    #print(src.shape)
    oldCoords = np.array([[raOld, decOld]])
    #print(oldCoords.shape)
    newCoords = tform2(oldCoords)
    #print('New Coords:', newCoords)
    newRAarray.append(newCoords[0][0])
    newDECarray.append(newCoords[0][1])
    #print('New RA:', newRAarray)
    #print('New DEC:', newDECarray)

    #plt.show()
    #sys.exit()
  #plt.scatter(t1['ra'], t1['dec'], label='WISE', alpha=0.5)
  #plt.legend()
  '''
  plt.figure(9210)
  plt.scatter(t['ra'][slice1][slice2][group], t['dec'][slice1][slice2][group], label='Raw', alpha=0.5)
  plt.scatter(newRAarray, newDECarray, label='Calibrated', alpha=0.5)
  plt.legend()
  plt.show()
  sys.exit()
  '''
  return newRAarray, newDECarray
