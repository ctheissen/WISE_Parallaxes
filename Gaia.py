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
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
Vizier.ROW_LIMIT = -1

d2a  = 3600.
d2ma = 3600000.
d2y  = 1/365.25 

##################

def calibrate(MJDs, RAs, DECs, radecstr=None, ra0=None, dec0=None, radius=10, cache=True):

  #### Gaia calibration

  print('Gaia Calibration Starting (Experimental mode, this might take a while)')
  v    = Vizier(timeout=1000000, columns=['**', '+_r'], vizier_server='vizier.cfa.harvard.edu', catalog='I/350/gaiaedr3')
  v.ROW_LIMIT = -1
  c        = SkyCoord(ra0, dec0, unit=('deg','deg'), frame='icrs')
  results  = v.query_region(c, radius = radius*u.arcmin)
  #print(results)
  #print(results[0])
  #print(results[0].colnames)
  Gaia = results[0][np.where( (results[0]['Gmag']<18) & (results[0]['Plx']<5) & (results[0]['RPlx']>5) )]
  print('Number of Gaia Sources within %s arcmin: %s'%(radius, len(Gaia)) )

  selcols = "ra,dec,sigra,sigdec,w1mpro,w2mpro,w1sigmpro,w2sigmpro,mjd,qual_frame"

  t1w = Irsa.query_region(coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs'), 
                     catalog="allsky_4band_p1bs_psd", spatial="Cone", radius=radius * u.arcmin, cache=cache,
                     selcols=selcols)
  t2w = Irsa.query_region(coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs'), 
                         catalog="allsky_3band_p1bs_psd", spatial="Cone", radius=radius * u.arcmin, cache=cache,
                         selcols=selcols)
  if len(t2w) == 0:
    t2w = Irsa.query_region(coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs'), 
                           catalog="allsky_2band_p1bs_psd", spatial="Cone", radius=radius * u.arcmin, cache=cache,
                           selcols=selcols)
  t3w = Irsa.query_region(coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs'), 
                         catalog="neowiser_p1bs_psd", spatial="Cone", radius=radius * u.arcmin, cache=cache,
                         selcols=selcols)

  t00w  = vstack([t1w[np.where(t1w['qual_frame']!=0)], t2w[np.where(t2w['qual_frame']!=0)]], join_type='inner')
  t0w   = vstack([t00w, t3w[np.where(t3w['qual_frame']!=0)]], join_type='inner')
  tWG  = t0w
  print(len(tWG))

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
    catalog         = SkyCoord(ra=Gaia['RA_ICRS'], dec=Gaia['DE_ICRS'])
    idx, d2d, d3d   = c.match_to_catalog_3d(catalog)
    sep_constraint  = d2d < max_sep
    c_matches       = tWG[j][sep_constraint]
    catalog_matches = Gaia[idx[sep_constraint]]
    print('Number of matched sources: %s', len(c_matches))
    '''
    plt.scatter(Gaia['RA_ICRS'], Gaia['DE_ICRS'], label='Gaia', alpha=0.5)
    plt.scatter(tWG['ra'][j], tWG['dec'][j], label='WISE', alpha=0.5)
    #plt.scatter(c_matches['ra'], c_matches['dec'], label='matched', marker='x', alpha=0.5)
    plt.legend()
    plt.show()
    '''
    # Warp to new coordinates
    src = np.array([[a,b] for a,b in zip(c_matches['ra'], c_matches['dec'])])
    dst = np.array([[a,b] for a,b in zip(catalog_matches['RA_ICRS'], catalog_matches['DE_ICRS'])])
    from skimage import transform as tf
    tform1 = tf.estimate_transform('euclidean', src, dst)
    tform2 = tf.estimate_transform('polynomial', src, dst, 1)
    tform3 = tf.estimate_transform('polynomial', src, dst, 5)
    #print(tform)
    #print(tform(src))
    #print(tform(src)[:,0])
    newRAs1  = tform1(src)[:,0]
    newDECs1 = tform1(src)[:,1]
    newRAs2  = tform2(src)[:,0]
    newDECs2 = tform2(src)[:,1]
    newRAs3  = tform3(src)[:,0]
    newDECs3 = tform3(src)[:,1]

    '''
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(121)
    ax2 = fig2.add_subplot(122)
    bins = np.linspace(-0.001, 0.001, 30)
    ax1.hist(c_matches['ra'] - catalog_matches['RA_ICRS'], bins=bins, histtype='step', label='raw', alpha=0.5)
    ax1.hist(newRAs1 - catalog_matches['RA_ICRS'], bins=bins, histtype='step', label='calibrated (euc)', alpha=0.5)
    ax1.hist(newRAs2 - catalog_matches['RA_ICRS'], bins=bins, histtype='step', label='calibrated (poly)', alpha=0.5)
    ax2.hist(c_matches['dec'] - catalog_matches['DE_ICRS'], bins=bins, histtype='step', label='raw', alpha=0.5)
    ax2.hist(newDECs1 - catalog_matches['DE_ICRS'], bins=bins, histtype='step', label='calibrated (euc)', alpha=0.5)
    ax2.hist(newDECs2 - catalog_matches['DE_ICRS'], bins=bins, histtype='step', label='calibrated (poly)', alpha=0.5)
    plt.legend()
    '''
    print(np.std(c_matches['ra'] - catalog_matches['RA_ICRS']), np.std(newRAs1 - catalog_matches['RA_ICRS']), np.std(newRAs1 - catalog_matches['RA_ICRS'])/np.std(c_matches['ra'] - catalog_matches['RA_ICRS']), 'RA euc')
    print(np.std(c_matches['dec'] - catalog_matches['DE_ICRS']), np.std(newDECs1 - catalog_matches['DE_ICRS']), np.std(newDECs1 - catalog_matches['DE_ICRS'])/np.std(c_matches['dec'] - catalog_matches['DE_ICRS']), 'DEC euc')
    print(np.std(c_matches['ra'] - catalog_matches['RA_ICRS']), np.std(newRAs2 - catalog_matches['RA_ICRS']), np.std(newRAs2 - catalog_matches['RA_ICRS'])/np.std(c_matches['ra'] - catalog_matches['RA_ICRS']), 'RA poly 1')
    print(np.std(c_matches['dec'] - catalog_matches['DE_ICRS']), np.std(newDECs2 - catalog_matches['DE_ICRS']), np.std(newDECs2 - catalog_matches['DE_ICRS'])/np.std(c_matches['dec'] - catalog_matches['DE_ICRS']), 'DEC poly 1')
    print(np.std(c_matches['ra'] - catalog_matches['RA_ICRS']), np.std(newRAs3 - catalog_matches['RA_ICRS']), np.std(newRAs3 - catalog_matches['RA_ICRS'])/np.std(c_matches['ra'] - catalog_matches['RA_ICRS']), 'RA poly 3')
    print(np.std(c_matches['dec'] - catalog_matches['DE_ICRS']), np.std(newDECs3 - catalog_matches['DE_ICRS']), np.std(newDECs3 - catalog_matches['DE_ICRS'])/np.std(c_matches['dec'] - catalog_matches['DE_ICRS']), 'DEC poly 3')
    print()

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



 
