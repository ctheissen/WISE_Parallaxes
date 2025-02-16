import numpy as np
import sys, os, os.path, time, gc
from astropy.table import Table, vstack, hstack, join
import matplotlib.pyplot as plt
from astropy import units as u
import astropy.coordinates as coords
from astropy.stats import sigma_clip
from astroquery.ipac.irsa import Irsa
Irsa.ROW_LIMIT = -1
#Irsa.TIMEOUT = 60*10 # 10 minutes

d2a  = 3600.
d2ma = 3600000.
d2y  = 1/365.25 

##################

def GetPositionsAndEpochs(ra, dec, Epochs, radius=6, cache=False):

  selcols = "ra,dec,sigra,sigdec,w1mpro,w2mpro,w1sigmpro,w2sigmpro,mjd,qual_frame"
  
  t1 = Irsa.query_region(coords.SkyCoord(ra, dec, unit=(u.deg,u.deg), frame='icrs'), 
                         catalog="allsky_4band_p1bs_psd", spatial="Cone", radius=radius * u.arcsec, cache=cache,
                         columns=selcols)
  t2 = Irsa.query_region(coords.SkyCoord(ra, dec, unit=(u.deg,u.deg), frame='icrs'), 
                         catalog="allsky_3band_p1bs_psd", spatial="Cone", radius=radius * u.arcsec, cache=cache,
                         columns=selcols)
  if len(t2) == 0:
    t2 = Irsa.query_region(coords.SkyCoord(ra, dec, unit=(u.deg,u.deg), frame='icrs'), 
                           catalog="allsky_2band_p1bs_psd", spatial="Cone", radius=radius * u.arcsec, cache=cache,
                           columns=selcols)
  t3 = Irsa.query_region(coords.SkyCoord(ra, dec, unit=(u.deg,u.deg), frame='icrs'), 
                         catalog="neowiser_p1bs_psd", spatial="Cone", radius=radius * u.arcsec, cache=cache,
                         columns=selcols)

  t00  = vstack([t1[np.where(t1['qual_frame']!=0)], t2[np.where(t2['qual_frame']!=0)]], join_type='inner')
  t0   = vstack([t00, t3[np.where(t3['qual_frame']!=0)]], join_type='inner')
  t    = t0

  #### Find the date clusters
  Groups = []
  for bottom, top in Epochs:
    group  = np.where( (t['mjd'] > bottom) & (t['mjd'] < top) )
    if len(group[0]) != 0:
      Groups.append(group[0])
    else: print('Skipping Epoch')

  MJDs   = []
  W1MAGs = []
  Ys1    = []
  Ys2    = []
  unYs1  = []
  unYs2  = []

  i = 0
  
  for group in Groups:

    filteredRA  = sigma_clip(t['ra'][group],  sigma=3, maxiters=None)
    filteredDEC = sigma_clip(t['dec'][group], sigma=3, maxiters=None)

    index = np.where( (~filteredRA.mask) & (~filteredDEC.mask) )[0]

    i += 1

    MJDs.append(np.ma.average(t['mjd'][group][index]))
    W1MAGs.append(np.ma.average(t['w1mpro'][group][index]))
    Ys1.append(np.ma.average(t['ra'][group][index], weights = 1/(t['sigra'][group][index]/d2a)**2))
    Ys2.append(np.ma.average(t['dec'][group][index], weights = 1/(t['sigdec'][group][index]/d2a)**2))

  return Ys1, Ys2, MJDs, W1MAGs



def GetCalibrators(name, Epochs, radecstr=None, ra0=None, dec0=None, radius=10, writeout=True, w1limit=14, cache=False, overwriteReg=False):

  print('Getting calibrators within %s arcmin'%radius)

  # First check if the file exits already
  #print(os.path.isfile('Calib_Sources.csv'))
  if os.path.isfile('%s/Results/Calib_Sources.csv'%name) and overwriteReg != True:
    print('Calibration source file already exists. Using current file.')
    C = Table.read('%s/Results/Calib_Sources.csv'%name)

  else:

    ############### Grab the calibration sources
    EngouhCalibrators = True
    while EngouhCalibrators:

      if radecstr != None:
        T = Irsa.query_region(coords.SkyCoord(radecstr, unit=(u.deg,u.deg), frame='icrs'), 
                              catalog="allwise_p3as_psd", spatial="Cone", radius=radius * u.arcmin, cache=cache,
                              columns="ra,dec,w1mpro,w1snr,w1sat,w2sat,cc_flags,ext_flg")

      elif ra0 != None and dec0 != None:
        T = Irsa.query_region(coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs'), 
                              catalog="allwise_p3as_psd", spatial="Cone", radius=radius * u.arcmin, cache=cache,
                              columns="ra,dec,w1mpro,w1snr,w1sat,w2sat,cc_flags,ext_flg")

      print('Number of Potential Calibration Sources: %s'%len(T))

      # Calculate the dist of each source from the target in arcseconds
      source1  = coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs')
      sources1 = coords.SkyCoord(T['ra'], T['dec'], unit=(u.deg,u.deg), frame='icrs')
      dist = source1.separation(sources1)
      T['dist'] = dist.arcsecond
      
      # Just get the first two cc flags (W1 and W2)
      ccFlg1 = np.array([e[0] for e in T['cc_flags'].data])
      ccFlg2 = np.array([e[1] for e in T['cc_flags'].data])

      Tnew = T[np.where( (T['w1sat'] == 0) & (T['w2sat'] == 0) & #(T['qual_frame'] != 0) & 
                         #(T['cc_flags'] == b'0000') & 
                         (ccFlg1 == '0') & (ccFlg2 == '0') & 
                         (T['ext_flg'] == 0) & 
                         (T['w1mpro'] <= w1limit) & (T['w1snr'] >= 10) 
                         )]
      
      print('Number of Good Calibration Sources: %s'%len(Tnew))
      if len(Tnew) < 40:
        radius += 1
        print("Not enough calibration sources. Increasing search radius to %s arcmin."%radius)
      else: 
        EngouhCalibrators = False

    ############### Grab the calibration source(s) in each epoch

    ## Check how many epochs the source is found in
    # Create the file for the first time
    sourcecount = 0
    for source, ra, dec, dist in zip(range(len(Tnew)), Tnew['ra'], Tnew['dec'], Tnew['dist']):

      print('Getting source: %s / %s'%(source+1, len(Tnew)), end='\r')#,MJDs)

      if dist <= 6: # Don't do the target object
        print('Skipping the target')
        continue 

      RAs, DECs, MJDs, W1MAGs = GetPositionsAndEpochs(ra, dec, Epochs, cache=cache)

      if sourcecount == 0: 
        Twrite = Table([np.zeros(len(RAs))+source, W1MAGs, RAs, DECs, MJDs], names=['SOURCE','W1MAG','RA','DEC','MJD'])
        sourcecount += 1

      else: 
        Ttoss  = Table([np.zeros(len(RAs))+source, W1MAGs, RAs, DECs, MJDs], names=['SOURCE','W1MAG','RA','DEC','MJD'])
        Twrite = vstack([Twrite, Ttoss])

    Twrite.write('%s/Results/Calib_Sources.csv'%name, overwrite=True)
    C = Table.read('%s/Results/Calib_Sources.csv'%name)

    print('\nDone\n')

  # Find Date demarcation points
  GroupDates = []
  DateGrps = np.arange(-1, 80, 2)
  for i in range(len(DateGrps)-1): 
    bottom = np.min(C['MJD']) + 365.25/4*DateGrps[i]
    top    = np.min(C['MJD']) + 365.25/4*DateGrps[i+1]
    group  = np.where( (C['MJD'] > bottom) & (C['MJD'] < top) )
    if len(group[0]) != 0:
      GroupDates.append(top)


  Epochs = []
  for g in range(len(GroupDates)):
    if g == 0:
      Epochs.append(C[np.where(  C['MJD'] < GroupDates[g])])
    else: 
      Epochs.append(C[np.where( (C['MJD'] > GroupDates[g-1]) & (C['MJD'] < GroupDates[g]) ) ])

  RA_SHIFTS  = []
  DEC_SHIFTS = []

  # No shift for the first epoch
  RA_SHIFTS.append(0.)
  DEC_SHIFTS.append(0.)

  for epoch in np.arange(len(Epochs[1:]))+1:
    # Initialize empty lists for all the shifts in the epoch
    RADiffs  = []
    DECDiffs = []
    
    for source in np.unique(C['SOURCE']):

      T01 = Epochs[0]
      T02 = Epochs[epoch]

      # Check for source in both epochs
      length1 = len(T01['RA'][ np.where(T01['SOURCE'] == source)])
      length2 = len(T02['RA'][ np.where(T02['SOURCE'] == source)])
      #print(length1, length2)
      if length1 != length2 or length1 == 0: 
        continue

      # Future epoch minus first epoch
      #print(T02['RA'][ np.where(T02['SOURCE'] == source)].data)
      RAdiff  = T02['RA'][ np.where(T02['SOURCE'] == source)].data[0] - T01['RA'][ np.where(T01['SOURCE'] == source)].data[0]
      DECdiff = T02['DEC'][np.where(T02['SOURCE'] == source)].data[0] - T01['DEC'][np.where(T01['SOURCE'] == source)].data[0]
      RADiffs.append(RAdiff*d2ma) 
      DECDiffs.append(DECdiff*d2ma)

    RADiffs  = np.array(RADiffs)
    DECDiffs = np.array(DECDiffs)
    step=100
    bins = range(-500,500+step, step)

    plt.figure(1001)
    hist = plt.hist2d(RADiffs, DECDiffs, bins=bins, cmap=plt.cm.Greys, range=((-500,500),(-500,500)))
    #plt.show()

    # Find the maximum of the histogram
    counts, xedges, yedges, im1 = hist

    x1 = np.where(counts == np.ma.max(counts))[0][0]
    y1 = np.where(counts == np.ma.max(counts))[1][0]

    SiftedRAdiff  = RADiffs[np.where(  (RADiffs >=  xedges[x1]) & (RADiffs  <= xedges[x1+1]) &
                                       (DECDiffs >= yedges[y1]) & (DECDiffs <= yedges[y1+1]) )]
    SiftedDECdiff = DECDiffs[np.where( (RADiffs >=  xedges[x1]) & (RADiffs  <= xedges[x1+1])&
                                       (DECDiffs >= yedges[y1]) & (DECDiffs <= yedges[y1+1]) )]
    SHIFT_RA  = np.ma.median(SiftedRAdiff)
    SHIFT_DEC = np.ma.median(SiftedDECdiff)

    RA_SHIFTS.append(SHIFT_RA)
    DEC_SHIFTS.append(SHIFT_DEC)
    
    plt.close(1001)

  if writeout:
    np.savetxt('%s/Results/ra_shifts.txt'%name, np.array(RA_SHIFTS)/d2ma)
    np.savetxt('%s/Results/dec_shifts.txt'%name, np.array(DEC_SHIFTS)/d2ma)

  return np.array(RA_SHIFTS)/d2ma, np.array(DEC_SHIFTS)/d2ma
