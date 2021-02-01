import numpy as np
import sys, os, os.path, time, gc
from astropy.table import Table, vstack, hstack, join
import matplotlib.pyplot as plt
from astropy import units as u
import astropy.coordinates as coords
from astroquery.irsa import Irsa
Irsa.ROW_LIMIT = -1
Irsa.TIMEOUT = 60*10 # 10 minutes

d2a  = 3600.
d2ma = 3600000.
d2y  = 1/365.25 

##################

def GetPositionsAndEpochs(ra, dec, Epochs, radius=6):

  t1 = Irsa.query_region(coords.SkyCoord(ra, dec, unit=(u.deg,u.deg), frame='icrs'), 
                        catalog="allsky_4band_p1bs_psd", spatial="Cone", radius=radius * u.arcsec)
  t2 = Irsa.query_region(coords.SkyCoord(ra, dec, unit=(u.deg,u.deg), frame='icrs'), 
                         catalog="allsky_3band_p1bs_psd", spatial="Cone", radius=radius * u.arcsec)
  if len(t2) == 0:
    t2 = Irsa.query_region(coords.SkyCoord(ra, dec, unit=(u.deg,u.deg), frame='icrs'), 
                          catalog="allsky_2band_p1bs_psd", spatial="Cone", radius=radius * u.arcsec)
  t3 = Irsa.query_region(coords.SkyCoord(ra, dec, unit=(u.deg,u.deg), frame='icrs'), 
                          catalog="neowiser_p1bs_psd", spatial="Cone", radius=radius * u.arcsec)

  t00  = vstack([t1, t2], join_type='inner')
  t0   = vstack([t00, t3], join_type='inner')
  t = t0

  #### Find the epoch clusters
  Groups = []
  for epoch in Epochs:
    group  = np.where( t['mjd'] == epoch )
    if len(group[0]) != 0:
      Groups.append(group[0][0])

  #print(Epochs)
  #print(Groups)
  #print(len(Epochs),len(Groups))

  if len(Groups) >= 0.5*len(Epochs):
    #print('Good')
    #print(t['ra'][Groups].data, t['dec'][Groups].data, t['mjd'][Groups].data, t['w1mpro'][Groups].data )
    return t['ra'][Groups].data, t['dec'][Groups].data, t['mjd'][Groups].data, t['w1mpro'][Groups].data
  else:
    return [-9999], [-9999], [-9999], [-9999]




def GetRegistrators(name, Epochs, subepoch=0, ra0=None, dec0=None, radius=10, writeout=True, w1limit=14):

  print('Getting registration sources within %s arcmin'%radius)

  # First check if the file exits already
  #print(os.path.isfile('Calib_Sources.csv'))
  if os.path.isfile('%s/Results/Registration_Sources_Epoch%s.csv'%(name, subepoch)):
    print('Registration source file for this epoch already exists. Using current file.')
    C = Table.read('%s/Results/Registration_Sources_Epoch%s.csv'%(name, subepoch))

  else:

    ############### Grab the registration sources
    EnoughRegisrators = True
    while EnoughRegisrators:

      T = Irsa.query_region(coords.SkyCoord(ra0, dec0, unit=(u.deg,u.deg), frame='icrs'), 
                              catalog="allwise_p3as_psd", spatial="Cone", radius=radius * u.arcmin)

      print('Number of Potential Registration Sources: %s'%len(T))

      # Just get the first two cc flags (W1 and W2)
      ccFlg1 = np.array([e[0] for e in T['cc_flags'].data])
      ccFlg2 = np.array([e[1] for e in T['cc_flags'].data])
      
      Tnew = T[np.where( (T['w1sat'] == 0) & (T['w2sat'] == 0) & #(T['qual_frame'] != 0) & 
                         #(T['cc_flags'] == b'0000') & 
                         (ccFlg1 == '0') & #(ccFlg2 == '0') & 
                         (T['ext_flg'] == 0) & 
                         (T['w1mpro'] <= w1limit) & (T['w1snr'] >= 10) 
                         )]
      
      print('Number of Good Registration Sources: %s'%len(Tnew))
      if len(Tnew) < 40:
        radius += 1
        print("Not enough registration sources. Increasing search radius to %s arcmin."%radius)

      else: 
        EnoughRegisrators = False

    ############### Grab the registration source(s) in each epoch

    ## Check how many epochs the source is found in
    # Create the file for the first time
    sourcecount = 0
    for source, ra, dec, dist in zip(range(len(Tnew)), Tnew['ra'], Tnew['dec'], Tnew['dist']):
      print('Getting source: %s / %s'%(source+1, len(Tnew)))#,MJDs)

      if dist <= 6: # Don't do the target objects
        print('Skipping the target')
        continue 

      RAs, DECs, MJDs, W1MAGs = GetPositionsAndEpochs(ra, dec, Epochs)
      #print(RAs, DECs, MJDs, W1MAGs)

      if sourcecount == 0: 
        Twrite = Table([np.zeros(len(RAs))+source, W1MAGs, RAs, DECs, MJDs], names=['SOURCE','W1MAG','RA','DEC','MJD'])
        sourcecount += 1

      else: 
        Ttoss  = Table([np.zeros(len(RAs))+source, W1MAGs, RAs, DECs, MJDs], names=['SOURCE','W1MAG','RA','DEC','MJD'])
        Twrite = vstack([Twrite, Ttoss])

    Twrite.write('%s/Results/Registration_Sources_Epoch%s.csv'%(name, subepoch), overwrite=True)
    C = Table.read('%s/Results/Registration_Sources_Epoch%s.csv'%(name, subepoch))

    print('Done')
    #sys.exit()

  # Find the subepochs
  """
  GroupDates = []
  DateGrps = np.arange(-1, 30, 2)
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
  """
  RA_SHIFTS  = []
  DEC_SHIFTS = []

  # No shift for the first subepoch
  RA_SHIFTS.append(0.)
  DEC_SHIFTS.append(0.)

  for epoch in Epochs[1:]:

    #print('Epoch:', epoch)
    # Initialize empty lists for all the shifts in the epoch
    RADiffs  = []
    DECDiffs = []
    
    for source in np.unique(C['SOURCE']):

      subcat = C[np.where(C['SOURCE'] == source)]

      T01    = subcat[np.where(subcat['MJD'] == Epochs[0])]
      T02    = subcat[np.where(subcat['MJD'] == epoch)]

      # Check for source in both epochs
      length1 = len(T01['RA'])
      length2 = len(T02['RA'])

      if length1 != length2 or length1 == 0 or length2 == 0: 
        continue

      # Future epoch minus first epoch
      RAdiff  = T02['RA'].data[0] - T01['RA'].data[0]
      DECdiff = T02['DEC'].data[0] - T01['DEC'].data[0]
      #print('Diff:', RAdiff*d2ma, DECdiff*d2ma)
      RADiffs.append(RAdiff*d2ma) 
      DECDiffs.append(DECdiff*d2ma)

    RADiffs  = np.array(RADiffs)
    DECDiffs = np.array(DECDiffs)
    step=150
    bins = range(-300,300+step, step)

    plt.figure(1001)
    hist = plt.hist2d(RADiffs, DECDiffs, bins=bins, cmap=plt.cm.Greys)
    #plt.show()

    # Find the maximum of the histogram
    counts, xedges, yedges, im1 = hist

    x1 = np.where(counts == np.max(counts))[0][0]
    y1 = np.where(counts == np.max(counts))[1][0]

    SiftedRAdiff  = RADiffs[np.where(  (RADiffs >=  xedges[x1]) & (RADiffs  <= xedges[x1+1]) &
                                       (DECDiffs >= yedges[y1]) & (DECDiffs <= yedges[y1+1]) )]
    SiftedDECdiff = DECDiffs[np.where( (RADiffs >=  xedges[x1]) & (RADiffs  <= xedges[x1+1])&
                                       (DECDiffs >= yedges[y1]) & (DECDiffs <= yedges[y1+1]) )]
    SHIFT_RA  = np.median(SiftedRAdiff)
    SHIFT_DEC = np.median(SiftedDECdiff)

    RA_SHIFTS.append(SHIFT_RA)
    DEC_SHIFTS.append(SHIFT_DEC)

    plt.close(1001)

  if writeout:
    np.savetxt('%s/Results/ra_shifts_epoch%s.txt'%(name, subepoch), np.array(RA_SHIFTS)/d2ma)
    np.savetxt('%s/Results/dec_shifts_epoch%s.txt'%(name, subepoch), np.array(DEC_SHIFTS)/d2ma)

  return np.array(RA_SHIFTS)/d2ma, np.array(DEC_SHIFTS)/d2ma, radius



 
