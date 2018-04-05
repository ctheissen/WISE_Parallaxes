from WISE_Parallax import MeasureParallax, PlotParallax

# Example using Luhman 16AB

Name   = 'Luhman 16AB'
#Name   = r'WISE J104915.57$-$531906.1' # You can give the name using LaTeX
radius = 30 # Search radius in arcseconds

radecstr = '10h49m15.57s -53d19m06.1s'
MeasureParallax(Name=Name, radecstr=radecstr, radius=radius)

"""
# Of you could use 
ra0, dec0 = 162.3147060, -53.3183847
Measure_Parallax(Name=Name, ra0=ra0, dec0=dec0, radius=radius)
"""

# Plot the parallax solution
offset = 1 # offsets between RA and DEC solutions in arcseconds
place  = 3 # Use 2 for upper left, 3 for lower left. Those are probably the only ones we care about.
PlotParallax(Name, place, offset, offset*2.5, -offset*5.5)

 
