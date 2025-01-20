#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.aperture import CircularAnnulus, CircularAperture, RectangularAnnulus, RectangularAperture
from photutils.aperture import ApertureStats
from astroquery.sdss import SDSS
from astropy.stats import SigmaClip
from scipy.signal import medfilt

template = SDSS.get_spectral_template('qso')[0] #
first_pixel_wavelength = template[0].header['COEFF0']
delta_lambda = template[0].header['COEFF1']
flux_sdss = template[0].data[0]
waves_sdss = 10**(first_pixel_wavelength + delta_lambda*np.arange(flux_sdss.size))

arms = ['Blue', 'Red-']

hdul_b = fits.open('data/SMSS 072242.59-390159.6/Spectra/OBK-943744-WiFeS-'+arms[0]+'-UT20241224T153733-5.cube.fits')
hdul_r = fits.open('data/SMSS 072242.59-390159.6/Spectra/OBK-943744-WiFeS-'+arms[1]+'-UT20241224T153733-5.cube.fits')
print(hdul_b.info())
print(hdul_r.info())

hdr_b = hdul_b[0].header
data_b = hdul_b[0].data
Nb, Ny, Nx = data_b.shape
print(Nb, Ny, Nx)
l0b = float(hdr_b['CRVAL3'])
dlb = float(hdr_b['CDELT3'])
lb = l0b + np.arange(Nb)*dlb

hdr_r = hdul_r[0].header
data_r = hdul_r[0].data
Nr, Ny, Nx = data_r.shape
print(Nr, Ny, Nx)
l0r = float(hdr_r['CRVAL3'])
dlr = float(hdr_r['CDELT3'])
lr = l0r + np.arange(Nr)*dlr

xc, yc = 12, 21
positions = [(xc, yc)]
aperture = CircularAperture(positions, r=2.0)
annulus_aperture = CircularAnnulus(positions, r_in=7, r_out=10)

#aperture = RectangularAperture(positions, 1.0, 1.0)
#annulus_aperture = RectangularAnnulus(positions, w_in=7.5, w_out=10.5, h_out=10.5)


sigclip = SigmaClip(sigma=3.0, maxiters=10)
sky_b = np.zeros(Nb)
specb = np.zeros(Nb)
for k in range(Nb):
	data = data_b[k,:,:]
	aperstats = ApertureStats(data, annulus_aperture, sigma_clip=sigclip)
	sky_b[k] = aperstats.median[0]
	data = data_b[k,:,:] - sky_b[k]
	aperstats = ApertureStats(data, aperture)
	specb[k] = aperstats.sum[0]
	
sky_r = np.zeros(Nr)
specr = np.zeros(Nr)
for k in range(Nr):
	data = data_r[k,:,:]
	aperstats = ApertureStats(data, annulus_aperture, sigma_clip=sigclip)
	sky_r[k] = aperstats.median[0]
	data = data_r[k,:,:] - sky_r[k] 
	aperstats = ApertureStats(data, aperture)
	specr[k] = aperstats.sum[0]

MgII = [2796.3543, 2803.5315]
zs = 2.096
ksdss = specr.mean()/flux_sdss.mean()

ymin, ymax = 0, max(specb)*0.5
lmin, lmax = 3500, 9300 #3358, 9558
print('lmin, lmax', lmin, lmax) 

indb = np.logical_and(lb > 3500, lb < 5850)
indr = np.logical_and(lr > 5450, lr < 9300)

specb_med = medfilt(specb, 3)
specr_med = medfilt(specr, 3)

zab = 1.205
Nab = 8
ab = np.array([2344,2374,2382,2585.9,2599.4,2795.5,2802.7,2852.1])*1.00028
abname = ['','','','','','','']
print('absorption lines rest', ab)

plt.figure(figsize=(12,8))
y1 = 0.3E-16; y2 = 0.5E-16
for i in range(Nab):
	plt.plot([ab[i]*(1+zab),ab[i]*(1+zab)],[y1,y2], 'C3')
plt.plot([ab[0]*(1+zab),ab[7]*(1+zab)], [y1,y1], 'C3')
plt.text(5700, 0.05E-16, 'Fe,Mg $z_{abs}$ = '+str(zab), ha='center', fontsize=14, va='bottom', color='C3')

plt.plot(lb[indb], specb_med[indb], 'C0', label='blue arm')
plt.plot(lr[indr], specr_med[indr], 'C1', label='red arm')
plt.plot(waves_sdss*(1+zs), flux_sdss*ksdss, 'C2', label='SDSS quasar template, z='+str(zs))
plt.xlabel('Wavelength [Angstrom]', size='x-large')
plt.ylabel('Flux', size='x-large')
plt.grid(alpha=0.5)
plt.legend(fontsize='x-large')

plt.xlim(lmin, lmax)
plt.ylim(ymin, ymax)
plt.savefig('cube_view.png', dpi=300)
plt.show() 
