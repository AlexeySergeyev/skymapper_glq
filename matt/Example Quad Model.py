#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:32:10 2025

@author: matthewstanton
"""
# %%
import numpy as np
import copy
import matplotlib.pyplot as plt
import lenstronomy
import imageio
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import make_lupton_rgb, simple_norm
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats

from photutils.detection import DAOStarFinder

# %%

df_candidates = pd.read_csv('../data//candidates2.csv')
gr = df_candidates.groupby('source_id_smss')
source_id = 30885280
group = gr.get_group(source_id)
band = 'r'
hdul_r = fits.open("../data/30885280_r.fits")
w_r = WCS(hdul_r[0].header)
data_r = hdul_r[0].data
header_r = hdul_r[0].header
hdul_r.close()

mean, median, std = sigma_clipped_stats(data_r, sigma=3.0)  
daofind = DAOStarFinder(fwhm=3.0, threshold=10.*std)  
data_r -= median
sources = daofind(data_r)
for col in sources.colnames:  
    sources[col].info.format = '%.4g'
#print(sources)
x_centroid = np.array(sources['xcentroid'])
y_centroid = np.array(sources['ycentroid'])

plt.figure(figsize=(5, 5))
norm = simple_norm(data_r, 'sqrt', percent=99.5)
plt.imshow(data_r, origin='lower', cmap='gray', norm=norm)
plt.grid(True, color='gray', linestyle='--')
coords = SkyCoord(ra=group.iloc[0]['ra_smss'], 
                  dec=group.iloc[0]['dec_smss'], 
                  unit=(u.deg, u.deg))
plt.show()
# %%

src = Cutout2D(data_r.data, (56, 54), size=(20, 20))
simulated_image = src.data

center_pixel_x = 10
center_pixel_y = 10
x_image = np.array([0.345,  -0.206,  -0.849, 0.627])
y_image = np.array([ 0.598, 0.744,  -0.474, -0.465])

background_rms = .5  # background noise per pixel
exp_time = 90  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
numPix = 20  # cutout pixel size
deltaPix = 0.26  # pixel size in arcsec (area per pixel = deltaPix**2)
fwhm = 0.69  # full width half max of PSF


plt.figure(figsize=(5, 5))
norm = simple_norm(data_r, 'sqrt', percent=99.5)
plt.imshow(src.data, origin='lower', cmap='gray', norm=norm)
plt.grid(True, color='gray', linestyle='--')
x_pix = center_pixel_x + x_image / deltaPix
y_pix = center_pixel_y + y_image / deltaPix
x_angular = x_pix * deltaPix
y_angular = y_pix * deltaPix
print(x_angular, y_angular)

plt.scatter(x_pix, 
            y_pix)

plt.show()

# %%
# background_rms = .5  # background noise per pixel
# exp_time = 90  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
# numPix = 20  # cutout pixel size
# deltaPix = 0.26  # pixel size in arcsec (area per pixel = deltaPix**2)
# fwhm = 0.69  # full width half max of PSF

# Load the image (convert to grayscale if necessary)
#image_path = "/Users/matthewstanton/Desktop/Screenshot 2025-01-13 at 3.58.04 pm.png"
#simulated_image = imageio.imread(image_path)

# If the image is RGB, convert to grayscale (e.g., by averaging color channels)
#if simulated_image.ndim == 3:  # RGB image
#    simulated_image = simulated_image.mean(axis=-1)
    
#simulated_image = simulated_image / simulated_image.max()
#print(simulated_image)

lens_model_list = ['EPL', 'SHEAR']
source_model_list = ['SERSIC_ELLIPSE']
lens_light_model_list = ['SERSIC']
point_source_list = ['LENSED_POSITION']

kwargs_model = {'lens_model_list': lens_model_list,
                'source_light_model_list': source_model_list,
                'lens_light_model_list': lens_light_model_list,
                'point_source_model_list': point_source_list,
                'additional_images_list': [False],
                'fixed_magnification_list': [False]}

kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

num_source_model = len(source_model_list)

kwargs_constraints = {'joint_source_with_point_source': [[0, 0]],
                      'num_point_source_list': [4],
                      'solver_type': 'PROFILE_SHEAR',  # 'PROFILE', 'PROFILE_SHEAR', 'ELLIPSE', 'CENTER', 'NONE'
                      }

prior_lens = []

kwargs_likelihood = {'check_bounds': True,
                     'force_no_add_image': False,
                     'source_marg': False,
                     'image_position_uncertainty': 0.004,
                     'source_position_tolerance': 0.001,
                     'source_position_sigma': 0.001,
                     'prior_lens': prior_lens
                     }

# %%

# Initialising Data Class

from lenstronomy.Data.imaging_data import ImageData
import lenstronomy.Util.simulation_util as sim_util

kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, background_rms)
kwargs_data['image_data'] = simulated_image
#kwargs_data['image_data'] = src
kwargs_data['ra_at_xy_0'] = 0
kwargs_data['dec_at_xy_0'] = 0
data_class = ImageData(**kwargs_data)


# Initialising PSF Class

from lenstronomy.Data.psf import PSF

kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 5}
psf_class = PSF(**kwargs_psf)

image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
multi_band_list = [image_band]
kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}

# initial guess of non-linear parameters, we chose different starting parameters than the truth #
kwargs_lens_init = [{'theta_E': 1.0, 'e1': 0, 'e2': 0, 'gamma': 2., 'center_x': 0., 'center_y': 0},
    {'gamma1': 0, 'gamma2': 0}]
kwargs_source_init = [{'R_sersic': 0.03, 'n_sersic': 4., 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
kwargs_lens_light_init = [{'R_sersic': 0.1, 'n_sersic': 4, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
#kwargs_ps_init = [{'ra_image': np.array([-0.48, -0.45, 0.58, 0.73]), 'dec_image': [-0.82, 0.61, 0.41, -0.22]}]
kwargs_ps_init = [{'ra_image': x_image, 'dec_image': y_image}]

# initial spread in parameter estimation #
kwargs_lens_sigma = [{'theta_E': 0.3, 'e1': 0.2, 'e2': 0.2, 'gamma': .2, 'center_x': 0.1, 'center_y': 0.1},
    {'gamma1': 0.1, 'gamma2': 0.1}]
kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': .5, 'center_x': .1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2}]
kwargs_lens_light_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.2, 'e1': 0.1, 'e2': 0.1, 'center_x': .1, 'center_y': 0.1}]
kwargs_ps_sigma = [{'ra_image': [0.02] * 4, 'dec_image': [0.02] * 4}]

# hard bound lower limit in parameter space #
kwargs_lower_lens = [{'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'center_x': -10., 'center_y': -10},
    {'gamma1': -0.5, 'gamma2': -0.5}]
kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': .5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10}]
kwargs_lower_lens_light = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10}]
#kwargs_lower_ps = [{'ra_image': -10 * np.ones_like(np.array([-0.48, -0.45, 0.58, 0.73])), 'dec_image': -10 * np.ones_like(np.array([-0.48, -0.45, 0.58, 0.73]))}]
kwargs_lower_ps = [{'ra_image': -10 * np.ones_like(x_image), 'dec_image': -10 * np.ones_like(y_image)}]

# hard bound upper limit in parameter space #
kwargs_upper_lens = [{'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10., 'center_y': 10},
    {'gamma1': 0.5, 'gamma2': 0.5}]
kwargs_upper_source = [{'R_sersic': 10, 'n_sersic': 5., 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
kwargs_upper_lens_light = [{'R_sersic': 10, 'n_sersic': 5., 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
#kwargs_upper_ps = [{'ra_image': 10 * np.ones_like(np.array([-0.48, -0.45, 0.58, 0.73])), 'dec_image': 10 * np.ones_like(np.array([-0.48, -0.45, 0.58, 0.73]))}]
kwargs_upper_ps = [{'ra_image': 10 * np.ones_like(x_image), 'dec_image': 10 * np.ones_like(y_image)}]

# keeping parameters fixed
kwargs_lens_fixed = [{}, {'ra_0': 0, 'dec_0': 0}]
kwargs_source_fixed = [{}]
kwargs_lens_light_fixed = [{}]
kwargs_ps_fixed = [{}]

lens_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens, kwargs_upper_lens]
source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source, kwargs_upper_source]
lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light, kwargs_upper_lens_light]
ps_params = [kwargs_ps_init, kwargs_ps_sigma, kwargs_ps_fixed, kwargs_lower_ps, kwargs_upper_ps]

kwargs_params = {'lens_model': lens_params,
                'source_model': source_params,
                'lens_light_model': lens_light_params,
                'point_source_model': ps_params}

# %%

from lenstronomy.Workflow.fitting_sequence import FittingSequence
fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)

fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 10, 'n_iterations': 20}],
                       ['MCMC', {'n_burn': 20, 'n_run': 20, 'walkerRatio': 4, 'sigma_scale': .1}]
        ]

chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
kwargs_result = fitting_seq.best_fit()

#print(kwargs_result)



from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Plots import chain_plot
modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat")

param_class = fitting_seq.param_class
print(param_class.num_param())
#print(chain_list)

for i in range(len(chain_list)):
    chain_plot.plot_chain_list(chain_list, i)

f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

modelPlot.data_plot(ax=axes[0,0])
modelPlot.model_plot(ax=axes[0,1])
modelPlot.normalized_residual_plot(ax=axes[0,2], v_min=-6, v_max=6)
modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100)
modelPlot.convergence_plot(ax=axes[1, 1], v_max=1)
modelPlot.magnification_plot(ax=axes[1, 2])
f.tight_layout()
#f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
#plt.show()

f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

modelPlot.decomposition_plot(ax=axes[0,0], text='Lens light', lens_light_add=True, unconvolved=True)
modelPlot.decomposition_plot(ax=axes[1,0], text='Lens light convolved', lens_light_add=True)
modelPlot.decomposition_plot(ax=axes[0,1], text='Source light', source_add=True, unconvolved=True)
modelPlot.decomposition_plot(ax=axes[1,1], text='Source light convolved', source_add=True)
modelPlot.decomposition_plot(ax=axes[0,2], text='All components', source_add=True, lens_light_add=True, unconvolved=True)
modelPlot.decomposition_plot(ax=axes[1,2], text='All components convolved', source_add=True, lens_light_add=True, point_source_add=True)
f.tight_layout()
f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
plt.show()
#print(kwargs_result)















# %%
