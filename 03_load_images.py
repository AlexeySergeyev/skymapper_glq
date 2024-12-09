import os
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file

from dotenv import load_dotenv
load_dotenv()

from pyvo.dal import sia
from dl import authClient as ac

token = ac.login(os.getenv('DATALAB_LOGIN'), os.getenv('DATALAB_PASSWORD'))
print(token)

DEF_ACCESS_URL = "https://datalab.noirlab.edu/sia"
svc = sia.SIAService(DEF_ACCESS_URL)

df_candidates = pd.read_csv('data/candidates/candidates2.csv')
gr = df_candidates.groupby('source_id_smss')
fov = 30.0/3600 # in degrees

# A little function to download the deepest stacked images
def download_deepest_image(ra,dec,svc=sia.SIAService('https://datalab.noirlab.edu/sia'),
                           fov=30.0/3600, imgTable=None, band='r'):
    if imgTable is None:
        imgTable = svc.search((ra,dec), (fov/np.cos(dec*np.pi/180), fov), verbosity=2).to_table().to_pandas()
    
    sel0 = imgTable['obs_bandpass'].astype(str) == band
    print("The full image list contains", len(imgTable[sel0]), "entries with bandpass="+band)

    # sel = sel0 & ((imgTable['proctype'] == 'Stack') & (imgTable['prodtype'] == 'image')) # basic selection
    sel = sel0 & (imgTable['proctype'] == 'Resampled') & (imgTable['prodtype'] == 'image') # basic selection
    df = imgTable[sel] # select
    # print(imgTable[sel])
    if (len(df)>0):
        print(f"Max_deepness: {np.max(df['magzero'].astype('float'))}")
        idx = df['magzero'].astype('float').idxmax() # pick image with longest exposure time
        row = df.loc[idx, :] # pick image with longest exposure time
        url = row['access_url'] # get the download URL
        print ('downloading deepest ' + band + ' image...')
        try:
            hdul = fits.open(download_file(url,cache=True,show_progress=True,timeout=120))
        except:
            print ('Download failed.')
            hdul=None
 
    else:
        sel = (imgTable['proctype'] == 'Stack') & (imgTable['prodtype'] == 'image') # basic selection
        df = imgTable[sel] # select
        if (len(df)>0):
            print(f"Max_deepness: {np.max(df['magzero'].astype('float'))}")
            idx = df['magzero'].astype('float').idxmax()
            row = df.loc[idx, :]
            url = row['access_url']
            print ('downloading deepest ' + band + ' image...')
            try:
                hdul = fits.open(download_file(url,cache=True,show_progress=True,timeout=120))
            except:
                print ('Download failed.')
                hdul=None
        else:
            print ('No image available.')
            hdul=None
        
    return hdul

start, width = 0, 13_000
# for i, source_id in enumerate(list(gr.groups.keys())[start:start+width]):
for i, source_id in enumerate(list(gr.groups.keys())[start:start+width]):
    group = gr.get_group(source_id)
# for i, (source_id, group) in enumerate(gr[4000:]):
    imgTable = pd.read_csv(f'data/candidates/tables2/{source_id}.csv')
    print(f'Processing {source_id} ({i+start+1}/{len(gr)})')
    ra = group['ra_smss'].values[0]
    dec = group['dec_smss'].values[0]
    for band in 'gri':
        fits_name = f'data/candidates/fits2/{band}/{source_id}_{band}.fits'
        if os.path.exists(fits_name):
            print(f'Skipping {source_id} ({i+start+1}/{len(gr)}) as file already exists')
            continue
        
        hdul = download_deepest_image(ra, dec, svc=svc, imgTable=imgTable, band=band) # FOV in deg
        if hdul is None:
            continue
        hdul.writeto(fits_name, overwrite=True)
        hdul.close()