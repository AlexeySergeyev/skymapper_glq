# %%
import os
import numpy as np
import pandas as pd
import time

from dotenv import load_dotenv
load_dotenv()

from pyvo.dal import sia
from dl import authClient as ac

import warnings
warnings.filterwarnings("ignore")
# %%
token = ac.login(os.getenv('DATALAB_LOGIN'), os.getenv('DATALAB_PASSWORD'))
token
# %%
# The default endpoint points to the entire public Astro Data Archive
DEF_ACCESS_URL = "https://datalab.noirlab.edu/sia"
svc = sia.SIAService(DEF_ACCESS_URL)

df_candidates = pd.read_csv('data/candidates/candidates2.csv')
gr = df_candidates.groupby('source_id_smss')

fov = 30.0/3600 # in degrees
start, width = 12000, 2000
for i, source_id in enumerate(list(gr.groups.keys())[start:start+width]):
    group = gr.get_group(source_id)
    if i < 0:
        continue
    fname = f'data/candidates/tables2/{source_id}.csv'
    if os.path.exists(fname):
        print(f'Skipping {source_id} ({i+1}/{len(gr)}) as file already exists')
        continue

    print(f'Processing {source_id} ({start+ i+1}/{len(gr)})', time.ctime())
    ra = group['ra_smss'].values[0]
    dec = group['dec_smss'].values[0]
    imgTable = svc.search((ra,dec), (fov/np.cos(dec*np.pi/180), fov), verbosity=2).to_table()
    imgTable.to_pandas().to_csv(fname, index=False)